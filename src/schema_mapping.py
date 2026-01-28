import pandas as pd
import os
import re

def standardize_data(df):
    """
    Standardizzazione generale di tutte le colonne testuali.
    """
    for col in df.select_dtypes(include=['object']).columns:
        # 1. Minuscolo e rimozione spazi ai bordi
        df[col] = df[col].astype(str).str.lower().str.strip()
        # 2. Sostituzione di stringhe vuote o segnaposti con null reale
        df[col] = df[col].replace(['nan', 'none', '', 'null', 'nan'], pd.NA)
    return df

def deep_clean(df):
    """
    Pulizia profonda specifica per migliorare il record linkage.
    """
    # Normalizzazione Brand (usiamo 'make' come nome colonna nello schema mediato)
    brand_map = {
        'vw': 'volkswagen', 
        'chevy': 'chevrolet', 
        'mercedes-benz': 'mercedes',
        'mercedes benz': 'mercedes'
    }
    if 'make' in df.columns:
        df['make'] = df['make'].replace(brand_map)

    # Pulizia Modello: rimuove punteggiatura e spazi extra
    if 'model' in df.columns:
        df['model'] = df['model'].apply(lambda x: re.sub(r'[^a-z0-9]', '', str(x)) if pd.notna(x) else x)

    # Gestione Body Type: standardizziamo i nulli per stabilità
    if 'body_type' in df.columns:
        df['body_type'] = df['body_type'].fillna('unknown')
        # Semplificazione nomi carrozzeria se necessario (es. suv / crossover -> suv)
        df['body_type'] = df['body_type'].str.split('/').str[0].str.strip()
    
    return df

def final_polish(df):
    """
    Lucidatura finale: carburante, trasmissione, prezzi, km e anni.
    """
    import numpy as np
    
    # 1. Pulizia FUEL_TYPE
    if 'fuel_type' in df.columns:
        df['fuel_type'] = df['fuel_type'].astype(str).str.lower().str.strip()
        fuel_map = {'gasoline': 'gas', 'diesel': 'diesel', 'electric': 'electric', 'hybrid': 'hybrid'}
        df['fuel_type'] = df['fuel_type'].replace(fuel_map)
        df['fuel_type'] = df['fuel_type'].replace(['nan', 'none', 'other', '', '<na>'], 'unknown')

    # 2. Pulizia TRANSMISSION
    if 'transmission' in df.columns:
        df['transmission'] = df['transmission'].astype(str).str.lower().str.strip()
        trans_map = {'auto': 'automatic', 'man': 'manual', 'mnd': 'manual'}
        df['transmission'] = df['transmission'].replace(trans_map)
        df['transmission'] = df['transmission'].replace(['nan', 'none', 'other', '', '<na>'], 'unknown')

    # 3. Pulizia PRICE e MILEAGE (Numerici)
    for col in ['price', 'mileage']:
        if col in df.columns:
            # Rimuove tutto ciò che non è un numero o punto decimale
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Outliers: < 0 o 0 trattati come nulli
            df.loc[df[col] <= 0, col] = np.nan

    # 4. Pulizia YEAR
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        # Filtro realistico per le auto
        df.loc[(df['year'] < 1900) | (df['year'] > 2026), 'year'] = np.nan
        df['year'] = df['year'].astype('Int64')

    return df

def clean_vin_for_gt(vin):
    """
    Pulizia del VIN per la creazione della Ground Truth.
    """
    if pd.isna(vin) or str(vin).lower() in ['nan', 'none', '']: 
        return pd.NA
    # Rimuove caratteri non alfanumerici e rende maiuscolo
    clean = re.sub(r'[^A-HJ-NPR-Z0-9]', '', str(vin).upper())
    # Un VIN valido deve avere 17 caratteri
    return clean if len(clean) == 17 else pd.NA

def handle_placeholders(df):
    """
    Sostituzione segnaposto con nulli reali.
    """
    import numpy as np
    # 1. Torna ai nulli reali per le stringhe
    df = df.replace(['unknown', 'nan', 'none', '', '<na>'], np.nan)
    
    # 2. Gestione zeri nei campi numerici
    cols_no_zero = ['price', 'mileage']
    for col in cols_no_zero:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    return df

def load_and_map(file_path, mapping, source_name, **kwargs):
    """
    Carica, mappa, standardizza e pulisce un dataset con pipeline avanzata.
    """
    print(f"\n--- ELABORAZIONE AVANZATA SCHEMA: {source_name} ---")
    try:
        df = pd.read_csv(file_path, **kwargs)
        
        # Mapping
        df_mapped = df.rename(columns=mapping)
        
        # Selezione colonne
        mediated_columns = list(set(mapping.values()))
        available_columns = [col for col in mediated_columns if col in df_mapped.columns]
        df_final = df_mapped[available_columns].copy()
        
        # Pipeline di pulizia
        df_final = standardize_data(df_final)
        df_final = deep_clean(df_final)
        df_final = final_polish(df_final)
        
        # Pulizia VIN per Ground Truth
        if 'vin' in df_final.columns:
            df_final['vin_clean'] = df_final['vin'].apply(clean_vin_for_gt)
        
        # Gestione finale segnaposto
        df_final = handle_placeholders(df_final)
        
        print(f"Dataset '{source_name}' pronto. Righe: {len(df_final)}")
        return df_final
    except Exception as e:
        print(f"Errore durante l'elaborazione di {source_name}: {e}")
        return None

# Mapping aggiornati (rimosso 'city' e 'region')
CRAIGSLIST_MAPPING = {
    'VIN': 'vin',
    'manufacturer': 'make',
    'model': 'model',
    'year': 'year',
    'price': 'price',
    'odometer': 'mileage',
    'fuel': 'fuel_type',
    'transmission': 'transmission',
    'type': 'body_type'
}

US_CARS_MAPPING = {
    'vin': 'vin',
    'make_name': 'make',
    'model_name': 'model',
    'year': 'year',
    'price': 'price',
    'mileage': 'mileage',
    'fuel_type': 'fuel_type',
    'transmission': 'transmission',
    'body_type': 'body_type'
}

if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)
    
    # Campione per test (1 milione di righe)
    params = {'nrows': 1000000}
    
    # Elaborazione
    df_cl = load_and_map('data/raw/craiglist/vehicles.csv', CRAIGSLIST_MAPPING, 'Craigslist', **params)
    df_us = load_and_map('data/raw/us_used_cars/used_cars_data.csv', US_CARS_MAPPING, 'US Used Cars', **params, dtype={'dealer_zip': str, 'bed': str})
    
    # Salvataggio
    if df_cl is not None:
        df_cl.to_csv('data/processed/craigslist_aligned.csv', index=False)
        print("Salvato: data/processed/craigslist_aligned.csv")
        
    if df_us is not None:
        df_us.to_csv('data/processed/us_cars_aligned.csv', index=False)
        print("Salvato: data/processed/us_cars_aligned.csv")
