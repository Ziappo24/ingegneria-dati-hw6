import pandas as pd
import re
import os
import numpy as np
from tqdm import tqdm

# Inizializziamo tqdm per Pandas
tqdm.pandas()

def standardize_data(df):
    """ Standardizzazione generale delle colonne testuali. """
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.lower().str.strip()
        # Sostituzione diretta con pd.NA senza passare per stringhe segnaposto
        df[col] = df[col].replace(['nan', 'none', '', 'null', '<na>'], pd.NA)
    return df

def deep_clean(df):
    """ Pulizia specifica per brand e modelli. """
    brand_map = {
        'vw': 'volkswagen', 
        'chevy': 'chevrolet', 
        'mercedes-benz': 'mercedes',
        'mercedes benz': 'mercedes'
    }
    if 'make' in df.columns:
        df['make'] = df['make'].replace(brand_map)

    # Pulizia Modello: rimuove punteggiatura e spazi per un matching più forte
    if 'model' in df.columns:
        print("Pulizia stringhe modelli...")
        df['model'] = df['model'].progress_apply(
            lambda x: re.sub(r'[^a-z0-9]', '', str(x)) if pd.notna(x) else x
        )
    return df

def final_polish(df):
    """ Lucidatura finale: numerici, carburante e trasmissione. """
    
    # 1. Pulizia FUEL_TYPE (Mappatura diretta a null)
    if 'fuel_type' in df.columns:
        fuel_map = {'gasoline': 'gas', 'diesel': 'diesel', 'electric': 'electric', 'hybrid': 'hybrid'}
        df['fuel_type'] = df['fuel_type'].replace(fuel_map)
        # Riduciamo il rumore: se non è tra i principali, lo consideriamo nullo per il Record Linkage
        valid_fuels = ['gas', 'diesel', 'electric', 'hybrid']
        df.loc[~df['fuel_type'].isin(valid_fuels), 'fuel_type'] = pd.NA

    # 2. Pulizia TRANSMISSION
    if 'transmission' in df.columns:
        trans_map = {'auto': 'automatic', 'man': 'manual', 'mnd': 'manual'}
        df['transmission'] = df['transmission'].replace(trans_map)
        valid_trans = ['automatic', 'manual']
        df.loc[~df['transmission'].isin(valid_trans), 'transmission'] = pd.NA

    # 3. Pulizia PRICE e MILEAGE
    for col in ['price', 'mileage']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] <= 0, col] = np.nan

    # 4. Pulizia YEAR (Limite 2026)
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        # Filtro realistico aggiornato al 2026
        df.loc[(df['year'] < 1900) | (df['year'] > 2026), 'year'] = np.nan
        df['year'] = df['year'].astype('Int64')

    return df

def clean_vin_for_gt(vin):
    """ Pulizia VIN con filtro entropia (rimuove 000000...). """
    if pd.isna(vin) or str(vin).lower() in ['nan', 'none', '', 'null']: 
        return pd.NA
    # Regex standard VIN (rimuove I, O, Q non ammessi)
    clean = re.sub(r'[^A-HJ-NPR-Z0-9]', '', str(vin).upper())
    
    # MODIFICA SOTTOLINEATA: Un VIN valido ha 17 caratteri e non è tutto uguale
    if len(clean) == 17 and len(set(clean)) > 1:
        return clean
    return pd.NA

def load_and_map(file_path, mapping, source_name, **kwargs):
    """ Pipeline completa di caricamento e pulizia. """
    print(f"\n🚀 START PIPELINE: {source_name}")
    try:
        # Carichiamo solo le colonne che ci servono per risparmiare RAM
        needed_cols = list(mapping.keys())
        df = pd.read_csv(file_path, usecols=needed_cols, **kwargs)
        
        df_final = df.rename(columns=mapping)
        
        # Esecuzione pipeline
        df_final = standardize_data(df_final)
        df_final = deep_clean(df_final)
        df_final = final_polish(df_final)
        
        if 'vin' in df_final.columns:
            print(f"Validazione VIN per {source_name}...")
            df_final['vin_clean'] = df_final['vin'].progress_apply(clean_vin_for_gt)
        
        print(f"✅ {source_name} completato: {len(df_final)} righe.")
        return df_final
    except Exception as e:
        print(f"❌ Errore critico su {source_name}: {e}")
        return None

# --- CONFIGURAZIONE ---

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
    
    # Caricamento Craigslist (Full)
    df_cl = load_and_map('data/raw/craiglist/vehicles.csv', CRAIGSLIST_MAPPING, 'Craigslist')
    
    # Caricamento US Cars (Limitato per RAM, con dtype per stabilità)
    df_us = load_and_map(
        'data/raw/us_used_cars/used_cars_data.csv', 
        US_CARS_MAPPING, 
        'US Used Cars', 
        nrows=2000000, # Ridotto a 2M per sicurezza RAM, alza pure se il PC regge
        low_memory=False
    )
    
    # Salvataggio finale
    if df_cl is not None:
        df_cl.to_csv('data/processed/craigslist_aligned.csv', index=False)
        print("📁 Salvato: craigslist_aligned.csv")
        
    if df_us is not None:
        df_us.to_csv('data/processed/us_cars_aligned.csv', index=False)
        print("📁 Salvato: us_cars_aligned.csv")