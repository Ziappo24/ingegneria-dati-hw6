import pandas as pd
import re
import os
import numpy as np

def clean_vin_strict(vin):
    if pd.isna(vin) or str(vin).lower() in ['nan', 'none', '']:
        return None
    # Rimuove tutto ciò che non è lettera o numero e mette in maiuscolo
    clean = re.sub(r'[^A-Z0-9]', '', str(vin).upper())
    
    # --- MODIFICA MIGLIORATA: Filtro Entropia ---
    # Un VIN deve avere 17 caratteri E non deve essere composto dallo stesso carattere ripetuto
    if len(clean) == 17 and len(set(clean)) > 1:
        return clean
    return None

def verify_match(row):
    try:
        year_diff = abs(row['year_cl'] - row['year_us'])
    except:
        year_diff = 999
        
    brand_match = str(row['make_cl']).strip().lower() == str(row['make_us']).strip().lower()
    return (year_diff <= 1) and brand_match

def generate_ground_truth():
    print("--- GENERAZIONE GROUND TRUTH ---")
    
    df_cl = pd.read_csv('data/processed/craigslist_aligned.csv')
    df_us = pd.read_csv('data/processed/us_cars_aligned.csv')
    
    df_cl['id_cl'] = df_cl.index
    df_us['id_us'] = df_us.index
    
    print("Pulizia VIN...")
    df_cl['vin_gt'] = df_cl['vin'].apply(clean_vin_strict)
    df_us['vin_gt'] = df_us['vin'].apply(clean_vin_strict)
    
    # --- MODIFICA MIGLIORATA: Rimozione Duplicati prima del Join ---
    # Se un'auto è postata 10 volte, prendiamo una sola occorrenza per la GT
    df_cl_unique = df_cl[['id_cl', 'vin_gt', 'make', 'model', 'year']].dropna(subset=['vin_gt']).drop_duplicates('vin_gt')
    df_us_unique = df_us[['id_us', 'vin_gt', 'make', 'model', 'year']].dropna(subset=['vin_gt']).drop_duplicates('vin_gt')
    
    print(f"Ricerca match via VIN su {len(df_cl_unique)} (CL) e {len(df_us_unique)} (US) record unici...")
    ground_truth_matches = pd.merge(
        df_cl_unique,
        df_us_unique,
        on='vin_gt',
        suffixes=('_cl', '_us')
    )
    
    print("Validazione ad-hoc dei match...")
    ground_truth_final = ground_truth_matches[ground_truth_matches.apply(verify_match, axis=1)].copy()
    ground_truth_final['label'] = 1
    
    print(f"Match trovati via VIN: {len(ground_truth_matches)}")
    print(f"Match validati dopo verifica ad-hoc: {len(ground_truth_final)}")
    
    os.makedirs('data/gt', exist_ok=True)
    gt_output = ground_truth_final[['id_cl', 'id_us', 'label']]
    gt_output.to_csv('data/gt/ground_truth.csv', index=False)
    print(f"✅ Ground Truth salvata in data/gt/ground_truth.csv")
    
    return ground_truth_final

if __name__ == "__main__":
    generate_ground_truth()