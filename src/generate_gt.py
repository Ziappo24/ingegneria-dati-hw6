import pandas as pd
import re
import os
import numpy as np

def clean_vin_strict(vin):
    if pd.isna(vin) or str(vin).lower() in ['nan', 'none', '']:
        return None
    # Rimuove tutto ciò che non è lettera o numero e mette in maiuscolo
    clean = re.sub(r'[^A-Z0-9]', '', str(vin).upper())
    # Un VIN valido deve essere di 17 caratteri
    return clean if len(clean) == 17 else None

def verify_match(row):
    # Tolleranza di 1 anno per piccoli errori di inserimento
    try:
        year_diff = abs(row['year_cl'] - row['year_us'])
    except:
        year_diff = 999
        
    brand_match = str(row['make_cl']).strip() == str(row['make_us']).strip()
    return (year_diff <= 1) and brand_match

def generate_ground_truth():
    print("--- GENERAZIONE GROUND TRUTH ---")
    
    # 1. Caricamento dataset allineati
    df_cl = pd.read_csv('data/processed/craigslist_aligned.csv')
    df_us = pd.read_csv('data/processed/us_cars_aligned.csv')
    
    # Creiamo ID univoci
    df_cl['id_cl'] = df_cl.index
    df_us['id_us'] = df_us.index
    
    # 2. Pulizia VIN rigorosa per Ground Truth
    print("Pulizia VIN...")
    df_cl['vin_gt'] = df_cl['vin'].apply(clean_vin_strict)
    df_us['vin_gt'] = df_us['vin'].apply(clean_vin_strict)
    
    # 3. Join per trovare i match (coppie positive)
    print("Ricerca match via VIN...")
    ground_truth_matches = pd.merge(
        df_cl[['id_cl', 'vin_gt', 'make', 'model', 'year']].dropna(subset=['vin_gt']),
        df_us[['id_us', 'vin_gt', 'make', 'model', 'year']].dropna(subset=['vin_gt']),
        on='vin_gt',
        suffixes=('_cl', '_us')
    )
    
    # 4. Validazione ad-hoc (Marca e Anno)
    print("Validazione ad-hoc dei match...")
    ground_truth_final = ground_truth_matches[ground_truth_matches.apply(verify_match, axis=1)].copy()
    
    ground_truth_final['label'] = 1
    
    print(f"Match trovati via VIN: {len(ground_truth_matches)}")
    print(f"Match validati dopo verifica ad-hoc: {len(ground_truth_final)}")
    
    # 5. Salvataggio Ground Truth (solo ID e label)
    os.makedirs('data/gt', exist_ok=True)
    gt_output = ground_truth_final[['id_cl', 'id_us', 'label']]
    gt_output.to_csv('data/gt/ground_truth.csv', index=False)
    print(f"Ground Truth salvata in data/gt/ground_truth.csv")
    
    return ground_truth_final

if __name__ == "__main__":
    generate_ground_truth()
