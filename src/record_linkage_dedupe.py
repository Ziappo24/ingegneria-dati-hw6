import os
import csv
import json
import logging
import dedupe
import pandas as pd
import time
import gc
import numpy as np
from tqdm import tqdm
from dedupe import variables

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('dedupe')

def train_dedupe(blocking_strategy='B1'):
    print(f"\n--- DEDUPE EXTREME EVOLUTION - STRATEGIA: {blocking_strategy} ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'data', 'results')
    model_dir = os.path.join(base_dir, 'data', 'models')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'matches_dedupe_{blocking_strategy}.csv')
    settings_file = os.path.join(model_dir, f'dedupe_settings_{blocking_strategy}.bin')
    training_file = os.path.join(model_dir, f'dedupe_training_{blocking_strategy}.json')
    
    gt_train_path = os.path.join(base_dir, 'data', 'gt', 'gt_train.csv')
    cl_path = os.path.join(base_dir, 'data', 'processed', 'craigslist_final.csv')
    us_path = os.path.join(base_dir, 'data', 'processed', 'us_cars_final.csv')

    # 1. DEFINIZIONE CAMPI (Sostituzione Interaction con Brand_Model Sintetico)
    fields = [
        variables.String('make'),
        variables.String('model'),
        # MODIFICA: Campo sintetico per risolvere l'errore Interaction
        variables.String('brand_model'), 
        # MODIFICA: Price per gestire differenze d'anno (es. 2018 vs 2019)
        variables.Price('year', has_missing=True),  
        variables.ShortString('fuel_type', has_missing=True),
        variables.ShortString('transmission', has_missing=True)
    ]

    linker = dedupe.RecordLink(fields, num_cores=1)

    def clean_val(v, is_year=False):
        if pd.isna(v) or str(v).strip().lower() in ['nan', '', 'unknown']:
            return None
        if is_year:
            try: return float(v) 
            except: return None
        return str(v).strip()

    # 2. CARICAMENTO E SINTESI DATI
    print("Caricamento e creazione campo sintetico brand_model...")
    df_cl_raw = pd.read_csv(cl_path, index_col='id_cl')
    df_us_raw = pd.read_csv(us_path, index_col='id_us')

    # Creazione campo brand_model
    for df in [df_cl_raw, df_us_raw]:
        df['brand_model'] = df['make'].fillna('') + " " + df['model'].fillna('')

    # Campionamento per prepare_training
    SAMPLE_SIZE = 2000
    df_cl_sample = df_cl_raw.sample(n=min(SAMPLE_SIZE, len(df_cl_raw)), random_state=42)
    df_us_sample = df_us_raw.sample(n=min(SAMPLE_SIZE, len(df_us_raw)), random_state=42)

    data_1 = {str(i): {k: clean_val(v, k=='year') for k, v in r.items() if k in [f.field for f in fields]} 
              for i, r in df_cl_sample.iterrows()}
    data_2 = {str(i): {k: clean_val(v, k=='year') for k, v in r.items() if k in [f.field for f in fields]} 
              for i, r in df_us_sample.iterrows()}

    # 3. TRAINING OTTIMIZZATO (Target 1500)
    if os.path.exists(settings_file):
        print(f"Caricamento modello pre-addestrato...")
        with open(settings_file, 'rb') as f:
            linker = dedupe.StaticRecordLink(f, num_cores=1)
    else:
        print("Avvio addestramento con target Recall 95%...")
        gt_train = pd.read_csv(gt_train_path)
        
        # MODIFICA: Training bilanciato a 1500 per evitare rigidità
        target_size = 750 
        gt_matches = gt_train[gt_train['label'] == 1].sample(n=min(target_size, len(gt_train[gt_train['label'] == 1])), random_state=42)
        gt_distincts = gt_train[gt_train['label'] == 0].sample(n=min(target_size, len(gt_train[gt_train['label'] == 0])), random_state=42)
        gt_train_sub = pd.concat([gt_matches, gt_distincts])

        training_points = {'match': [], 'distinct': []}
        for _, row in tqdm(gt_train_sub.iterrows(), total=len(gt_train_sub), desc="Training"):
            id_cl, id_us = int(row['id_cl']), int(row['id_us'])
            if id_cl in df_cl_raw.index and id_us in df_us_raw.index:
                r_cl, r_us = df_cl_raw.loc[id_cl], df_us_raw.loc[id_us]
                # Creiamo i record includendo il brand_model
                rec_a = {k: clean_val(v, k=='year') for k, v in r_cl.items() if k in [f.field for f in fields]}
                rec_b = {k: clean_val(v, k=='year') for k, v in r_us.items() if k in [f.field for f in fields]}
                
                if int(row['label']) == 1: training_points['match'].append([rec_a, rec_b])
                else: training_points['distinct'].append([rec_a, rec_b])

        with open(training_file, 'w') as tf: json.dump(training_points, tf)
        
        # MODIFICA: sample_size aumentato a 5000 per migliorare il blocking
        linker.prepare_training(data_1, data_2, training_file=training_file, sample_size=5000)
        # MODIFICA: Forza la Recall al 95%
        linker.train(recall=0.95) 
        with open(settings_file, 'wb') as sf: linker.write_settings(sf)

    # 4. MATCHING AGGRESSIVO
    print("Preparazione dati per il Matching finale...")
    gt_test = pd.read_csv(os.path.join(base_dir, 'data', 'gt', 'gt_test.csv'))
    test_cl_ids = set(gt_test['id_cl'].unique())
    test_us_ids = set(gt_test['id_us'].unique())

    # Prepariamo i dataset filtrati per il matching
    BUFFER_SIZE = 30000 
    def get_scoped_data(df, test_ids, buffer):
        needed = list(test_ids.intersection(set(df.index)))
        extra = df.index.difference(needed)
        selected_extra = np.random.choice(extra, min(buffer, len(extra)), replace=False)
        return df.loc[list(needed) + list(selected_extra)]

    df_cl_filtered = get_scoped_data(df_cl_raw, test_cl_ids, BUFFER_SIZE)
    df_us_filtered = get_scoped_data(df_us_raw, test_us_ids, BUFFER_SIZE)

    data_1_final = {str(i): {k: clean_val(v, k=='year') for k, v in r.items() if k in [f.field for f in fields]} 
                    for i, r in df_cl_filtered.iterrows()}
    data_2_final = {str(i): {k: clean_val(v, k=='year') for k, v in r.items() if k in [f.field for f in fields]} 
                    for i, r in df_us_filtered.iterrows()}

    print(f"Avvio matching aggressivo (Threshold: 0.35)...")
    linked_records = linker.join(data_1_final, data_2_final, threshold=0.35)
    
    # 5. RAFFINAMENTO 1:1 (PROTEZIONE PRECISIONE)
    print("Filtraggio 1:1 e salvataggio risultati...")
    df_res = pd.DataFrame([(cl, us, conf) for (cl, us), conf in linked_records], columns=['cl_id', 'us_id', 'confidence'])
    
    if not df_res.empty:
        df_res = df_res.sort_values(by='confidence', ascending=False)
        df_res = df_res.drop_duplicates(subset=['cl_id'], keep='first')
        df_res = df_res.drop_duplicates(subset=['us_id'], keep='first')
    
    df_res.to_csv(output_file, index=False)
    print(f"✅ Completato! Match validati: {len(df_res)}")

if __name__ == "__main__":
    train_dedupe('B1')