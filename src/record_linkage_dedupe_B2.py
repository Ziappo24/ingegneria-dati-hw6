import os
import csv
import json
import logging
import dedupe
import pandas as pd
import numpy as np
from dedupe import variables

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_dedupe_B2():
    print(f"\n--- DEDUPE B2: TURBO MODE CON VINCOLO MATCH (Fix Hang) ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'data', 'results')
    model_dir = os.path.join(base_dir, 'data', 'models')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'matches_dedupe_B2.csv')
    settings_file = os.path.join(model_dir, 'dedupe_settings_B2.bin')
    training_file = os.path.join(model_dir, 'dedupe_training_B2.json')
    training_lite_file = os.path.join(model_dir, 'dedupe_training_B2_lite.json')
    
    gt_train_path = os.path.join(base_dir, 'data', 'gt', 'gt_train.csv')
    cl_path = os.path.join(base_dir, 'data', 'processed', 'craigslist_final.csv')
    us_path = os.path.join(base_dir, 'data', 'processed', 'us_cars_final.csv')

    # 1. DEFINIZIONE CAMPI (Tutto ShortString per velocità)
    fields = [
        variables.ShortString('make'),
        variables.ShortString('model'),
        variables.ShortString('brand_model'), 
        variables.Price('year', has_missing=True),
        variables.ShortString('body_type', has_missing=True),
        variables.ShortString('fuel_type', has_missing=True),
        variables.ShortString('transmission', has_missing=True)
    ]

    linker = dedupe.RecordLink(fields, num_cores=4)

    def clean_val(v, is_year=False):
        if pd.isna(v) or str(v).strip().lower() in ['nan', '', 'unknown']:
            return None
        return float(v) if is_year else str(v).strip()

    # 2. CARICAMENTO E CAMPIONAMENTO INTELLIGENTE
    print("Caricamento dataset e protezione training set...")
    df_cl_raw = pd.read_csv(cl_path, index_col='id_cl')
    df_us_raw = pd.read_csv(us_path, index_col='id_us')
    
    for df in [df_cl_raw, df_us_raw]:
        df['brand_model'] = df['make'].fillna('') + " " + df['model'].fillna('')

    gt_train = pd.read_csv(gt_train_path)
    needed_cl = set(gt_train['id_cl'].astype(str))
    needed_us = set(gt_train['id_us'].astype(str))

    def safe_sample(df, needed_ids, n_random=2500):
        present_needed = [idx for idx in df.index if str(idx) in needed_ids]
        df_needed = df.loc[present_needed]
        remaining = df.index.difference(df_needed.index)
        df_random = df.loc[np.random.choice(remaining, min(n_random, len(remaining)), replace=False)]
        return pd.concat([df_needed, df_random])

    df_cl_sample = safe_sample(df_cl_raw, needed_cl)
    df_us_sample = safe_sample(df_us_raw, needed_us)

    data_1 = {str(i): {k: clean_val(v, k=='year') for k, v in r.items() if k in [f.field for f in fields]} for i, r in df_cl_sample.iterrows()}
    data_2 = {str(i): {k: clean_val(v, k=='year') for k, v in r.items() if k in [f.field for f in fields]} for i, r in df_us_sample.iterrows()}

    # 3. GESTIONE TRAINING CON VINCOLO (Capping Match)
    if os.path.exists(settings_file):
        print("Modello esistente trovato. Caricamento...")
        with open(settings_file, 'rb') as f:
            linker = dedupe.StaticRecordLink(f, num_cores=4)
    else:
        print(f"Preparazione training set completo...")
        training_points = {'match': [], 'distinct': []}
        for _, row in gt_train.iterrows():
            id_cl, id_us = str(row['id_cl']), str(row['id_us'])
            if id_cl in data_1 and id_us in data_2:
                label = 'match' if row['label'] == 1 else 'distinct'
                training_points[label].append((data_1[id_cl], data_2[id_us]))

        with open(training_file, 'w') as tf:
            json.dump(training_points, tf)

        # IL VINCOLO: Creiamo un file "Lite" solo per il Blocking
        # Prendiamo massimo 300 match per non far bloccare la ricerca delle regole
        print(f"Creazione set ridotto per Blocking (Capping a 300 match)...")
        blocking_lite = {
            'match': training_points['match'][:300], 
            'distinct': training_points['distinct'][:300]
        }
        with open(training_lite_file, 'w') as tfl:
            json.dump(blocking_lite, tfl)

        print("Ricerca regole di blocking (Sample size: 800)...")
        with open(training_lite_file, 'r') as tf_lite:
            linker.prepare_training(data_1, data_2, training_file=tf_lite, sample_size=800)
        
        print("Addestramento pesi (con training completo)...")
        linker.train(recall=0.85)
        with open(settings_file, 'wb') as sf:
            linker.write_settings(sf)

    # 4. MATCHING FINALE
    print("Avvio Matching B2 rapido...")
    data_1_f = {str(i): {k: clean_val(v, k=='year') for k, v in r.items() if k in [f.field for f in fields]} 
                for i, r in df_cl_raw.sample(n=min(5000, len(df_cl_raw))).iterrows()}
    data_2_f = {str(i): {k: clean_val(v, k=='year') for k, v in r.items() if k in [f.field for f in fields]} 
                for i, r in df_us_raw.sample(n=min(5000, len(df_us_raw))).iterrows()}

    linked_records = linker.join(data_1_f, data_2_f, threshold=0.35)
    
    # 5. SALVATAGGIO
    df_res = pd.DataFrame([(cl, us, conf) for (cl, us), conf in linked_records], columns=['cl_id', 'us_id', 'confidence'])
    if not df_res.empty:
        df_res = df_res.sort_values(by='confidence', ascending=False)
        df_res = df_res.drop_duplicates(subset=['cl_id'], keep='first').drop_duplicates(subset=['us_id'], keep='first')
    
    df_res.to_csv(output_file, index=False)
    print(f"✅ B2 Completato! Risultati salvati in: {output_file}")

if __name__ == "__main__":
    train_dedupe_B2()