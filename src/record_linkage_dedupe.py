import os
import csv
import re
import logging
import dedupe
import pandas as pd
import time

def train_dedupe(blocking_strategy='B1'):
    print(f"\n--- DEDUPE (ACTIVE LEARNING) - STRATEGIA: {blocking_strategy} ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(base_dir, 'data', 'results', f'matches_dedupe_{blocking_strategy}.csv')
    gt_train_path = os.path.join(base_dir, 'data', 'gt', 'gt_train.csv')
    cl_path = os.path.join(base_dir, 'data', 'processed', 'craigslist_final.csv')
    us_path = os.path.join(base_dir, 'data', 'processed', 'us_cars_final.csv')

    # 1. Caricamento dati
    print("Caricamento dati...")
    df_cl_full = pd.read_csv(cl_path, index_col='id_cl')
    df_us_full = pd.read_csv(us_path, index_col='id_us')
    
    # Per evitare NameError/ZeroDivisionError e garantire GT nel campione:
    # Prendiamo i primi 50 record dalla GT che esistono nei file
    gt_train = pd.read_csv(gt_train_path)
    gt_positive = gt_train[gt_train['label'] == 1].head(50)
    gt_negative = gt_train[gt_train['label'] == 0].head(50)
    
    id_cl_needed = set(gt_positive['id_cl']).union(set(gt_negative['id_cl']))
    id_us_needed = set(gt_positive['id_us']).union(set(gt_negative['id_us']))
    
    # Filtriamo solo quelli che esistono davvero nei dataset
    id_cl_needed = [i for i in id_cl_needed if i in df_cl_full.index]
    id_us_needed = [i for i in id_us_needed if i in df_us_full.index]

    # Campionamento aggiuntivo (totale 1000 per evitare MemoryError)
    SAMPLE_SIZE = 1000
    df_cl_sample = df_cl_full.sample(n=min(SAMPLE_SIZE - len(id_cl_needed), len(df_cl_full)), random_state=42)
    df_us_sample = df_us_full.sample(n=min(SAMPLE_SIZE - len(id_us_needed), len(df_us_full)), random_state=42)
    
    df_cl = pd.concat([df_cl_full.loc[id_cl_needed], df_cl_sample])
    df_us = pd.concat([df_us_full.loc[id_us_needed], df_us_sample])

    # Pulizia stringhe: Dedupe fallisce (ZeroDivisionError) se riceve stringhe vuote in alcuni campi
    def clean_val(v):
        if pd.isna(v) or str(v).strip().lower() in ['nan', '']:
            return "unknown" # Valore di fallback per evitare stringhe vuote
        return str(v).strip()

    def to_dict(df):
        return {str(i): {k: clean_val(v) for k, v in row.items()} for i, row in df.iterrows()}
    
    data_1 = to_dict(df_cl)
    data_2 = to_dict(df_us)

    # 2. Definizione campi
    import dedupe.variables
    fields = [
        dedupe.variables.String('make'),
        dedupe.variables.String('model'),
        dedupe.variables.Exact('year'),
        dedupe.variables.String('fuel_type'),
        dedupe.variables.String('transmission')
    ]

    # 3. Inizializzazione Dedupe
    linker = dedupe.RecordLink(fields)

    # 4. Semiautomatic Training
    matches = []
    distincts = []
    for _, row in gt_train.iterrows():
        id_cl, id_us = str(int(row['id_cl'])), str(int(row['id_us']))
        if id_cl in data_1 and id_us in data_2:
            example = (data_1[id_cl], data_2[id_us])
            if int(row['label']) == 1:
                matches.append(example)
            else:
                distincts.append(example)

    if matches or distincts:
        print(f"Pre-addestramento con {len(matches)} positivi e {len(distincts)} negativi...")
        linker.mark_pairs({'match': matches, 'distinct': distincts})

    print("Preparazione training...")
    linker.prepare_training(data_1, data_2, sample_size=2000)

    print("Train del modello...")
    linker.train(recall=0.90)

    # 5. Clustering
    print("Clustering dei record...")
    linked_records = linker.join(data_1, data_2, threshold=0.5)
    print(f"Match individuati: {len(linked_records)}")
    
    # 6. Salvataggio
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['cl_id', 'us_id', 'confidence'])
        for (cl_id, us_id), score in linked_records:
            writer.writerow([cl_id, us_id, score])

    print(f"Risultati salvati in: {output_file}")

if __name__ == "__main__":
    train_dedupe('B1')
