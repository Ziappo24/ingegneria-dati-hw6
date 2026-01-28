import os
import csv
import re
import logging
import optparse
import dedupe
import pandas as pd
import time

def train_dedupe(blocking_strategy='B1'):
    print(f"\n--- DEDUPE (ACTIVE LEARNING) - STRATEGIA: {blocking_strategy} ---")
    
    output_file = f'data/results/matches_dedupe_{blocking_strategy}.csv'
    settings_file = f'data/results/dedupe_settings_{blocking_strategy}'
    training_file = f'data/results/dedupe_training_{blocking_strategy}.json'

    # 1. Caricamento dati e campionamento ridotto per velocità
    print("Caricamento dati...")
    df_cl = pd.read_csv('data/processed/craigslist_final.csv', index_col='id_cl').sample(n=5000, random_state=42)
    df_us = pd.read_csv('data/processed/us_cars_final.csv', index_col='id_us').sample(n=5000, random_state=42)
    
    # Conversione in dizionario per dedupe (forzando stringhe e rimuovendo NaN)
    def to_dict(df):
        return {str(i): {k: str(v) if pd.notna(v) and str(v).lower() != 'nan' else "" for k, v in row.items()} for i, row in df.iterrows()}
    
    data_1 = to_dict(df_cl)
    data_2 = to_dict(df_us)

    # 2. Definizione campi (Nuova sintassi Dedupe 3.0)
    import dedupe.variables
    
    fields = [
        dedupe.variables.String('make'),
        dedupe.variables.String('model'),
        dedupe.variables.Exact('year'),
        # Usiamo String invece di Categorical perché i dati sono rumorosi (es. "a", "", "unknown")
        dedupe.variables.String('fuel_type'),
        dedupe.variables.String('transmission')
    ]

    # 3. Inizializzazione Dedupe
    linker = dedupe.RecordLink(fields)

    # 4. Addestramento semiautomatico (Active Learning)
    # Se abbiamo già una ground truth, la usiamo per pre-addestrare
    gt_train = pd.read_csv('data/gt/gt_train.csv')
    
    # Filtriamo la GT per i record presenti nel campione
    matches = []
    for _, row in gt_train.iterrows():
        id_cl, id_us = str(int(row['id_cl'])), str(int(row['id_us']))
        if id_cl in data_1 and id_us in data_2:
            matches.append((data_1[id_cl], data_2[id_us]))

    if matches:
        print(f"Pre-addestramento con {len(matches)} coppie dalla Ground Truth...")
        linker.mark_pairs({'match': matches, 'distinct': []})

    # Dedupe 3.0 richiede prepare_training prima di train
    print("Preparazione training...")
    linker.prepare_training(data_1, data_2, sample_size=15000)

    print("Train del modello...")
    linker.train(recall=0.90)

    # 5. Blocking e Clustering
    print("Clustering dei record...")
    linked_records = linker.join(data_1, data_2, threshold=0.5)

    print(f"Match individuati: {len(linked_records)}")
    
    # 6. Salvataggio
    os.makedirs('data/results', exist_ok=True)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cl_id', 'us_id', 'confidence'])
        for (cl_id, us_id), score in linked_records:
            writer.writerow([cl_id, us_id, score])

    print(f"Risultati salvati in: {output_file}")

if __name__ == "__main__":
    # Eseguiamo per B1 (blocking gestito internamente da dedupe basandosi sui campi)
    train_dedupe('B1')
