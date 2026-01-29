import pandas as pd
import recordlinkage
import time
import os

def record_linkage_rules(blocking_strategy='B1'):
    print(f"\n--- RECORD LINKAGE (RULES) - STRATEGIA: {blocking_strategy} ---")
    
    # 1. Caricamento dataset con campionamento per gestire la memoria
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cl_path = os.path.join(base_dir, 'data', 'processed', 'craigslist_final.csv')
    us_path = os.path.join(base_dir, 'data', 'processed', 'us_cars_final.csv')

    print(f"Caricamento dataset da: {cl_path} e {us_path}...")
    try:
        df_cl = pd.read_csv(cl_path, index_col='id_cl')
        df_us = pd.read_csv(us_path, index_col='id_us')
        
        # Campionamento per efficienza locale
        SAMPLE_SIZE = 50000
        df_cl = df_cl.sample(n=min(SAMPLE_SIZE, len(df_cl)), random_state=42)
        df_us = df_us.sample(n=min(SAMPLE_SIZE, len(df_us)), random_state=42)
        print(f"Dataset caricati. Craigslist: {df_cl.shape}, US Cars: {df_us.shape}")
    except FileNotFoundError as e:
        print(f"ERRORE: File non trovato. {e}")
        return
    
    # 2. Definizione Blocking
    start_time = time.time()
    indexer = recordlinkage.Index()
    
    if blocking_strategy == 'B1':
        indexer.block(['make', 'year'])
    elif blocking_strategy == 'B2':
        indexer.block(['year', 'body_type'])
    
    print("Indicizzazione candidati...")
    candidate_links = indexer.index(df_cl, df_us)
    print(f"Candidate links trovati: {len(candidate_links)}")
    
    # 3. Confronto
    compare_cl = recordlinkage.Compare()
    compare_cl.string('model', 'model', method='jarowinkler', threshold=0.85, label='model')
    compare_cl.exact('fuel_type', 'fuel_type', label='fuel')
    compare_cl.exact('transmission', 'transmission', label='transmission')
    
    print("Calcolo similitudini (compute)...")
    features = compare_cl.compute(candidate_links, df_cl, df_us)
    
    # 4. Classificazione (Regola)
    features['total_score'] = (features['model'] * 3 + 
                               features['fuel'] * 0.5 + 
                               features['transmission'] * 0.5)
    
    matches = features[features['total_score'] >= 3.0]
    end_time = time.time()
    
    print(f"Match identificati: {len(matches)}")
    print(f"Tempo di esecuzione: {end_time - start_time:.2f}s")
    
    # Salvataggio risultati
    results_dir = os.path.join(base_dir, 'data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f'matches_rl_{blocking_strategy}.csv')
    matches.to_csv(out_path)
    print(f"Risultati salvati in: {out_path}")
    
    return matches

if __name__ == "__main__":
    record_linkage_rules('B1')
    record_linkage_rules('B2')
