import pandas as pd
import recordlinkage
import time
import os
import gc

def record_linkage_rules(blocking_strategy='B1'):
    print(f"\n--- RECORD LINKAGE (RULES) - STRATEGIA: {blocking_strategy} ---")
    
    # Percorsi file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    results_dir = os.path.join(base_dir, 'data', 'results')
    gt_path = os.path.join(base_dir, 'data', 'gt', 'ground_truth.csv')

    cl_path = os.path.join(processed_dir, 'craigslist_final.csv')
    us_path = os.path.join(processed_dir, 'us_cars_final.csv')

    print(f"Caricamento dataset...")
    try:
        df_cl_full = pd.read_csv(cl_path, index_col='id_cl')
        df_us_full = pd.read_csv(us_path, index_col='id_us')
        
        # --- LOGICA INCLUSIONE GT (Per garantire la valutabilità) ---
        if os.path.exists(gt_path):
            gt = pd.read_csv(gt_path)
            id_cl_needed = set(gt['id_cl']).intersection(set(df_cl_full.index))
            id_us_needed = set(gt['id_us']).intersection(set(df_us_full.index))
        else:
            id_cl_needed, id_us_needed = set(), set()
        
        SAMPLE_SIZE = 70000
        cl_gt = df_cl_full.loc[list(id_cl_needed)]
        cl_rand = df_cl_full.drop(index=list(id_cl_needed)).sample(n=min(SAMPLE_SIZE-len(cl_gt), len(df_cl_full)-len(cl_gt)), random_state=42)
        df_cl = pd.concat([cl_gt, cl_rand])
        
        us_gt = df_us_full.loc[list(id_us_needed)]
        us_rand = df_us_full.drop(index=list(id_us_needed)).sample(n=min(SAMPLE_SIZE-len(us_gt), len(df_us_full)-len(us_gt)), random_state=42)
        df_us = pd.concat([us_gt, us_rand])
        
    except Exception as e:
        print(f"❌ ERRORE CARICAMENTO: {e}")
        return
    
    # 2. DEFINIZIONE BLOCKING
    start_time = time.time()
    indexer = recordlinkage.Index()
    
    if blocking_strategy == 'B1':
        indexer.block(['make', 'year'])
    elif blocking_strategy == 'B2':
        indexer.block(['make', 'year', 'body_type'])
    
    print(f"Indicizzazione candidati...")
    candidate_links = indexer.index(df_cl, df_us)
    print(f"Candidate links trovati: {len(candidate_links)}")
    
    # 3. CONFRONTO (Inizio Inferenza)
    start_inference = time.time()
    compare_cl = recordlinkage.Compare()
    compare_cl.string('model', 'model', method='jarowinkler', threshold=0.92, label='model')
    compare_cl.exact('fuel_type', 'fuel_type', label='fuel')
    compare_cl.exact('transmission', 'transmission', label='transmission')
    
    print("Calcolo similitudini...")
    features = compare_cl.compute(candidate_links, df_cl, df_us)
    
    # 4. CLASSIFICAZIONE E FILTRAGGIO 1:1
    features['total_score'] = (features['model'] * 4.0 + 
                               features['fuel'] * 0.5 + 
                               features['transmission'] * 0.5)
    
    # SOGLIA DI QUALITÀ
    potential_matches = features[features['total_score'] >= 4.5].reset_index()
    potential_matches.rename(columns={'level_0': 'id_cl', 'level_1': 'id_us'}, inplace=True)

    # --- SOTTOLINEATO: LOGICA ONE-TO-ONE MATCHING ---
    print("Raffinamento match (Vincolo di Unicità 1:1)...")
    # Ordiniamo per punteggio decrescente
    matches_sorted = potential_matches.sort_values(by='total_score', ascending=False)
    
    # Rimuoviamo i duplicati: ogni auto di CL può matchare solo 1 auto di US e viceversa
    matches = matches_sorted.drop_duplicates(subset=['id_cl'], keep='first')
    matches = matches.drop_duplicates(subset=['id_us'], keep='first')
    
    end_time = time.time()
    
    print(f"✅ Match validati (1:1): {len(matches)}")
    
    # Calcolo tempi
    training_time = 0.0 # Rule-based non ha addestramento
    inference_time = end_time - start_inference
    total_time = end_time - start_time
    
    print(f"⏱️ Tempo Addestramento: {training_time:.4f}s")
    print(f"⏱️ Tempo Inferenza: {inference_time:.4f}s")
    print(f"⏱️ Tempo Totale: {total_time:.4f}s")
    
    # 5. SALVATAGGIO
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f'matches_rl_{blocking_strategy}.csv')
    matches[['id_cl', 'id_us', 'total_score']].to_csv(out_path, index=False)
    
    print(f"💾 Risultati salvati in: {out_path}")
    
    del candidate_links, features, df_cl, df_us, matches_sorted
    gc.collect()
    return matches

if __name__ == "__main__":
    record_linkage_rules('B1')
    record_linkage_rules('B2')