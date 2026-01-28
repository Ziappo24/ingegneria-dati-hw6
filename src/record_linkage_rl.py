import pandas as pd
import recordlinkage
import time
import os

def record_linkage_rules(blocking_strategy='B1'):
    print(f"\n--- RECORD LINKAGE (RULES) - STRATEGIA: {blocking_strategy} ---")
    
    # 1. Caricamento dataset con campionamento per gestire la memoria
    print("Caricamento dataset e campionamento...")
    df_cl = pd.read_csv('data/processed/craigslist_final.csv', index_col='id_cl').sample(n=100000, random_state=42)
    df_us = pd.read_csv('data/processed/us_cars_final.csv', index_col='id_us').sample(n=100000, random_state=42)
    
    # 2. Definizione Blocking (Reso più restrittivo per evitare ArrayMemoryError)
    start_time = time.time()
    indexer = recordlinkage.Index()
    
    if blocking_strategy == 'B1':
        # B1: Blocking su Make E Year (invece di solo Make)
        indexer.block(['make', 'year'])
    elif blocking_strategy == 'B2':
        # B2: Blocking su Year e Body Type
        indexer.block(['year', 'body_type'])
    
    print("Indicizzazione candidati...")
    candidate_links = indexer.index(df_cl, df_us)
    print(f"Candidate links trovati: {len(candidate_links)}")
    
    if len(candidate_links) > 10000000:
        print("ATTENZIONE: Troppi candidati. Campionamento forzato per evitare crash.")
        candidate_links = candidate_links[:10000000]
    
    # 3. Confronto
    compare_cl = recordlinkage.Compare()
    
    # Confronto testuale (Jaro-Winkler) per modello
    compare_cl.string('model', 'model', method='jarowinkler', threshold=0.85, label='model')
    
    # Confronto esatto per carburante e trasmissione
    compare_cl.exact('fuel_type', 'fuel_type', label='fuel')
    compare_cl.exact('transmission', 'transmission', label='transmission')
    
    # Esecuzione confronto
    print("Calcolo similitudini...")
    features = compare_cl.compute(candidate_links, df_cl, df_us)
    
    # 4. Classificazione (Regola)
    # Dato che make/year sono già bloccati, contano come match perfetto (1.0)
    features['total_score'] = (features['model'] * 3 + 
                               features['fuel'] * 0.5 + 
                               features['transmission'] * 0.5)
    
    # Soglia basata sulle feature caricate
    matches = features[features['total_score'] >= 3.0]
    end_time = time.time()
    
    print(f"Match identificati: {len(matches)}")
    print(f"Tempo di esecuzione: {end_time - start_time:.2f}s")
    
    # Salvataggio risultati
    os.makedirs('data/results', exist_ok=True)
    matches.to_csv(f'data/results/matches_rl_{blocking_strategy}.csv')
    
    return matches

if __name__ == "__main__":
    record_linkage_rules('B1')
    record_linkage_rules('B2')
