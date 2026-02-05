import pandas as pd
import json
import os
import argparse

def convert_results(strategy):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Percorsi Input
    csv_path = os.path.join(base_dir, 'data', 'results', f'matches_rl_{strategy}.csv')
    jsonl_path = os.path.join(base_dir, 'ditto_repository', 'FAIR-DA4ER-main', 'ditto', 'output', f'matches_candidates_{strategy}.jsonl')
    
    # Percorso Output
    output_path = os.path.join(base_dir, 'data', 'results', f'matches_ditto_{strategy}.csv')

    print(f"Converting Ditto results for strategy {strategy}...")
    print(f"Input CSV (IDs): {csv_path}")
    print(f"Input JSONL (Preds): {jsonl_path}")

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
    if not os.path.exists(jsonl_path):
        print(f"Error: JSONL file not found: {jsonl_path}")
        return

    # 1. Carica IDs dal CSV originale (usato come input per Ditto)
    df_ids = pd.read_csv(csv_path)
    
    # 2. Carica Predizioni da JSONL
    predictions = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    
    df_preds = pd.DataFrame(predictions)

    # Controllo allineamento
    if len(df_ids) != len(df_preds):
        print(f"Warning: Line count mismatch! CSV: {len(df_ids)}, JSONL: {len(df_preds)}")
        # Se c'è mismatch, prova a troncare al minimo comune (solitamente Ditto salta righe vuote o errori)
        min_len = min(len(df_ids), len(df_preds))
        df_ids = df_ids.iloc[:min_len]
        df_preds = df_preds.iloc[:min_len]
        print(f"Truncating to {min_len} lines.")

    # 3. Merge
    # Assumiamo che l'ordine sia preservato (Dedupe/RL -> txt -> Ditto -> jsonl)
    df_final = pd.concat([df_ids[['id_cl', 'id_us']].reset_index(drop=True), df_preds[['match', 'match_confidence']].reset_index(drop=True)], axis=1)

    # 4. Filtra i Match (match == 1)
    df_matches = df_final[df_final['match'] == 1].copy()
    
    # Rinomina per coerenza
    df_matches['confidence'] = df_matches['match_confidence']
    
    # 5. Salva
    cols_to_save = ['id_cl', 'id_us'] # Aggiungi 'confidence' se serve per debug, ma evaluation.py vuole id_cl, id_us
    df_matches[cols_to_save].to_csv(output_path, index=False)
    
    print(f"✅ Successo! Salvato {output_path}")
    print(f"Match trovati: {len(df_matches)} su {len(df_final)} coppie candidate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", help="Blocking strategy (B1 or B2)")
    args = parser.parse_args()
    convert_results(args.strategy)
