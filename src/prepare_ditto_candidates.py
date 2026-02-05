import os
import pandas as pd
import argparse
import sys

def serialize(row, cols):
    """Formatta la riga: COL [nome] VAL [valore] ..."""
    return " ".join([f"COL {c} VAL {str(row[c]).strip() if pd.notna(row[c]) else 'NaN'}" for c in cols])

def run_preparation(input_csv, output_name):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Percorsi file - assicurati che combacino con la tua struttura
    cl_path = os.path.join(base_path, 'data', 'processed', 'craigslist_final.csv')
    us_path = os.path.join(base_path, 'data', 'processed', 'us_cars_final.csv')
    
    repo_path = os.path.join(base_path, 'ditto_repository', 'FAIR-DA4ER-main')
    output_dir = os.path.join(repo_path, 'ditto', 'data', 'auto_task')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, output_name)

    print(f"Caricamento dati...")
    df_cl = pd.read_csv(cl_path, index_col='id_cl')
    df_us = pd.read_csv(us_path, index_col='id_us')
    
    # Colonne da usare per la serializzazione
    cols = ['make', 'model', 'year', 'transmission', 'fuel_type']

    print(f"Lettura candidati da: {input_csv}")
    try:
        candidates = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Errore: File {input_csv} non trovato.")
        sys.exit(1)

    # Verifica colonne
    if 'id_cl' not in candidates.columns or 'id_us' not in candidates.columns:
        print("Errore: Il CSV di input deve contenere le colonne 'id_cl' e 'id_us'.")
        print(f"Colonne trovate: {candidates.columns.tolist()}")
        # Tentativo di fallback se i nomi sono diversi (es. cl_id)
        if 'cl_id' in candidates.columns and 'us_id' in candidates.columns:
            candidates.rename(columns={'cl_id': 'id_cl', 'us_id': 'id_us'}, inplace=True)
            print("Rinominate colonne cl_id -> id_cl, us_id -> id_us")
        else:
            sys.exit(1)

    lines = []
    skipped = 0
    print(f"Elaborazione {len(candidates)} coppie...")
    
    for _, row in candidates.iterrows():
        id_cl, id_us = row['id_cl'], row['id_us']
        
        # Gestione float/int
        try:
            id_cl = int(id_cl)
            id_us = int(id_us)
        except:
            continue

        if id_cl in df_cl.index and id_us in df_us.index:
            s1 = serialize(df_cl.loc[id_cl], cols)
            s2 = serialize(df_us.loc[id_us], cols)
            # 0 è un'etichetta fittizia per l'inferenza
            lines.append(f"{s1}\t{s2}\t0")
        else:
            skipped += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    
    print(f"✅ File salvato in: {output_file}")
    print(f"Coppie scritte: {len(lines)}")
    print(f"Coppie saltate (ID mancanti): {skipped}")
    print(f"\nPer eseguire l'inferenza con Ditto:")
    print(f"> cd ditto_repository/FAIR-DA4ER-main/ditto")
    print(f"> python matcher.py --task auto_task --input_path data/auto_task/{output_name} --output_path output/matches_{output_name.replace('.txt', '.jsonl')} --lm distilbert --max_len 256 --use_gpu --fp16 --checkpoint_path checkpoints/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara i candidati per Ditto")
    parser.add_argument("input_csv", help="Percorso al file CSV dei candidati (es. data/results/matches_rl_B2.csv)")
    parser.add_argument("--output", default="candidates.txt", help="Nome del file di output (default: candidates.txt)")
    
    args = parser.parse_args()
    run_preparation(args.input_csv, args.output)
