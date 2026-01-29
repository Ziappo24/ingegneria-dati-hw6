import os
import pandas as pd

def row_to_ditto(row, columns):
    """Converte una riga di pandas nel formato [COL] col [VAL] val per DITTO."""
    parts = []
    for col in columns:
        val = str(row[col]).strip() if pd.notna(row[col]) else "NaN"
        parts.append(f"[COL] {col} [VAL] {val}")
    return " ".join(parts)

def prepare_ditto():
    print("🚀 Inizio preparazione dataset per DITTO...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gt_dir = os.path.join(base_dir, 'data', 'gt')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    output_dir = os.path.join(base_dir, 'data', 'ditto')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Caricamento dataset originali
    print("Caricamento dataset car...")
    df_cl = pd.read_csv(os.path.join(processed_dir, 'craigslist_final.csv'), index_col='id_cl')
    df_us = pd.read_csv(os.path.join(processed_dir, 'us_cars_final.csv'), index_col='id_us')
    
    # Colonne da includere (escludendo gli ID)
    columns = ['make', 'model', 'year', 'mileage', 'price', 'transmission', 'body_type', 'fuel_type']
    
    splits = {
        'train.txt': 'gt_train.csv',
        'valid.txt': 'gt_val.csv',
        'test.txt': 'gt_test.csv'
    }
    
    for out_file, gt_file in splits.items():
        print(f"Elaborazione {gt_file} -> {out_file}...")
        gt_path = os.path.join(gt_dir, gt_file)
        if not os.path.exists(gt_path):
            print(f"⚠️ Avviso: {gt_file} non trovato, salto.")
            continue
            
        gt = pd.read_csv(gt_path)
        lines = []
        
        for _, row in gt.iterrows():
            id_cl = row['id_cl']
            id_us = row['id_us']
            label = int(row['label'])
            
            if id_cl in df_cl.index and id_us in df_us.index:
                rec1 = row_to_ditto(df_cl.loc[id_cl], columns)
                rec2 = row_to_ditto(df_us.loc[id_us], columns)
                # Formato: record1 \t record2 \t label
                lines.append(f"{rec1}\t{rec2}\t{label}")
        
        with open(os.path.join(output_dir, out_file), 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
            
    print(f"✅ Preparazione DITTO completata! File salvati in: {output_dir}")

if __name__ == "__main__":
    prepare_ditto()
