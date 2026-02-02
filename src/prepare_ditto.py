import os
import pandas as pd

def serialize(row, cols):
    """Formatta la riga: COL [nome] VAL [valore] ..."""
    return " ".join([f"COL {c} VAL {str(row[c]).strip() if pd.notna(row[c]) else 'NaN'}" for c in cols])

def run_preparation():
    # Definiamo la cartella di destinazione dentro il repo FAIR-DA4ER
    base_path = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.join(base_path, 'ditto_repository', 'FAIR-DA4ER-main')
    output_path = os.path.join(repo_path, 'data', 'auto_task') #
    os.makedirs(output_path, exist_ok=True)

    # Caricamento dati
    df_cl = pd.read_csv('data/processed/craigslist_final.csv', index_col='id_cl')
    df_us = pd.read_csv('data/processed/us_cars_final.csv', index_col='id_us')
    cols = ['make', 'model', 'year', 'transmission', 'fuel_type']

    splits = {'train.txt': 'gt_train.csv', 'valid.txt': 'gt_val.csv', 'test.txt': 'gt_test.csv'}

    for out_name, gt_name in splits.items():
        gt = pd.read_csv(f'data/gt/{gt_name}')
        lines = []
        for _, row in gt.iterrows():
            if row['id_cl'] in df_cl.index and row['id_us'] in df_us.index:
                s1 = serialize(df_cl.loc[row['id_cl']], cols)
                s2 = serialize(df_us.loc[row['id_us']], cols)
                lines.append(f"{s1}\t{s2}\t{int(row['label'])}")
        
        with open(os.path.join(output_path, out_name), 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
    print(f"✅ File salvati in: {output_path}")

if __name__ == "__main__":
    run_preparation()