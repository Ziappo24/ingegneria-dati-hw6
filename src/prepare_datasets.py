import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split

def prepare_linkage_datasets():
    print("--- PREPARAZIONE DATASET E GROUND TRUTH BILANCIATA ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    gt_dir = os.path.join(base_dir, 'data', 'gt')
    
    cl_aligned = os.path.join(processed_dir, 'craigslist_aligned.csv')
    us_aligned = os.path.join(processed_dir, 'us_cars_aligned.csv')
    gt_positive_path = os.path.join(gt_dir, 'ground_truth.csv')
    
    if not all(os.path.exists(p) for p in [cl_aligned, us_aligned, gt_positive_path]):
        print(f"❌ Errore: Esegui prima generate_gt.py")
        return

    df_cl = pd.read_csv(cl_aligned)
    df_us = pd.read_csv(us_aligned)
    gt_pos = pd.read_csv(gt_positive_path)
    
    df_cl['id_cl'] = df_cl.index
    df_us['id_us'] = df_us.index
    
    # --- MODIFICA: Rimozione VIN per cecità del modello (Punto 4.B) ---
    cols_to_drop = ['vin', 'vin_clean', 'vin_gt']
    df_cl_final = df_cl.drop(columns=[c for c in cols_to_drop if c in df_cl.columns])
    df_us_final = df_us.drop(columns=[c for c in cols_to_drop if c in df_us.columns])
    
    df_cl_final.to_csv(os.path.join(processed_dir, 'craigslist_final.csv'), index=False)
    df_us_final.to_csv(os.path.join(processed_dir, 'us_cars_final.csv'), index=False)

    # 3. Generazione esempi NEGATIVI (label=0)
    print("Generazione esempi negativi bilanciati...")
    pos_pairs = set(zip(gt_pos['id_cl'], gt_pos['id_us']))
    id_cl_list = df_cl['id_cl'].tolist()
    id_us_list = df_us['id_us'].tolist()
    
    neg_pairs = []
    num_needed = len(gt_pos)
    
    random.seed(42)
    while len(neg_pairs) < num_needed:
        c = random.choice(id_cl_list)
        u = random.choice(id_us_list)
        if (c, u) not in pos_pairs:
            neg_pairs.append({'id_cl': c, 'id_us': u, 'label': 0})
            
    df_neg = pd.DataFrame(neg_pairs)
    gt_balanced = pd.concat([gt_pos, df_neg]).sample(frac=1, random_state=42)
    
    # --- MODIFICA MIGLIORATA: Split Stratificato ---
    # Lo stratify assicura che Train, Val e Test abbiano tutti il 50% di match e 50% di non-match
    train_gt, temp_gt = train_test_split(
        gt_balanced, 
        test_size=0.30, 
        random_state=42, 
        stratify=gt_balanced['label']
    )
    val_gt, test_gt = train_test_split(
        temp_gt, 
        test_size=0.50, 
        random_state=42, 
        stratify=temp_gt['label']
    )
    
    # Salvataggio
    os.makedirs(gt_dir, exist_ok=True)
    train_gt.to_csv(os.path.join(gt_dir, 'gt_train.csv'), index=False)
    val_gt.to_csv(os.path.join(gt_dir, 'gt_val.csv'), index=False)
    test_gt.to_csv(os.path.join(gt_dir, 'gt_test.csv'), index=False)
    
    print(f"✅ Ground Truth pronta e stratificata (Train: {len(train_gt)}, Val: {len(val_gt)}, Test: {len(test_gt)})")

if __name__ == "__main__":
    prepare_linkage_datasets()