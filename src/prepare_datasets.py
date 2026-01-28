import pandas as pd
import os
from sklearn.model_selection import train_test_split

def prepare_linkage_datasets():
    print("--- PREPARAZIONE DATASET PER LINKAGE ---")
    
    # 1. Caricamento dati e Ground Truth
    df_cl = pd.read_csv('data/processed/craigslist_aligned.csv')
    df_us = pd.read_csv('data/processed/us_cars_aligned.csv')
    gt = pd.read_csv('data/gt/ground_truth.csv')
    
    # 2. Rimozione colonne VIN dai dataset originali (per evitare bias)
    cols_to_drop = ['vin']
    if 'vin_clean' in df_cl.columns: cols_to_drop.append('vin_clean')
    
    df_cl_no_vin = df_cl.drop(columns=[c for c in cols_to_drop if c in df_cl.columns])
    df_us_no_vin = df_us.drop(columns=[c for c in cols_to_drop if c in df_us.columns])
    
    # Salvataggio dataset senza VIN
    df_cl_no_vin.to_csv('data/processed/craigslist_final.csv', index=True, index_label='id_cl')
    df_us_no_vin.to_csv('data/processed/us_cars_final.csv', index=True, index_label='id_us')
    print("Salvati dataset senza VIN in data/processed/")
    
    # 3. Split Ground Truth in Train/Val/Test (60/20/20)
    train_gt, temp_gt = train_test_split(gt, test_size=0.4, random_state=42)
    val_gt, test_gt = train_test_split(temp_gt, test_size=0.5, random_state=42)
    
    # Salvataggio split
    train_gt.to_csv('data/gt/gt_train.csv', index=False)
    val_gt.to_csv('data/gt/gt_val.csv', index=False)
    test_gt.to_csv('data/gt/gt_test.csv', index=False)
    
    print(f"Ground Truth splittata:")
    print(f" - Train: {len(train_gt)}")
    print(f" - Val: {len(val_gt)}")
    print(f" - Test: {len(test_gt)}")

if __name__ == "__main__":
    prepare_linkage_datasets()
