import os
import pandas as pd

def evaluate_results():
    print("📊 Avvio Valutazione Modelli (Versione Corretta)...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'data', 'results')
    gt_test_path = os.path.join(base_dir, 'data', 'gt', 'gt_test.csv')
    
    if not os.path.exists(gt_test_path):
        print(f"❌ Errore: Ground Truth di test non trovata in {gt_test_path}")
        return

    # 1. Caricamento Ground Truth di Test
    gt_test = pd.read_csv(gt_test_path)
    
    # Set di ID inclusi nel test per definire lo "scope" della valutazione
    test_ids_cl = set(gt_test['id_cl'].unique())
    test_ids_us = set(gt_test['id_us'].unique())

    # Coppie positive certe (label 1)
    positives_gt = set(tuple(x) for x in gt_test[gt_test['label'] == 1][['id_cl', 'id_us']].values)
    
    total_positives_in_gt = len(positives_gt)
    print(f"INFO: Ground Truth caricata. Positivi nel Test Set: {total_positives_in_gt}")

    # 2. Iterazione sui file di risultato
    files_to_evaluate = [f for f in os.listdir(results_dir) if f.endswith('.csv') and f != 'evaluation_report.csv']
    evaluation_report = []

    for file_name in files_to_evaluate:
        file_path = os.path.join(results_dir, file_name)
        
        try:
            df_res = pd.read_csv(file_path)
            
            # Normalizzazione nomi colonne (Dedupe/RL)
            if 'cl_id' in df_res.columns:
                df_res = df_res.rename(columns={'cl_id': 'id_cl', 'us_id': 'id_us'})
            
            # Coppie totali trovate dal modello
            found_pairs_all = set(tuple(x) for x in df_res[['id_cl', 'id_us']].values)
            
            # FILTRO SCOPE: Consideriamo solo i match tra record che fanno parte del Test Set
            # Questo evita di penalizzare il modello per match corretti "fuori test"
            found_pairs = { (c, u) for c, u in found_pairs_all if c in test_ids_cl and u in test_ids_us }
            
            # --- CALCOLO METRICHE CORRETTO ---
            # True Positives: quanti match trovati sono nella lista dei positivi GT
            tp = len(found_pairs.intersection(positives_gt))
            
            # SOTTOLINEATO: False Positives: tutto ciò che è stato trovato ma non è un TP
            # Questo corregge l'errore dello 0 nella colonna FP
            found_count = len(found_pairs)
            fp = found_count - tp
            
            # Precision = TP / Found
            precision = tp / found_count if found_count > 0 else 0
            
            # Recall = TP / Positivi Totali in GT
            recall = tp / total_positives_in_gt if total_positives_in_gt > 0 else 0
            
            # F1-Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            evaluation_report.append({
                'Modello': file_name,
                'TP': tp,
                'FP': fp,
                'Found': found_count,
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1-Score': round(f1, 4)
            })
            
        except Exception as e:
            print(f"⚠️ Errore su {file_name}: {e}")

    # 3. Visualizzazione e Salvataggio
    if evaluation_report:
        report_df = pd.DataFrame(evaluation_report).sort_values(by='F1-Score', ascending=False)
        print("\n" + "="*80)
        print("REPORT FINALE DI VALUTAZIONE")
        print("="*80)
        print(report_df.to_string(index=False))
        print("="*80)
        
        report_path = os.path.join(results_dir, 'evaluation_report.csv')
        report_df.to_csv(report_path, index=False)
        print(f"✅ Report salvato in: {report_path}")

if __name__ == "__main__":
    evaluate_results()