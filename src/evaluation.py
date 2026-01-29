import os
import pandas as pd

def evaluate_results():
    print("📊 Avvio Valutazione Modelli...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'data', 'results')
    gt_test_path = os.path.join(base_dir, 'data', 'gt', 'gt_test.csv')
    
    if not os.path.exists(gt_test_path):
        print(f"❌ Errore: Ground Truth di test non trovata in {gt_test_path}")
        return

    # Caricamento Ground Truth di Test
    gt_test = pd.read_csv(gt_test_path)
    # Creiamo un set di coppie positive e negative per un look-up veloce
    # Usiamo tuple (id_cl, id_us)
    positives_gt = set(tuple(x) for x in gt_test[gt_test['label'] == 1][['id_cl', 'id_us']].values)
    negatives_gt = set(tuple(x) for x in gt_test[gt_test['label'] == 0][['id_cl', 'id_us']].values)
    
    total_positives_in_gt = len(positives_gt)
    print(f"INFO: Ground Truth di test caricata ({len(gt_test)} record totali, {total_positives_in_gt} positivi)")

    # Risultati da valutare
    files_to_evaluate = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    evaluation_report = []

    for file_name in files_to_evaluate:
        file_path = os.path.join(results_dir, file_name)
        print(f"\nValutazione: {file_name}...")
        
        try:
            df_res = pd.read_csv(file_path)
            
            # Normalizzazione nomi colonne (RL usa id_cl/id_us, Dedupe usa cl_id/us_id)
            if 'cl_id' in df_res.columns:
                df_res = df_res.rename(columns={'cl_id': 'id_cl', 'us_id': 'id_us'})
            
            # Solo coppie uniche trovate dal modello
            found_pairs = set(tuple(x) for x in df_res[['id_cl', 'id_us']].values)
            
            # Metriche
            tp = len(found_pairs.intersection(positives_gt))
            fp = len(found_pairs.intersection(negatives_gt))
            
            # Calcolo Precision/Recall/F1
            # Precision = TP / (TP + FP)
            # Nota: se ci sono coppie trovate che non sono proprio nella GT test, 
            # tecnicamente dovrebbero essere FP se assumiamo la GT completa per quei record.
            # Per semplicità qui contiamo FP solo rispetto ai negativi espliciti in GT test.
            
            precision = tp / len(found_pairs) if len(found_pairs) > 0 else 0
            recall = tp / total_positives_in_gt if total_positives_in_gt > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            evaluation_report.append({
                'Modello': file_name,
                'TP': tp,
                'FP': fp,
                'Found': len(found_pairs),
                'Precision': round(precision, 4),
                'Recall': round(recall, 4),
                'F1-Score': round(f1, 4)
            })
            
        except Exception as e:
            print(f"⚠️ Errore durante l'elaborazione di {file_name}: {e}")

    # Visualizzazione Report Finale
    if evaluation_report:
        report_df = pd.DataFrame(evaluation_report)
        print("\n" + "="*80)
        print("REPORT FINALE DI VALUTAZIONE")
        print("="*80)
        print(report_df.to_string(index=False))
        print("="*80)
        
        # Salvataggio report
        report_path = os.path.join(results_dir, 'evaluation_report.csv')
        report_df.to_csv(report_path, index=False)
        print(f"Report salvato in: {report_path}")
    else:
        print("Nessun risultato trovato da valutare.")

if __name__ == "__main__":
    evaluate_results()
