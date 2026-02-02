import json
import pandas as pd
import os

# Percorsi (Usa 'r' per Windows)
path_json = r'C:\Users\astor\Desktop\UNI\MAGISTRALE\SECONDO ANNO\INGEGNERIA DEI DATI\Homework - 6 V.2\ingegneria-dati-hw6\ditto_repository\FAIR-DA4ER-main\ditto\output\predictions.jsonl'
path_test = r'C:\Users\astor\Desktop\UNI\MAGISTRALE\SECONDO ANNO\INGEGNERIA DEI DATI\Homework - 6 V.2\ingegneria-dati-hw6\ditto_repository\FAIR-DA4ER-main\ditto\data\auto_task\test.txt'
path_gt_test = r'C:\Users\astor\Desktop\UNI\MAGISTRALE\SECONDO ANNO\INGEGNERIA DEI DATI\Homework - 6 V.2\ingegneria-dati-hw6\data\gt\gt_test.csv'
output_path = r'data/results/final_matches_ditto.csv'

# 1. Carica le predizioni di Ditto
predictions = []
with open(path_json, 'r') as f:
    for line in f:
        if line.strip():
            predictions.append(json.loads(line))
df_ditto = pd.DataFrame(predictions)

# 2. Carica la Ground Truth di test per recuperare gli ID reali
# Usiamo questa perché ha sicuramente id_cl e id_us nell'ordine corretto
df_gt = pd.read_csv(path_gt_test)

# 3. Uniamo i risultati
# Ditto mantiene lo stesso ordine del file di test
df_combined = pd.concat([df_gt[['id_cl', 'id_us']], df_ditto], axis=1)

# 4. Identifica la colonna del match (match o target)
col_match = 'match' if 'match' in df_combined.columns else 'target'

# 5. Filtra solo i match confermati da Ditto
df_final = df_combined[df_combined[col_match].astype(int) == 1].copy()

# 6. Salvataggio
if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))

# Salviamo solo le colonne richieste da evaluation.py
df_final[['id_cl', 'id_us']].to_csv(output_path, index=False)

print(f"✅ Successo! Creato {output_path} con {len(df_final)} match e colonne id_cl/id_us.")