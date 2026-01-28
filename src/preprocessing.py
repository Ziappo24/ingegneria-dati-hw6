import pandas as pd

def exploratory_analysis(file_path, source_name, **kwargs):
    """
    Esegue l'analisi esplorativa richiesta: percentuale nulli e valori unici.
    """
    print(f"\n--- ANALISI ESPLORATIVA: {source_name} ---")
    
    # Caricamento del dataset
    # Nota: i dataset di Kaggle possono essere pesanti, limitiamo a un campione per l'analisi iniziale
    try:
        # Passiamo eventuali kwargs (es. nrows, dtype) a read_csv
        df = pd.read_csv(file_path, **kwargs)
    except FileNotFoundError:
        print(f"Errore: File '{file_path}' non trovato.")
        return None

    # Numero totale di righe
    total_rows = len(df)
    
    # Calcolo delle metriche
    analysis_df = pd.DataFrame({
        'Tipo Dato': df.dtypes,
        'Valori Nulli': df.isnull().sum(),
        'Percentuale Nulli (%)': (df.isnull().sum() / total_rows * 100).round(2),
        'Valori Unici': df.nunique()
    })
    
    # Ordiniamo per percentuale di nulli decrescente per vedere subito le colonne "vuote"
    analysis_df = analysis_df.sort_values(by='Percentuale Nulli (%)', ascending=False)
    
    print(analysis_df)
    print(f"\nDimensioni totali: {df.shape[0]} righe, {df.shape[1]} colonne.")
    return analysis_df

# 1. Analisi Sorgente Craigslist (Sorgente 1)
# Percorso corretto: data/raw/craiglist/vehicles.csv
stats_craigslist = exploratory_analysis('data/raw/craiglist/vehicles.csv', 'Craigslist', nrows=1000000)

# 2. Analisi Sorgente US Used Cars (Sorgente 2)
# Percorso corretto: data/raw/us_used_cars/used_cars_data.csv
# Specifichiamo i tipi per evitare DtypeWarning su colonne con dati misti
stats_us_cars = exploratory_analysis(
    'data/raw/us_used_cars/used_cars_data.csv', 
    'US Used Cars', 
    nrows=1000000,
    dtype={'dealer_zip': str, 'bed': str}
)