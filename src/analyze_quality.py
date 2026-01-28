import pandas as pd

def analyze_dataset(path, name):
    print(f"\n{'='*20} {name} {'='*20}")
    # Carichiamo un campione per velocità se necessario, ma qui proviamo il totale per precisione nulli
    df = pd.read_csv(path)
    
    total = len(df)
    print(f"Righe totali: {total}")
    
    # 1. Percentuale Nulli
    null_pct = (df.isnull().sum() / total * 100).round(2)
    print("\nPROPORZIONE VALORI NULLI (%):")
    print(null_pct)
    
    # 2. Analisi body_type
    if 'body_type' in df.columns:
        print("\nTOP 10 VALUES FOR 'body_type':")
        print(df['body_type'].value_counts(dropna=False, normalize=True).head(10).apply(lambda x: f"{x*100:.2f}%"))
        print(f"Valori unici: {df['body_type'].nunique()}")
        
    # 3. Analisi city
    if 'city' in df.columns:
        print("\nTOP 10 VALUES FOR 'city':")
        print(df['city'].value_counts(dropna=False, normalize=True).head(10).apply(lambda x: f"{x*100:.2f}%"))
        print(f"Valori unici: {df['city'].nunique()}")

if __name__ == "__main__":
    analyze_dataset('data/processed/craigslist_aligned.csv', 'CRAIGSLIST ALIGNED')
    analyze_dataset('data/processed/us_cars_aligned.csv', 'US CARS ALIGNED')
