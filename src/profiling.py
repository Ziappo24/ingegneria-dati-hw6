import pandas as pd

def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola:
    - percentuale di valori nulli
    - percentuale di valori unici
    - tipo di dato
    """
    profile = pd.DataFrame({
        "null_percentage": df.isnull().mean() * 100,
        "unique_percentage": df.nunique() / len(df) * 100,
        "dtype": df.dtypes
    })

    return profile.sort_values(by="null_percentage", ascending=False)
