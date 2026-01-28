MEDIATED_SCHEMA = [
    "make",
    "model",
    "year",
    "price",
    "mileage",
    "fuel",
    "transmission",
    "state",
    "description",
    "vin"
]

CRAIGSLIST_MAPPING = {
    "manufacturer": "make",
    "model": "model",
    "year": "year",
    "price": "price",
    "odometer": "mileage",
    "fuel": "fuel",
    "transmission": "transmission",
    "state": "state",
    "description": "description",
    "vin": "vin"
}

USED_CARS_MAPPING = {
    "brand": "make",
    "model": "model",
    "model_year": "year",
    "price": "price",
    "milage": "mileage",
    "fuel_type": "fuel",
    "transmission": "transmission",
    "state": "state",
    "description": "description",
    "vin": "vin"
}


def align_schema(df, mapping):
    """
    Seleziona e rinomina le colonne secondo lo schema mediato
    """
    df = df[list(mapping.keys())]
    df = df.rename(columns=mapping)
    return df
