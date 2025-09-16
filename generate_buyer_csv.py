import pandas as pd
import numpy as np

def generate_buyer_csv(input_file="used_cars_processed.csv", output_file="buyer_listings.csv", n=10, random_state=42):
    np.random.seed(random_state)

    # Cargar el dataset limpio
    df = pd.read_csv(input_file)

    # Seleccionar una muestra aleatoria de n autos
    sample_df = df.sample(n=n, random_state=random_state).copy()

    # Aplicar variación aleatoria de ±15% a la columna 'price'
    variation = np.random.uniform(1, 1.50, size=len(sample_df))
    sample_df["price"] = (sample_df["price"] * variation).round(0).astype(int)

    # Guardar como CSV de prueba para el Buyer mode
    sample_df.to_csv(output_file, index=False)
    print(f"✅ Sample of {n} cars with random ±15% price variation saved to {output_file}")


if __name__ == "__main__":
    generate_buyer_csv()
