import pandas as pd

# Charger le fichier CSV
input_file = 'app_datas_full_imputed_scaled.csv'
df = pd.read_csv(input_file)

# Calculer le point de scission
split_index = len(df) // 2

# Scinder le DataFrame en deux
df1 = df.iloc[:split_index]
df2 = df.iloc[split_index:]

# Enregistrer les deux fichiers CSV
df1.to_csv('app_datas_full_imputed_scaled_part1.csv', index=False)
df2.to_csv('app_datas_full_imputed_scaled_part2.csv', index=False)

print(f"Fichier scindé en deux parties: 'app_datas_full_imputed_scaled_part1.csv' et 'app_datas_full_imputed_scaled_part2.csv'")
