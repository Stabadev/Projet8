import streamlit as st
import pandas as pd
import joblib
import os
import shap

# Chemins des fichiers de données et du modèle
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
client_data_path = os.path.join(BASE_DIR, '../data/app_datas_light_imputed_scaled.csv')
descriptions_path = os.path.join(BASE_DIR, '../data/HomeCredit_columns_description.csv')
model_path = os.path.join(BASE_DIR, '../data/best_xgboost_model.pkl')

# Charger les données clients
client_data_df = pd.read_csv(client_data_path)

# Charger les descriptions des caractéristiques avec gestion des encodages
encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
descriptions_df = None
for encoding in encodings:
    try:
        descriptions_df = pd.read_csv(descriptions_path, usecols=['Row', 'Description'], encoding=encoding)
        break
    except UnicodeDecodeError as e:
        print(f"Failed to read with encoding {encoding}: {e}")

if descriptions_df is None:
    raise ValueError("Failed to read the descriptions file with all attempted encodings.")

descriptions_dict = descriptions_df.set_index('Row')['Description'].to_dict()

# Charger le modèle
model = joblib.load(model_path)

# Configuration de la page
st.set_page_config(page_title="Dashboard Client", layout="wide")

# Titre principal
st.title("Dashboard Client")

# Champ de saisie pour le numéro de client
client_id_input = st.text_input("Entrez le numéro de client", "")

def get_client_data(client_id):
    client_data = client_data_df[client_data_df['SK_ID_CURR'] == int(client_id)]
    if client_data.empty:
        return None
    return client_data

# Vérifier si un numéro de client a été entré
if client_id_input:
    client_data = get_client_data(client_id_input)
    if client_data is not None:
        st.subheader(f"Caractéristiques du client {client_id_input}")
        client_data_display = client_data.drop(columns=['SK_ID_CURR', 'TARGET']).T
        client_data_display.columns = ['Valeur']
        client_data_display['Description'] = client_data_display.index.map(descriptions_dict)
        st.dataframe(client_data_display)

        # Prédire le score du client
        features = client_data.drop(columns=['SK_ID_CURR', 'TARGET']).values.flatten()
        prediction_proba = model.predict_proba([features])[0]
        score_percentage = prediction_proba[1] * 100  # Probabilité pour la classe positive
        st.subheader(f"Score de crédit du client {client_id_input}")
        st.write(f"Le score de crédit prédit pour le client {client_id_input} est : {score_percentage:.2f}%")

        # Calculer et afficher les importances locales des caractéristiques
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values([features])
        
        # Créer un DataFrame pour les importances locales
        local_importances_df = pd.DataFrame({
            'Feature': client_data.drop(columns=['SK_ID_CURR', 'TARGET']).columns,
            'Importance': shap_values[0]
        })
        local_importances_df['Description'] = local_importances_df['Feature'].map(descriptions_dict)
        local_importances_df = local_importances_df.sort_values(by='Importance', ascending=False)
        
        st.subheader("Importances locales des caractéristiques")
        st.dataframe(local_importances_df)
        
     
    else:
        st.error("Client ID non trouvé.")
