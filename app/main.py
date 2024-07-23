import streamlit as st
import pandas as pd
import joblib
import os
import shap
import altair as alt

# Chemins des fichiers de données et du modèle
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
client_data_raw_path = os.path.join(BASE_DIR, '../data/app_datas_full_raw.csv')
client_data_part1_path = os.path.join(BASE_DIR, '../data/app_datas_full_imputed_scaled_part1.csv')
client_data_part2_path = os.path.join(BASE_DIR, '../data/app_datas_full_imputed_scaled_part2.csv')
descriptions_path = os.path.join(BASE_DIR, '../data/HomeCredit_columns_description.csv')
model_path = os.path.join(BASE_DIR, '../data/best_xgboost_model.pkl')

# Reconstituer le fichier CSV si les fichiers partiels existent
if os.path.exists(client_data_part1_path) and os.path.exists(client_data_part2_path):
    # Charger les deux fichiers CSV
    df1 = pd.read_csv(client_data_part1_path)
    df2 = pd.read_csv(client_data_part2_path)
    
    # Combiner les deux DataFrames
    client_data_imputed_scaled_df = pd.concat([df1, df2], ignore_index=True)
    
    print("Fichier reconstitué en mémoire.")
else:
    raise FileNotFoundError("Les fichiers partiels nécessaires à la reconstitution des données sont introuvables.")


# Charger les données clients brutes 
client_data_raw_df = pd.read_csv(client_data_raw_path)

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
#pipeline = joblib.load(model_path)
#model = pipeline.named_steps['xgb']  # Extraire le modèle XGBClassifier du pipeline
model = joblib.load(model_path)

# Configuration de la page
st.set_page_config(page_title="Dashboard Client", layout="wide")

# Titre principal
st.title("Dashboard Client")

# Champ de saisie pour le numéro de client
client_id_input = st.text_input("Entrez le numéro de client", "")

def get_client_data(client_id, data_df):
    client_data = data_df[data_df['SK_ID_CURR'] == int(client_id)]
    if client_data.empty:
        return None
    return client_data


# Fonction pour générer le dégradé CSS avec une marque à une valeur spécifiée
def get_gradient_css(value):
    return f"""
    <style>
    .bar {{
        width: 100%;
        height: 30px;
        background: linear-gradient(to right, green 0%, yellow 10%, red 100%);
        border-radius: 5px;
        position: relative;
    }}
    .bar::after {{
        content: '';
        position: absolute;
        top: 0;
        left: {value}%;
        width: 2px;
        height: 100%;
        background-color: black;
    }}
    </style>
    """


# Vérifier si un numéro de client a été entré
if client_id_input:
    client_data_imputed_scaled = get_client_data(client_id_input, client_data_imputed_scaled_df)
    client_data_raw = get_client_data(client_id_input, client_data_raw_df)
    if client_data_imputed_scaled is not None and client_data_raw is not None:
        if st.checkbox("Afficher infos client"):
            # Afficher les informations principales du client
            st.subheader(f"Le client #{client_id_input} :")

            main_features = ['CODE_GENDER_M', 'CODE_GENDER_F', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']
            available_features = [feature for feature in main_features if feature in client_data_raw.columns]

            if available_features:
                client_info = client_data_raw[available_features].iloc[0]
                gender = " - est un homme" if client_info['CODE_GENDER_M'] == 1 and client_info['CODE_GENDER_F'] == 0 else " - est une femme"
                own_car = " - possède une voiture" if client_info['FLAG_OWN_CAR'] == 1 else " - ne possède pas de voiture"
                own_realty = " - est propriétaire" if client_info['FLAG_OWN_REALTY'] == 1 else " - n'est pas propriétaire"
                children = " - n'a pas d'enfants" if client_info['CNT_CHILDREN'] == 0 else f" - a {client_info['CNT_CHILDREN']} enfant(s) à sa charge"
                
                st.write(gender)
                st.write(own_car)
                st.write(own_realty)
                st.write(children)
                
                # Préparer les données pour le graphique
                financial_data = pd.DataFrame({
                    'Catégorie': ['Salaire annuel', 'Montant du crédit', 'Montant de l\'annuité'],
                    'Montant': [client_info['AMT_INCOME_TOTAL'], client_info['AMT_CREDIT'], client_info['AMT_ANNUITY']]
                })
                
                # Créer le graphique à barres horizontales
                bar_chart = alt.Chart(financial_data).mark_bar().encode(
                    x=alt.X('Montant', title='Montant en USD'),
                    y=alt.Y('Catégorie', sort=None, title=''),
                    tooltip=['Catégorie', 'Montant']
                ).properties(
                    width=600,
                    height=300,
                    title="Informations financières"
                )
                
                st.altair_chart(bar_chart)
            else:
                st.warning("Aucune des principales informations n'est disponible pour ce client.")


        # Prédire le score du client
        features = client_data_imputed_scaled.drop(columns=['SK_ID_CURR', 'TARGET']).values.flatten()
        prediction_proba = model.predict_proba([features])[0]
        score_percentage = prediction_proba[1] * 100  # Probabilité pour la classe positive

        # Appliquer le seuil personnalisé de 0.1
        threshold = 0.1
        if prediction_proba[1] > threshold:
            prediction = 1
        else:
            prediction = 0


        if st.checkbox("Afficher la décision de l'algorithme"):

            # Afficher les résultats
            if prediction == 0:
                st.success("Le client a peu de chance d'avoir des soucis de remboursement.")
            else:
                st.error("Le client a de fortes chances d'avoir des soucis de remboursement.")
            
            # Afficher le dégradé dans Streamlit
            st.markdown(get_gradient_css(score_percentage), unsafe_allow_html=True)
            st.markdown('<div class="bar"></div>', unsafe_allow_html=True)
            st.write(f"Le score de crédit prédit pour le client {client_id_input} est : {score_percentage:.2f}%")
        



        # Calculer et afficher les importances locales des caractéristiques
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values([features])

        # Créer un DataFrame pour les importances locales
        local_importances_df = pd.DataFrame({
            'Feature': client_data_imputed_scaled.drop(columns=['SK_ID_CURR', 'TARGET']).columns,
            'Importance': shap_values[0]
        })
        local_importances_df['Description'] = local_importances_df['Feature'].map(descriptions_dict)

        # Trier les importances locales par valeur absolue
        local_importances_df = local_importances_df.sort_values(by='Importance', ascending=False, key=abs)

        # Séparer les importances positives et négatives
        positive_importances_df = local_importances_df[local_importances_df['Importance'] > 0].head(5)
        negative_importances_df = local_importances_df[local_importances_df['Importance'] < 0].head(5)

        # Calculer les importances globales des caractéristiques
        importance = model.get_booster().get_score(importance_type='weight')

        # Convertir en DataFrame
        importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        })

        # Assigner des noms significatifs aux features basées sur leur ordre dans les données d'origine
        feature_names = client_data_imputed_scaled.drop(columns=['SK_ID_CURR', 'TARGET']).columns
        importance_df['Feature'] = importance_df['Feature'].apply(lambda x: feature_names[int(x[1:])])

        # Ajouter les descriptions des caractéristiques
        importance_df['Description'] = importance_df['Feature'].map(descriptions_dict)
        global_importances_df = importance_df.sort_values(by='Importance', ascending=False)

        if st.checkbox("Afficher facteurs de décision"):
            # Afficher les importances locales et globales côte à côte
            st.subheader("Importances des caractéristiques")

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Importances locales")
                st.markdown("""
                **Note explicative :**
                - Les **importances locales positives** indiquent les caractéristiques qui augmentent le score de risque de crédit pour ce client.
                - Les **importances locales négatives** indiquent les caractéristiques qui diminuent le score de risque de crédit pour ce client.
                - Les valeurs de ces importances montrent l'impact de chaque caractéristique sur la prédiction du modèle.
                """)

                st.subheader("Importances locales positives")
                st.dataframe(positive_importances_df)

                st.subheader("Importances locales négatives")
                st.dataframe(negative_importances_df)

            with col2:
                st.subheader("Importances globales")

                st.markdown("""
                **Note explicative :**
                - Les **importances globales** indiquent les caractéristiques les plus influentes sur l'ensemble des prédictions du modèle.
                - **Caractéristiques Clés** : Portez une attention particulière aux caractéristiques avec des importances globales élevées, car elles ont le plus grand impact sur les prédictions du modèle.
                - **Comparaison** : Utilisez les importances globales pour comparer et comprendre comment le modèle prend ses décisions de manière générale.
                """)
                st.dataframe(global_importances_df)

        


        
        
    else:
        st.error("Client ID non trouvé.")
