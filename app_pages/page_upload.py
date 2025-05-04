# pages/page_upload.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def page_upload():
    st.title("📂 Chargement des données")
    st.markdown("Importez une base de données au format CSV pour appliquer les plans de sondage.")

    # Choix du séparateur
    sep = st.radio(
        "Séparateur du fichier CSV :",
        options=[",", ";", "\t"],
        index=1,
        format_func=lambda x: f'"{x}" ({"virgule" if x=="," else "point-virgule" if x==";" else "tabulation"})'
    )

    # Upload du fichier
    uploaded_file = st.file_uploader("Sélectionnez votre fichier CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # Lecture du fichier CSV avec détection étendue des valeurs manquantes
            df = pd.read_csv(
                uploaded_file,
                sep=sep,
                na_values=["", " ", "NA", "N/A", "null", "Null", "NaN"]
            )

            # Remplacement des cellules vides ou espaces par NA
            df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)

            # Nettoyage des noms de colonnes
            df.columns = df.columns.str.strip()

            # Stockage dans session_state
            st.session_state["data"] = df

            # Aperçu des données
            st.success("✅ Fichier chargé avec succès !")
            st.subheader("Aperçu des données")
            st.dataframe(df.head())

            # Infos générales
            st.markdown(f"**Nombre total d’observations** : `{df.shape[0]}`")
            st.markdown(f"**Nombre de variables** : `{df.shape[1]}`")

            # Noms des variables
            st.markdown("**Variables disponibles :**")
            st.write(list(df.columns))

            # Boxplots des variables quantitatives (hors première colonne)
            st.subheader("📊 Boxplots des variables quantitatives")
            numeric_cols = df.select_dtypes(include="number").columns

            # Exclure la première colonne si elle est numérique
            if len(df.columns) > 1:
                first_col = df.columns[0]
                numeric_cols = [col for col in numeric_cols if col != first_col]

            if len(numeric_cols) == 0:
                st.info("Aucune variable numérique détectée (hors première colonne).")
            else:
                for col in numeric_cols:
                    fig, ax = plt.subplots()
                    sns.boxplot(x=df[col], ax=ax)
                    ax.set_title(f"Boxplot - {col}")
                    st.pyplot(fig)

            # Tableaux de contingence pour variables qualitatives
            st.subheader("🔁 Tableau de contingence (variables qualitatives)")
            cat_cols = df.select_dtypes(include="object").columns
            if len(cat_cols) < 2:
                st.info("Pas assez de variables qualitatives pour créer des tableaux croisés.")
            else:
                var1 = st.selectbox("Variable 1 :", cat_cols, key="var1")
                var2 = st.selectbox("Variable 2 :", cat_cols, key="var2")

                if var1 != var2:
                    contingency = pd.crosstab(df[var1], df[var2])
                    st.write(f"Tableau de contingence entre **{var1}** et **{var2}** :")
                    st.dataframe(contingency)
                else:
                    st.warning("Veuillez sélectionner deux variables différentes.")

        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier : {e}")
    else:
        st.info("Veuillez importer un fichier CSV pour commencer.")
