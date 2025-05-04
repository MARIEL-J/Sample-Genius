import streamlit as st

def page_home():
    st.title("🎓 Application de Simulation des Plans de Sondage")

    st.markdown("""
    ---
    ### 👋 Bienvenue !

    Cette application interactive vous permet de **simuler différents plans de sondage statistiques**, dans un environnement intuitif et visuel.  
    Elle est développée pour les étudiants, enseignants et professionnels travaillant sur les **méthodologies d’enquêtes** et les **techniques d’échantillonnage**.

    ---
    ### 🚀 Que pouvez-vous faire ici ?
    
    - 📦 **Importer vos données** (fichier CSV)
    - 🎲 **Simuler différents plans de sondage** :
        - Aléatoire simple (avec/sans remise)
        - Stratifié
        - Par grappes
        - À plusieurs degrés (2 ou plus)
        - À probabilités inégales
    - 📈 **Consulter et exporter** les échantillons générés
    - 📤 **Télécharger les résultats** sous format `.csv`

    ---
    ### 🔍 Comment commencer ?
    1. Cliquez sur **“Chargement des données”** pour importer votre base.
    2. Naviguez dans les différents modules dans le menu à gauche pour appliquer des plans de sondage.
    3. Lisez les messages d’aide et ajustez vos paramètres.

    ---
    ⚠️ **Note** : Cette application ne stocke aucune donnée. Vos fichiers restent en local.

    ---
    """)
