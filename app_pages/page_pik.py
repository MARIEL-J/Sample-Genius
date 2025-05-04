import streamlit as st
import pandas as pd
import numpy as np
from unequal_prob_sampling import (
    piar_defaut,
    piar_lahiri,
    pisr_poisson,
    pisr_systematique,
    pisr_sunter
)
from estimation import tableau_resultats

def run_proba_inegale_interface(df):
    st.title("🎯 Échantillonnage à probabilités inégales")
    st.markdown("""
    Dans cette section, vous pouvez réaliser un échantillonnage selon des **probabilités d’inclusion inégales**.
    Ces méthodes permettent de **favoriser certaines unités** en fonction de leur poids (taille, importance, etc.).
    """)

    # Vérification des données
    if "data" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord importer une base de données via l’onglet **Chargement des données**.")
        return

    df = st.session_state["data"].copy()

    # Section : Configuration
    with st.expander("⚙️ Paramètres d'échantillonnage", expanded=True):
        st.markdown("Veuillez configurer les paramètres ci-dessous pour réaliser l’échantillonnage :")
        
        col_id = st.selectbox("📌 **Colonne identifiant les unités (ID)**", options=df.columns)
        col_poids = st.selectbox("⚖️ **Colonne des poids / probabilités πᵢ**", options=df.columns)
        
        méthode = st.selectbox("📈 **Méthode d’échantillonnage**", [
            "PIAR - Méthode par défaut (cumuls)",
            "PIAR - Méthode de Lahiri",
            "PISR - Poisson",
            "PISR - Systématique",
            "PISR - Méthode de Sunter"
        ])

        if méthode != "PISR - Poisson":
            n = st.number_input(
                "🔢 **Taille de l’échantillon à tirer (n)**",
                min_value=1,
                max_value=len(df),
                step=1
            )
        else:
            n = None
            st.markdown("ℹ️ La méthode **Poisson** tire un échantillon de taille variable selon les probabilités d’inclusion.")

        bouton = st.button("🎲 Lancer l’échantillonnage")

    if bouton:
        try:
            if méthode == "PIAR - Méthode par défaut (cumuls)":
                résultat = piar_defaut(df, n=n, col_id=col_id, col_poids=col_poids)
            elif méthode == "PIAR - Méthode de Lahiri":
                résultat = piar_lahiri(df, n=n, col_id=col_id, col_poids=col_poids)
            elif méthode == "PISR - Poisson":
                résultat = pisr_poisson(df, col_id=col_id, col_pi=col_poids)
            elif méthode == "PISR - Systématique":
                somme_pi = df[col_poids].sum()
                if abs(somme_pi - n) > 1e-3:
                    st.warning(f"⚠️ La somme des πᵢ est {somme_pi:.2f}, mais elle devrait être proche de n = {n}.")
                résultat = pisr_systematique(df, n=n, col_id=col_id, col_pi=col_poids)
            elif méthode == "PISR - Méthode de Sunter":
                résultat = pisr_sunter(df, n=n, col_id=col_id, col_pi=col_poids)
            else:
                st.error("❌ Méthode non reconnue.")
                return

            st.success("✅ Échantillonnage réalisé avec succès ! Voici l’échantillon obtenu :")
            st.dataframe(résultat)

            csv = résultat.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Télécharger l’échantillon", data=csv, file_name="echantillon_inegal.csv", mime='text/csv')

            # ===== Estimation des paramètres (si Y présente) =====
            st.markdown("---")
            st.subheader("📊 Résultats d'estimation sur la variable d'intérêt")

            if "Y" in résultat.columns:
                echantillon_clean = résultat.dropna(subset=["Y", col_poids])

                if len(echantillon_clean) == 0:
                    st.warning("⚠️ Aucune donnée exploitable pour l'estimation : la variable `Y` est manquante pour toutes les unités sélectionnées.")
                else:
                    y = echantillon_clean["Y"]
                    pik = echantillon_clean[col_poids].astype(float).to_numpy()
                    N_pop = len(df)

                    # Matrice des probabilités d’inclusion doubles (sous hypothèse rho=1)
                    rho = 1
                    pikl = np.outer(pik, pik) * rho
                    np.fill_diagonal(pikl, pik)

                    resultats = tableau_resultats(y, pik, pikl, N=N_pop, alpha=0.05)
                    st.dataframe(resultats.dropna().style.format(precision=3).set_caption("Tableau des résultats statistiques"))

                    with st.expander("ℹ️ Hypothèses utilisées pour l'estimation"):
                        st.markdown("""
                        Les estimateurs affichés sont :
                        - **Moyenne empirique**  
                        - **Total empirique**  
                        - **Hájek** (moyenne et total)  
                        - **Horvitz-Thompson** (total)

                        Hypothèses :
                        - Probabilités d'inclusion fournies via la colonne πᵢ
                        - Indépendance supposée entre les unités (ρ = 1)
                        """)
            else:
                st.info("ℹ️ La variable `Y` n'est pas présente dans l’échantillon — estimation non effectuée.")

        except Exception as e:
            st.error(f"❌ Une erreur est survenue lors de l’échantillonnage : **{str(e)}**")
