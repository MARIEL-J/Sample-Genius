import streamlit as st
import pandas as pd
import numpy as np
from sondage_par_grappes import methodes_tirage
from estimation import tableau_resultats

def page_grappes():
    st.title("📦 Sondage par Grappes")
    st.markdown("Cette page permet de réaliser un tirage par grappes à l’aide de plusieurs méthodes de tirage aléatoire.")

    if "data" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord importer une base de données dans l'onglet **Chargement des données**.")
        return

    df = st.session_state["data"].copy()

    # ======== Étape 1 : Choix de la variable de grappes =========
    st.subheader("🧩 Variable caractérisant les grappes")
    colonnes = df.columns.tolist()

    var_grappe = st.selectbox(
        "Sélectionnez la variable des grappes :",
        options=colonnes,
        help="La variable doit identifier les groupes de votre base (ex : villages, classes, zones...)."
    )

    df = df[df[var_grappe].notna()].copy()
    df[var_grappe] = df[var_grappe].astype(str)

    grappes_uniques = sorted(df[var_grappe].unique())
    N = len(grappes_uniques)

    st.info(f"🔢 Nombre total de grappes disponibles : **{N}**")

    # ======== Étape 2 : Nombre de grappes à tirer =========
    n = st.number_input(
        f"📏 Nombre de grappes à tirer (≤ {N})",
        min_value=1,
        max_value=N,
        value=min(3, N),
        step=1
    )

    # ======== Étape 3 : Choix de la méthode =========
    st.subheader("🎲 Méthode de tirage")

    method_names = [name for _, (name, _) in methodes_tirage.items()]
    method_dict = {name: fct for _, (name, fct) in methodes_tirage.items()}

    method_selected = st.radio(
        "Choisissez une méthode :",
        options=method_names,
        help="Les méthodes déterminent la façon aléatoire dont les grappes sont sélectionnées.",
        horizontal=True
    )

    # ======== Étape 4 : Exécution du tirage =========
    if st.button("🚀 Lancer le tirage"):
        try:
            tirage_function = method_dict[method_selected]
            indices = tirage_function(N, n)

            indices_valides = [i for i in indices if 1 <= i <= N]
            grappes_tirees = [grappes_uniques[i - 1] for i in indices_valides]

            echantillon = df[df[var_grappe].isin(grappes_tirees)].copy()
            echantillon = echantillon.dropna().reset_index(drop=True)

            st.success("✅ Tirage effectué avec succès.")
            st.markdown(f"**📦 Grappes sélectionnées :** `{', '.join(grappes_tirees)}`")
            st.markdown(f"**👥 Taille finale de l’échantillon :** `{len(echantillon)}` individus")

            st.markdown("### 🧾 Échantillon sélectionné")
            st.dataframe(echantillon)

            # Export CSV
            csv = echantillon.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger l'échantillon",
                data=csv,
                file_name="echantillon_grappes.csv",
                mime="text/csv"
            )

            # ======== Étape 5 : Estimation sur la variable Y =========
            st.markdown("---")
            st.subheader("📊 Résultats d'estimation sur la variable d'intérêt")

            if "Y" in echantillon.columns:
                y = echantillon["Y"]
                N_pop = len(echantillon)

                # Hypothèse : tous les individus ont la même probabilité d'inclusion
                pik_value = n / N  # Probabilité d'inclusion constante
                pik = np.full(N_pop, pik_value)

                # Construction d'une matrice pikl cohérente
                # On suppose une dépendance modérée entre unités (rho = 0.9 par exemple)
                rho = 1
                pik1 = np.outer(pik, pik)           # Produit π_i * π_j
                pikl = rho * pik1                   # π_ij = ρ × π_i × π_j
                np.fill_diagonal(pikl, pik)        # π_ii = π_i

                # Appel à la fonction de résultat
                resultats = tableau_resultats(y, pik, pikl, N_pop, alpha=0.05)

                st.dataframe(resultats.style.format(precision=3).set_caption("Tableau des résultats statistiques"))

                with st.expander("ℹ️ Hypothèses utilisées"):
                    st.markdown("""
                    Les estimateurs affichés sont :
                    - **Moyenne empirique** 
                    - **Total empirique** 
                    - **Hájek** (pour la moyenne et le total)
                    - **Horvitz-Thompson (HT)** pour le total

                    - Tous les individus ont une **probabilité d’inclusion identique**.
                    - Les inclusions sont supposées **indépendantes** (aucune dépendance entre les tirages).

                    > ⚠️ **Attention** : Si dans la réalité les probabilités d’inclusion varient ou sont corrélées, les résultats peuvent être **biaisés**.
                    """)
            else:
                st.warning("⚠️ La variable `Y` n’est pas présente dans l’échantillon final.")

        except Exception as e:
            st.error(f"❌ Une erreur est survenue : {e}")
