import streamlit as st
import pandas as pd
import numpy as np
from tirages_sas import STRATIFICATION, allocations_proportionnelles, repartition_neyman
from estimation import tableau_resultats

def page_sas():
    st.title("🎯 Tirage SAS (Sondage Aléatoire Simple)")
    st.markdown("Cette page vous permet d’effectuer un tirage aléatoire simple avec différentes stratégies.")

    if "data" not in st.session_state:
        st.warning("Veuillez d'abord importer une base de données via l'onglet 'Chargement des données'.")
        return

    data = st.session_state["data"].copy()
    colonnes_qualitatives = data.select_dtypes(include=["object", "category"]).columns.tolist()

    st.subheader("🔍 Choix du plan")
    plan = st.radio("Type de tirage", ["Tirage global", "Tirage stratifié"], horizontal=True)

    st.subheader("🎲 Méthode de tirage")
    methode = st.radio("Sélectionnez une méthode :", [
        "sas_sans_remise", "sas_avec_remise", "bernoulli", "tri_aleatoire",
        "selection_rejet", "draw_by_draw", "reservoir"
    ], horizontal=True)

    if plan == "Tirage global":
        n = st.number_input("Taille de l’échantillon souhaitée", min_value=1, max_value=len(data), value=10)
        if st.button("🎲 Lancer le tirage global"):
            try:
                # Suppression des NaN dans l'échantillon global avant le tirage
                data_clean = data.dropna(subset=["Y"])  # Ne garder que les lignes où "Y" n'est pas NaN
                echantillon = STRATIFICATION(data_clean.assign(Strate='A'), {'A': n}, mode=methode)
                st.success(f"✅ Tirage effectué. Échantillon de {len(echantillon)} unités.")
                st.dataframe(echantillon.dropna())

                # ======== Estimation sur la variable Y (tirage global) =========
                st.markdown("---")
                st.subheader("📊 Résultats d'estimation sur la variable d'intérêt")

                if "Y" in echantillon.columns:
                    echantillon_clean = echantillon.dropna(subset=["Y"])  # Filtrer les NaN de Y dans l'échantillon final
                    y = echantillon_clean["Y"]
                    N_pop = len(data_clean)  # Population totale après suppression des NaN dans "Y"
                    pik_value = n / N_pop
                    pik = np.full(len(echantillon_clean), pik_value)

                    rho = 1
                    pik1 = np.outer(pik, pik)
                    pikl = rho * pik1
                    np.fill_diagonal(pikl, pik)

                    resultats = tableau_resultats(y, pik, pikl, N=N_pop, alpha=0.05)
                    st.dataframe(resultats.style.format(precision=3).set_caption("Tableau des résultats statistiques"))

                    with st.expander("ℹ️ Hypothèses utilisées"):
                        st.markdown(""" 
                        Les estimateurs affichés sont :
                        - **Moyenne empirique** 
                        - **Total empirique** 
                        - **Hájek** (moyenne et total)
                        - **Horvitz-Thompson (HT)** pour le total

                        Hypothèses :
                        - Probabilité d’inclusion **constante**
                        - **Indépendance** entre les unités (ρ = 1)

                        > ⚠️ **Attention** : Si ces hypothèses sont fausses, les résultats peuvent être biaisés.
                        """)
                else:
                    st.warning("⚠️ La variable `Y` n’est pas présente dans l’échantillon final.")

            except Exception as e:
                st.error(f"❌ Erreur lors du tirage : {e}")

    else:
        if not colonnes_qualitatives:
            st.error("❌ Aucune variable qualitative n'est disponible pour la stratification.")
            return

        var_strate = st.selectbox("🔢 Variable de stratification", colonnes_qualitatives)

        mode_repartition = st.radio("🧮 Mode de répartition", ["Fixe par strate", "Proportionnelle", "Neyman"], horizontal=True)
        allocations = {}

        try:
            # Suppression des NaN avant de procéder à la répartition
            data_clean = data.dropna(subset=[var_strate, "Y"])  # Supprime les NaN dans les colonnes nécessaires
            if "Strate" in data_clean.columns and var_strate != "Strate":
                data_temp = data_clean.drop(columns=["Strate"]).rename(columns={var_strate: "Strate"})
            else:
                data_temp = data_clean.rename(columns={var_strate: "Strate"})

            data_temp["Strate"] = data_temp["Strate"].astype(str)
            strates = sorted(data_temp["Strate"].unique())

            if mode_repartition == "Fixe par strate":
                st.markdown("Définissez la taille de l’échantillon pour chaque strate :")
                tailles_disponibles = data_temp["Strate"].value_counts().to_dict()
                for strate in strates:
                    max_val = tailles_disponibles.get(strate, 0)
                    taille = st.number_input(f"→ {strate}", min_value=0, max_value=max_val, value=min(2, max_val), step=1, key=f"taille_{strate}")
                    allocations[strate] = taille

            elif mode_repartition == "Proportionnelle":
                n_total = st.number_input("Taille totale de l’échantillon", min_value=1, value=10)
                allocations = allocations_proportionnelles(data_temp, n_total)
                st.success("📊 Répartition proportionnelle calculée :")
                st.write(allocations)

            elif mode_repartition == "Neyman":
                var_quant = st.selectbox("Variable d’intérêt (quantitative)", data.select_dtypes(include='number').columns)
                n_total = st.number_input("Taille totale de l’échantillon", min_value=1, value=10)
                allocations = repartition_neyman(data_temp, n_total, var_quant)
                st.success("📊 Répartition selon Neyman :")
                st.write(allocations)

        except Exception as e:
            st.error(f"❌ Erreur dans la configuration des strates : {e}")
            return

        if st.button("🎯 Lancer le tirage stratifié"):
            try:
                data_clean = data.dropna(subset=[var_strate, "Y"])  # Assurez-vous de retirer les lignes NaN ici aussi
                if "Strate" in data_clean.columns and var_strate != "Strate":
                    data_temp = data_clean.drop(columns=["Strate"]).rename(columns={var_strate: "Strate"})
                else:
                    data_temp = data_clean.rename(columns={var_strate: "Strate"})

                data_temp["Strate"] = data_temp["Strate"].astype(str)
                strates_valides = data_temp["Strate"].unique()
                allocations_valides = {k: v for k, v in allocations.items() if k in strates_valides}

                if not allocations_valides:
                    st.error("❌ Aucune strate valide détectée dans les allocations.")
                    return

                tailles_dispo = data_temp["Strate"].value_counts().to_dict()
                erreurs = []

                for strate, taille_demandee in allocations_valides.items():
                    taille_disponible = tailles_dispo.get(strate, 0)
                    if taille_demandee > taille_disponible:
                        erreurs.append(f"Strate '{strate}' : {taille_demandee} > {taille_disponible}")

                if erreurs:
                    st.error("❌ La taille demandée dépasse la population disponible pour certaines strates :")
                    for msg in erreurs:
                        st.markdown(f"- {msg}")
                    return

                echantillon = STRATIFICATION(data_temp, allocations_valides, mode=methode)
                st.success(f"✅ Tirage réussi. Taille finale de l’échantillon : {len(echantillon)}")
                st.dataframe(echantillon.dropna())  # Filtrer les NaN avant d'afficher l'échantillon final

                csv = echantillon.dropna().to_csv(index=False).encode('utf-8')
                st.download_button("📥 Télécharger l’échantillon", data=csv, file_name="echantillon_SAS.csv", mime='text/csv')

                # ======== Estimation sur la variable Y (tirage stratifié) =========
                st.markdown("---")
                st.subheader("📊 Résultats d'estimation sur la variable d'intérêt")

                if "Y" in echantillon.columns:
                    echantillon_clean = echantillon.dropna(subset=["Y"])

                    if len(echantillon_clean) == 0:
                        st.warning("⚠️ Toutes les observations de la variable `Y` sont manquantes dans l’échantillon.")
                    else:
                        y = echantillon_clean["Y"]
                        N_pop = len(data_clean)  # Population totale après suppression des NaN dans "Y"
                        n = len(echantillon_clean)
                        pik_value = n / N_pop
                        pik = np.full(n, pik_value)

                        rho = 1
                        pik1 = np.outer(pik, pik)
                        pikl = rho * pik1
                        np.fill_diagonal(pikl, pik)

                        resultats = tableau_resultats(y, pik, pikl, N=N_pop, alpha=0.05)
                        st.dataframe(resultats.dropna().style.format(precision=3).set_caption("Tableau des résultats statistiques"))

                    with st.expander("ℹ️ Hypothèses utilisées"):
                        st.markdown(""" 
                        Les estimateurs affichés sont :
                        - **Moyenne empirique** 
                        - **Total empirique** 
                        - **Hájek** (moyenne et total)
                        - **Horvitz-Thompson (HT)** pour le total

                        Hypothèses :
                        - Probabilité d’inclusion **constante**
                        - **Indépendance** entre les unités (ρ = 1)

                        > ⚠️ **Attention** : Si ces hypothèses sont fausses, les résultats peuvent être biaisés.
                        """)
                else:
                    st.warning("⚠️ La variable `Y` n’est pas présente dans l’échantillon final.")

            except Exception as e:
                st.error(f"❌ Erreur pendant le tirage : {e}")
