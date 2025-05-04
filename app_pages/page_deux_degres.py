import streamlit as st
import pandas as pd
import numpy as np
from sondage_deux_degres import sample_degree
from tirages_sas import sas_sans_remise_base, sas_avec_remise_base, draw_by_draw, tirage_bernoulli, tri_aleatoire, selection_rejet, reservoir_sampling
from estimation import tableau_resultats  # Assurez-vous que la fonction tableau_resultats est disponible

# Dictionnaire d'affichage utilisateur vers noms internes
method_labels = {
    "Tirage SAS sans remise": "sas_sans_remise",
    "Tirage SAS avec remise": "sas_avec_remise",
    "Draw-by-draw": "draw_by_draw",
    "Bernoulli": "tirage_bernoulli",
    "Tri aléatoire": "tri_aleatoire",
    "Sélection-rejet": "selection_rejet",
    "Reservoir sampling": "reservoir_sampling"
}

# Dictionnaire des fonctions correspondantes si besoin
method_functions = {
    "sas_sans_remise": sas_sans_remise_base,
    "sas_avec_remise": sas_avec_remise_base,
    "draw_by_draw": draw_by_draw,
    "tirage_bernoulli": tirage_bernoulli,
    "tri_aleatoire": tri_aleatoire,
    "selection_rejet": selection_rejet,
    "reservoir_sampling": reservoir_sampling
}

def page_deux_degres():
    st.title("🔁 Sondage à Degrés Multiples")
    
    st.markdown("""
    ## Guide d'utilisation

    Cette interface permet de configurer et exécuter des plans de sondage complexes:

    1️⃣ **Charger vos données** dans l'onglet 'Chargement des données'  
    2️⃣ **Configurer les étapes** de tirage ci-dessous  
    3️⃣ **Lancer le sondage** et analyser les résultats  

    ### Types de tirage disponibles:
    - **Stratifié** : Tirage dans des sous-populations homogènes  
    - **Par grappes** : Tirage de groupes puis d'unités dans les groupes  
    - **Aléatoire simple** : Tirage sans stratification  
    """)

    if "data" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord importer une base de données dans l'onglet 'Chargement des données'")
        return

    data = st.session_state["data"]
    colonnes = list(data.columns)

    st.subheader("🔢 Nombre de degrés à simuler")
    nb_degres = st.radio(
        "Sélectionnez le nombre d'étapes:",
        [2, 3],
        horizontal=True,
        help="Choisissez entre 2 ou 3 étapes de tirage"
    )

    st.subheader("📌 Paramétrage des étapes")
    varnames = []
    size = []
    methods = []
    variables_disponibles = colonnes.copy()

    for i in range(nb_degres):
        st.markdown(f"### 🎯 Étape {i+1}")
        col1, col2 = st.columns(2)

        with col1:
            variable = st.selectbox(
                f"Variable pour l'étape {i+1}",
                variables_disponibles,
                key=f"var_{i}",
                help="Sélectionnez la variable de stratification ou de grappe"
            )
        varnames.append(variable)
        variables_disponibles = [v for v in variables_disponibles if v != variable]

        with col2:
            methode_label = st.selectbox(
                f"Méthode de tirage - Étape {i+1}",
                list(method_labels.keys()),
                key=f"meth_{i}",
                help="Choisissez une méthode de tirage"
            )
            methode = method_labels[methode_label]
        methods.append(methode)

        if i == 0:
            taille_texte = st.text_input(
                "Taille par strate (format dictionnaire)",
                value="{'Q': 2, 'R': 2, 'S': 2, 'T': 2}",
                key=f"taille_{i}",
                help="""Format: {'strata1': taille1, 'strata2': taille2}"""
            )
            
            with st.expander("ℹ️ Aide sur le format des tailles"):
                st.markdown("""
                **Format requis:**  
                ```python
                {'NOM_STRATE': taille, ...}
                ```

                **Exemple:**  
                ```python
                {'Urbain': 5, 'Rural': 3}
                ```
                - Tirera 5 grappes en zone urbaine  
                - Tirera 3 grappes en zone rurale  

                Les noms doivent correspondre exactement aux valeurs dans vos données.
                """)

            try:
                taille1 = eval(taille_texte)
                if not isinstance(taille1, dict):
                    raise ValueError("Le format doit être un dictionnaire")
                size.append(taille1)
            except Exception as e:
                st.error(f"❌ Format incorrect: {e}")
                return
        else:
            taille_val = st.number_input(
                f"Taille d'échantillon - Étape {i+1}",
                min_value=1,
                value=5,
                step=1,
                key=f"taille_{i}",
                help="Nombre d'unités à tirer à cette étape"
            )
            size.append(taille_val)

    if st.button("🚀 Lancer le plan de sondage", help="Exécute le tirage selon la configuration"):
        with st.spinner("Tirage en cours..."):
            try:
                # Assurez-vous que 'methods' contient bien des chaînes de caractères, et 'size' est une liste correcte
                methods_str = [str(method) for method in methods]
                
                res = sample_degree(
                    data=data,
                    size=size,
                    stage=["stratified"] + ["cluster"] * (nb_degres - 1),
                    varnames=varnames,
                    method=methods_str,  # Nous passons une liste de chaînes de caractères
                    description=True
                )

                st.success("✅ Tirage terminé avec succès!")
                
                final_sample = res[max(res.keys())]
                total_final = len(final_sample)
                st.markdown(f"### 📊 **Taille de l'échantillon final: {total_final} unités**")

                # Affichage des étapes
                for i, df in res.items():
                    st.markdown(f"#### 📁 Étape {i+1} - {len(df)} unités sélectionnées")
                    
                    with st.expander(f"Voir les détails de l'étape {i+1}"):
                        st.dataframe(df.head(50))
                        st.write(f"Colonnes: {list(df.columns)}")

                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"💾 Télécharger Étape {i+1}",
                            data=csv,
                            file_name=f"sondage_etape_{i+1}.csv",
                            mime="text/csv",
                            key=f"dl_{i}"
                        )

                # Calcul des estimateurs après la présentation de l'échantillon final
                if "Y" in final_sample.columns:
                    final_sample_clean = final_sample.dropna(subset=["Y"])

                    # Si l'échantillon est vide après avoir supprimé les NaN
                    if final_sample_clean.empty:
                        st.warning("⚠️ Toutes les observations de la variable `Y` sont manquantes dans l’échantillon.")
                    else:
                        # Calcul des estimateurs
                        y = final_sample_clean["Y"]
                        N_pop = len(data)  # Population totale connue
                        n = len(final_sample_clean)
                        pik_value = n / N_pop
                        pik = np.full(n, pik_value)

                        rho = 1
                        pik1 = np.outer(pik, pik)
                        pikl = rho * pik1
                        np.fill_diagonal(pikl, pik)

                        # Estimation des résultats
                        resultats = tableau_resultats(y, pik, pikl, N=N_pop, alpha=0.05)
                        st.dataframe(resultats.dropna().style.format(precision=3).set_caption("Tableau des résultats statistiques"))

            except Exception as e:
                st.error(f"❌ Erreur lors du tirage: {str(e)}")
                st.error("Veuillez vérifier votre configuration et réessayer.")

# Point d'entrée
if __name__ == "__main__":
    page_deux_degres()
