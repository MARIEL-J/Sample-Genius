# 📊 Sondage2Degrés - Application de Plans de Sondage

*Application Streamlit pour la simulation de plans de sondage suivant différents plans : stratifiés, par grappes, à 2 ou 3degrés multiples, à probabilités inégales*

## 🌟 Fonctionnalités
- **Méthodes implémentées** :
  - Sondages Aléatoires Simples (avec/sans remise)
  - Sondages stratifiés
  - Sondages par grappes
  - Sondages à 2/3 degrés
  - Sondages à probabilités inégales (PPS)
- **Algorithmes disponibles** :
  - Tirage SAS (base/draw-by-draw)
  - Bernoulli / Tri aléatoire
  - Sélection-rejet / Reservoir sampling

## 🚀 Déploiement
L'application est hébergée sur Streamlit Cloud :  
👉 [https://sondage2degres.streamlit.app](https://samplegenius-ise.streamlit.app/)

*Alternative locale :*
```bash
git clone https://github.com/votrecompte/projet-sondage.git
cd projet-sondage
pip install -r requirements.txt
streamlit run main.py

## 📚 Structure du Projet
.
├── Base.csv/                # Exemples de bases de données
├── app_pages/
│   ├── pages_deux_degres.py       # Sondage à 2 ou 3 degrés
│   ├── page_grappes.py            # Sondage par grappes
│   ├── pages_home.py              # Page d'acceuil
│   └── page_pik.py                # Pour le sondage à proba inégale
│   └── page_sas.py                # Pour faire du SAS ou de la stratification
│   └── page_team.py               # Présentation de l'équipe de développement
│   └── page_upload.py             # Pour charger la base
├── app.py                         # Application Streamlit principale
└── estimation.py                  # Codes pour le calcul des différents estimateurs : la moyenne et le total empirique, l'estimateur de Hajek et celui de Horvitz                                       Thompson ainsi que les intervalles de confiances 
└── requirements.txt               # Dépendances Python
└── sondage_deux_degres.py         # Codes pour 2 et/ou 3 degrés
└── sondage_par_grappes.py         # Codes pour sondage par grappes
└── tirage_sas.py                  # Codes pour SAS
└── unequal_prob_sampling.py       # Codes pour sondage à proba inégale

## 🧑‍💻 Utilisation

- Charger vos données (CSV/Excel) via l'interface
- Paramétrer les étapes (tailles, méthodes, etc.)
- Exporter les résultats (CSV/Excel)

