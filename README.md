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
👉 [https://sondage2degres.streamlit.app](https://sondage2degres.streamlit.app)

*Alternative locale :*
```bash
git clone https://github.com/votrecompte/projet-sondage.git
cd projet-sondage
pip install -r requirements.txt
streamlit run app.py

## 📚 Structure du Projet
.
├── data/                    # Exemples de bases de données
│   ├── population.csv       # Fichier modèle fourni dans l'énoncé
├── modules/
│   ├── tirages_sas.py       # Algorithmes de tirage SAS (Q1)
│   ├── stratification.py    # Sondages stratifiés (Q2)
│   ├── grappes.py           # Sondages par grappes (Q3)
│   └── deux_degres.py       # Sondages multi-étapes (Q4-Q6)
├── app.py                   # Application Streamlit principale
└── requirements.txt         # Dépendances Python
