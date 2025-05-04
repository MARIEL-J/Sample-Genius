import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict
from tirages_sas import sas_sans_remise_base, sas_avec_remise_base, draw_by_draw, tirage_bernoulli, tri_aleatoire, selection_rejet, reservoir_sampling

# Fonction de tirage stratifié
def strata(
    data: pd.DataFrame,  # Données source
    stratanames: Optional[List[str]],  # Variables de stratification
    size: Union[int, List[int], Dict],  # Taille(s) d'échantillon par strate
    method: str = "sas_sans_remise",  # Méthode de tirage par défaut
    description: bool = False,  # Affiche les descriptions intermédiaires
    stage_num: int = 1  # Numéro de l’étape (utile pour multi-degrés)
) -> pd.DataFrame:
    """
    Réalise un tirage stratifié selon différentes méthodes, avec tirage uniforme.
    """
    # Vérifie que la méthode choisie est bien parmi les options autorisées
    if method not in ["sas_sans_remise", "sas_avec_remise", "draw_by_draw", "tirage_bernoulli", "tri_aleatoire", "selection_rejet", "reservoir_sampling"]:
        raise ValueError("Méthode non reconnue. Options: 'sas_sans_remise', 'sas_avec_remise', 'draw_by_draw', 'tirage_bernoulli', 'tri_aleatoire', 'selection_rejet', 'reservoir_sampling'")

    samples = []  # Liste pour stocker les échantillons par strate

    # Si stratification activée
    if stratanames is not None:
        grouped = data.groupby(stratanames)  # Regroupe les données selon les strates

        size_list = size.copy() if isinstance(size, list) else None  # Copie défensive si taille est une liste

        # Boucle sur chaque strate
        for idx, (name, group) in enumerate(grouped):
            # Détermine la taille de l’échantillon selon le format de `size`
            if isinstance(size, dict):
                key = name if not isinstance(name, tuple) else name[0]  # Clé simple ou tuple
                n = size.get(key, 0)  # Taille pour cette strate
            elif isinstance(size, list):
                n = size_list[idx] if idx < len(size_list) else 0
            else:
                n = size  # Même taille pour toutes les strates

            if description:
                print(f"→ Strate: {name}, taille groupe: {len(group)}, taille demandée: {n}")

            n = min(n, len(group))  # Ne pas tirer plus que la taille réelle du groupe
            if n == 0:
                if description:
                    print(f"⚠️  Aucune unité tirée pour la strate {name}")
                continue  # Passe à la strate suivante

            # Tirage uniforme sans remise
            if method == "sas_sans_remise":
                sample = np.random.choice(group.index, n, replace=False)  # Tirage sans remise
                prob = n / len(group)
                sample = group.loc[sample]  # Applique les indices de l'échantillon
            elif method == "sas_avec_remise":
                sample = np.random.choice(group.index, n, replace=True)  # Tirage avec remise
                prob = n / len(group)
                sample = group.loc[sample]
            elif method == "draw_by_draw":
                sample = np.random.choice(group.index, n, replace=False)  # Tirage draw-by-draw sans remise
                prob = n / len(group)
                sample = group.loc[sample]
            elif method == "tirage_bernoulli":
                sample = np.random.choice(group.index, n, replace=False)  # Tirage bernoullien sans remise
                prob = n / len(group)
                sample = group.loc[sample]
            elif method == "tri_aleatoire":
                sample = np.random.choice(group.index, n, replace=False)  # Tirage par tri aléatoire sans remise
                prob = n / len(group)
                sample = group.loc[sample]
            elif method == "selection_rejet":
                sample = np.random.choice(group.index, n, replace=False)  # Sélection-rejet sans remise
                prob = n / len(group)
                sample = group.loc[sample]
            elif method == "reservoir_sampling":
                sample = np.random.choice(group.index, n, replace=False)  # Reservoir Sampling sans remise
                prob = n / len(group)
                sample = group.loc[sample]

            sample[f'Prob_{stage_num}_stage'] = prob  # Ajoute la probabilité d'inclusion
            samples.append(sample)  # Ajoute l’échantillon de la strate à la liste

        if not samples:
            raise ValueError("⚠️ Aucune strate n'a été échantillonnée.")

        sampled = pd.concat(samples)  # Fusionne tous les échantillons

    else:
        # Tirage non stratifié
        n = size
        if isinstance(n, list):
            n = n[0]  # Prend la première valeur si une liste

        # Tirage uniforme sans remise
        if method == "sas_sans_remise":
            sampled = np.random.choice(data.index, n, replace=False)  # Tirage sans remise
            prob = n / len(data)
            sampled = data.loc[sampled]  # Applique les indices de l'échantillon
        elif method == "sas_avec_remise":
            sampled = np.random.choice(data.index, n, replace=True)  # Tirage avec remise
            prob = n / len(data)
            sampled = data.loc[sampled]
        elif method == "draw_by_draw":
            sampled = np.random.choice(data.index, n, replace=False)  # Tirage draw-by-draw sans remise
            prob = n / len(data)
            sampled = data.loc[sampled]
        elif method == "tirage_bernoulli":
            sampled = np.random.choice(data.index, n, replace=False)  # Tirage bernoullien sans remise
            prob = n / len(data)
            sampled = data.loc[sampled]
        elif method == "tri_aleatoire":
            sampled = np.random.choice(data.index, n, replace=False)  # Tirage par tri aléatoire sans remise
            prob = n / len(data)
            sampled = data.loc[sampled]
        elif method == "selection_rejet":
            sampled = np.random.choice(data.index, n, replace=False)  # Sélection-rejet sans remise
            prob = n / len(data)
            sampled = data.loc[sampled]
        elif method == "reservoir_sampling":
            sampled = np.random.choice(data.index, n, replace=False)  # Reservoir Sampling sans remise
            prob = n / len(data)
            sampled = data.loc[sampled]

        sampled[f'Prob_{stage_num}_stage'] = prob  # Ajoute la probabilité d'inclusion

    return sampled  # Retourne l’échantillon

def cluster(
    data: pd.DataFrame,  # Données d’entrée
    clustername: Union[str, List[str]],  # Nom ou liste des variables de grappes
    size: Union[int, List[int]],  # Nombre de grappes à sélectionner
    method: str = "sas_sans_remise",  # Méthode de tirage
    description: bool = False,  # Affichage des étapes
    stage_num: int = 1  # Numéro d’étape (multi-degrés)
) -> pd.DataFrame:
    """
    Réalise un tirage par grappes avec tirage uniforme.
    """
    # Si clustername est une liste, on ne garde que la première variable
    if isinstance(clustername, list):
        clustername = clustername[0]

    # Si size est une liste, on ne garde que la première valeur
    if isinstance(size, list):
        size = size[0]

    clusters = data[clustername].unique()  # Liste des grappes uniques
    n_clusters = len(clusters)  # Nombre total de grappes
    size = min(size, n_clusters)  # Ajuste la taille demandée si > nombre de grappes

    # Tirage uniforme sans remise
    selected_clusters = np.random.choice(clusters, size, replace=False)  # Tirage sans remise
    prob = size / n_clusters

    # Filtre les données pour ne garder que les grappes sélectionnées
    sampled = data[data[clustername].isin(selected_clusters)].copy()

    # Ajoute la probabilité d’inclusion
    sampled[f'Prob_{stage_num}_stage'] = prob if isinstance(prob, (int, float)) else np.repeat(prob.mean(), len(sampled))

    return sampled  # Retourne les unités sélectionnées

def sample_degree(
    data: pd.DataFrame,  # Données complètes à échantillonner
    size: Union[List[int], List[List[int]], int],  # Taille d’échantillon à chaque étape
    stage: Optional[List[str]] = None,  # Liste des types d’étapes (e.g. "stratified", "cluster")
    varnames: Optional[Union[List[str], List[List[str]]]] = None,  # Variables de stratification/grappes
    method: Optional[Union[List[str], str]] = None,  # Méthodes de tirage à chaque étape
    description: bool = False  # Affichage des étapes
) -> Dict[int, pd.DataFrame]:
    """
    Réalise un plan de sondage à plusieurs degrés avec tirage uniforme.
    """
    # Vérifie que la taille est bien fournie
    if size is None:
        raise ValueError("La taille d'échantillon doit être spécifiée")

    # Si le plan implique des strates ou grappes, les variables doivent être spécifiées
    if stage is not None and varnames is None:
        raise ValueError("Les variables doivent être spécifiées pour les étapes stratifiées ou par grappes")

    # Détermine le nombre d’étapes
    number = len(stage) if stage is not None else len(size) if isinstance(size, list) else 1

    # Mise en forme des listes pour assurer une consistance
    size_list = size if isinstance(size, list) else [size]
    method_list = method if isinstance(method, list) else [method] * number if method else ["sas_sans_remise"] * number
    varnames_list = varnames if isinstance(varnames, list) and isinstance(varnames[0], list) else [varnames] * number

    results = {}  # Dictionnaire pour stocker les résultats

    # Effectuer chaque étape du plan de sondage
    for i in range(number):
        # Extraction des paramètres pour cette étape
        stage_size = size_list[i]
        stage_method = method_list[i]
        stage_varnames = varnames_list[i]

        if description:
            print(f"\nÉtape {i + 1} - Méthode: {stage_method}, Taille échantillon: {stage_size}, Variables: {stage_varnames}")

        # Applique un tirage en fonction du type d’étape (stratification, grappes ou tirage global)
        if "stratified" in stage[i]:
            sampled = strata(data, stage_varnames, stage_size, method=stage_method, description=description, stage_num=i+1)
        elif "cluster" in stage[i]:
            sampled = cluster(data, stage_varnames, stage_size, method=stage_method, description=description, stage_num=i+1)
        else:
            # Tirage simple dans le cas où il n'y a ni stratification ni grappes
            sampled = strata(data, None, stage_size, method=stage_method, description=description, stage_num=i+1)

        results[i] = sampled  # Sauvegarde du résultat de cette étape

    return results  # Retourne les résultats par étape
