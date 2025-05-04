import numpy as np
import pandas as pd
import warnings
from scipy import stats

#############################################################################################
### Fonction pour basique pour le calcul de la moyenne empirique et l'estimateur du total ###
#############################################################################################

def calculer_moyenne_et_ic(data, N, alpha=0.05):
    """
    Calcule la moyenne empirique, son intervalle de confiance asymptotique et l'estimateur du total 
    ainsi que son intervalle de confiance.
    
    Parameters:
    - data (list ou np.array) : échantillon de données
    - N (int) : taille de la population totale
    - alpha (float) : niveau de signification pour l'intervalle de confiance (par défaut 0.05 pour 95%)

    Retourne:
    - moyenne_empirique : la moyenne empirique de l'échantillon (float)
    - ic_moyenne : l'intervalle de confiance asymptotique pour la moyenne (tuple de floats)
    - estimateur_total : l'estimateur du total pour la population (float)
    - ic_total : l'intervalle de confiance pour l'estimateur du total (tuple de floats)
    """
    
    # 1. Calcul de la moyenne empirique et de l'écart type de l'échantillon
    moyenne_empirique = float(np.mean(data))
    ecart_type = float(np.std(data, ddof=1))  # ddof=1 pour l'estimation non biaisée de l'écart-type
    
    # 2. Calcul de l'intervalle de confiance pour la moyenne
    n = len(data)  # Taille de l'échantillon
    z_alpha2 = stats.norm.ppf(1 - alpha / 2)  # Quantile de la loi normale pour l'IC à 1-alpha
    marge_erreur = z_alpha2 * (ecart_type / np.sqrt(n))  # Marge d'erreur pour l'IC
    
    # Intervalle de confiance pour la moyenne
    ic_moyenne = (moyenne_empirique - marge_erreur, moyenne_empirique + marge_erreur)
    
    # 3. Estimateur du total pour la population
    estimateur_total = moyenne_empirique * N
    
    # 4. Intervalle de confiance pour l'estimateur du total
    ic_total = (ic_moyenne[0] * N, ic_moyenne[1] * N)
    
    # Retourner les résultats sous forme de types natifs
    return moyenne_empirique, ic_moyenne, estimateur_total, ic_total

########################################################
### Fonction pour le calcul de l'estimateur de Hájek ###
########################################################


def estimateur_Hajek(y, pik, N=None, type_estimateur="moyenne", alpha=0.05):
    """
    Calcule un estimateur de Hájek (moyenne ou total) à partir des données échantillonnées et des probabilités d'inclusion.
    
    Paramètres
    ----------
    y : array-like
        Valeurs observées de la variable d'intérêt (par exemple, revenus, tailles, etc.).
    
    pik : array-like
        Probabilités d'inclusion associées à chaque unité de l'échantillon (valeurs comprises entre 0 et 1).
    
    N : int, optionnel
        Taille totale de la population. Requise uniquement pour l'estimation du total.
    
    type_estimateur : str, optionnel
        Type d'estimateur à calculer :
        - "moyenne" (par défaut) : retourne l’estimateur de la moyenne de Hájek.
        - "total" : retourne l’estimateur du total de Hájek, pondéré par N.
    
    alpha : float, optionnel
        Niveau de confiance pour l'intervalle de confiance (par défaut à 0.05 pour un IC à 95%).
    
    Retourne
    --------
    dict
        Dictionnaire contenant :
        - estimation : la valeur estimée de la moyenne ou du total selon le type spécifié.
        - variance : l'estimation de la variance.
        - erreur_standard : l'erreur standard associée à l'estimation.
        - borne_inferieure_IC : la borne inférieure de l'intervalle de confiance à alpha.
        - borne_superieure_IC : la borne supérieure de l'intervalle de confiance à alpha.
    
    Exceptions
    ----------
    ValueError :
        - Si y et pik ne sont pas de la même taille.
        - S'il y a des valeurs manquantes (NaN) dans y ou pik.
        - Si N n'est pas fourni pour une estimation de type "total".
    
    Avertissements
    --------------
    Si le type d’estimateur est invalide, l’estimateur de la moyenne est utilisé par défaut.
    """
    
    # Conversion des entrées en arrays numpy pour assurer les calculs
    y = np.asarray(y)
    pik = np.asarray(pik)
    
    # Vérification de la présence de valeurs manquantes
    if np.any(np.isnan(pik)):
        raise ValueError("Il y a des valeurs manquantes dans les probabilités d'inclusion (pik).")
    if np.any(np.isnan(y)):
        raise ValueError("Il y a des valeurs manquantes dans les observations (y).")
    
    # Vérification de la taille des vecteurs y et pik
    if len(y) != len(pik):
        raise ValueError("Les vecteurs y et pik doivent être de même taille.")
    
    # Estimation de l'inverse des probabilités d'inclusion
    w = 1 / pik
    
    # Estimation de la moyenne ou du total selon le type spécifié
    if type_estimateur not in ["total", "moyenne"]:
        warnings.warn("Le type d’estimateur est manquant ou invalide. Par défaut, l’estimateur de la moyenne est utilisé.")
        estimateur_resultat = np.dot(y, w) / np.sum(w)
    elif type_estimateur == "total":
        if N is None:
            raise ValueError("La taille de la population N doit être fournie pour l’estimation du total.")
        estimateur_resultat = N * np.dot(y, w) / np.sum(w)
    else:  # type_estimateur == "moyenne"
        estimateur_resultat = np.dot(y, w) / np.sum(w)
    
    # Estimation de la variance pour le type "moyenne" ou "total"
    variance = np.sum((y - estimateur_resultat) ** 2 * w) / (np.sum(w) ** 2)
    
    # Erreur standard associée à l'estimation
    erreur_standard = np.sqrt(variance)
    
    # Calcul de l'intervalle de confiance à alpha (par défaut à 0.05 pour un IC à 95%)
    z = stats.norm.ppf(1 - alpha / 2)  # Quantile pour l'intervalle de confiance à 95%
    borne_inferieure_IC = estimateur_resultat - z * erreur_standard
    borne_superieure_IC = estimateur_resultat + z * erreur_standard
    
    # Retourne un dictionnaire avec les résultats
    return {
        "estimation": estimateur_resultat,
        "variance": variance,
        "erreur_standard": erreur_standard,
        "borne_inferieure_IC": borne_inferieure_IC,
        "borne_superieure_IC": borne_superieure_IC
    }



###################################################################
### Fonction pour le calcul de l'estimateur de Horvitz-Thompson ###
###################################################################

import numpy as np
from scipy import stats

def estimateur_HT_IC_exact(y, pikl, method=1, alpha=0.05):
    """
    Calcule l'estimateur de Horvitz-Thompson (HT) du total d'une variable, 
    sa variance exacte basée sur les probabilités d'inclusion doubles (π_ij),
    et un intervalle de confiance asymptotique.

    Paramètres
    ----------
    y : array-like
        Valeurs observées de la variable d’intérêt (ex. : revenus, tailles, etc.) pour les unités de l'échantillon.
    pikl : 2D array-like (matrice carrée)
        Matrice des probabilités d’inclusion doubles π_ij. Les éléments diagonaux doivent être les π_i.
    method : int, optionnel
        Méthode pour le calcul de la variance :
        - 1 : méthode classique (produits croisés)
        - 2 : méthode par différences pondérées
    alpha : float, optionnel
        Niveau de risque pour l’intervalle de confiance (par défaut 0.05 pour un IC à 95%).

    Retourne
    --------
    dict
        Dictionnaire contenant :
        - "HT" : estimation du total par l'estimateur Horvitz-Thompson
        - "variance" : estimation de la variance
        - "erreur_standard" : écart-type (racine carrée de la variance)
        - "IC_borne_inf" : borne inférieure de l’intervalle de confiance
        - "IC_borne_sup" : borne supérieure de l’intervalle de confiance
    """

    # Conversion en tableaux numpy
    y = np.asarray(y, dtype=float)
    pikl = np.asarray(pikl, dtype=float)

    # Vérification de la présence de valeurs manquantes
    if np.any(np.isnan(y)):
        raise ValueError("Il y a des valeurs manquantes dans y.")
    if np.any(np.isnan(pikl)):
        raise ValueError("Il y a des valeurs manquantes dans pikl.")

    # Vérifie que pikl est une matrice carrée
    if pikl.ndim != 2 or pikl.shape[0] != pikl.shape[1]:
        raise ValueError("pikl doit être une matrice carrée.")

    # Vérifie que la taille de y correspond à la taille de pikl
    if len(y) != pikl.shape[0]:
        raise ValueError("La taille de y ne correspond pas à celle de pikl.")

    # Vérifie que la méthode est bien 1 ou 2
    if method not in [1, 2]:
        raise ValueError("La méthode doit être 1 ou 2.")

    # Récupération des probabilités d'inclusion simples π_i (diagonale de pikl)
    pik = np.diag(pikl)

    # Vérifie qu'aucune des π_i n'est nulle (évite division par zéro)
    if np.any(pik == 0):
        raise ValueError("Certaines probabilités d'inclusion π_i sont nulles.")

    # Produit extérieur des probabilités simples π_i * π_j
    pik1 = np.outer(pik, pik)

    # Calcul de la matrice delta = π_ij - π_i * π_j
    delta = pikl - pik1

    # Remplace les diagonales de delta par π_i * (1 - π_i), pour la variance individuelle
    np.fill_diagonal(delta, pik * (1 - pik))

    # Calcul de l'estimateur Horvitz-Thompson : somme des y_i / π_i
    HT = np.sum(y / pik)

    # === Méthode 1 : produits croisés ===
    if method == 1:
        # Produit extérieur des valeurs y_i * y_j
        y1 = np.outer(y, y)

        # Dénominateur : π_i * π_j * π_ij
        denom = pik1 * pikl

        # Remplace les zéros du dénominateur par NaN pour éviter les divisions invalides
        denom[denom == 0] = np.nan

        # Calcul de la variance avec les pondérations exactes
        variance = np.nansum(y1 * delta / denom)

    # === Méthode 2 : différences pondérées ===
    else:
        # Calcul de y_i / π_i pour toutes les unités
        y_ratio = y / pik

        # Différences au carré entre chaque paire (i, j)
        y_diff = np.subtract.outer(y_ratio, y_ratio) ** 2

        # Dénominateur : π_ij
        denom = pikl.copy()
        denom[denom == 0] = np.nan  # Évite les divisions par zéro

        # Calcul de la variance basée sur les différences
        variance = 0.5 * np.nansum(y_diff * (pik1 - pikl) / denom)

    # Si la variance est invalide (NaN ou négative), on évite les erreurs
    if np.isnan(variance) or variance < 0:
        erreur_standard = np.nan
        IC_inf = np.nan
        IC_sup = np.nan
    else:
        # Écart-type de l'estimateur HT
        erreur_standard = np.sqrt(variance)

        # Quantile normal pour l'IC
        z = stats.norm.ppf(1 - alpha / 2)

        # Bornes de l'intervalle de confiance
        IC_inf = HT - z * erreur_standard
        IC_sup = HT + z * erreur_standard

    # Renvoi des résultats sous forme de dictionnaire
    return {
        "HT": HT,
        "variance": variance,
        "erreur_standard": erreur_standard,
        "IC_borne_inf": IC_inf,
        "IC_borne_sup": IC_sup
    }


###################################
#### FONCTION POUR RECAPITULER ####
###################################


def tableau_resultats(y, pik, pikl, N, alpha=0.05):
    # Calcul de la moyenne empirique, intervalle de confiance et total empirique
    moyenne, ic_m, total, ic_t = calculer_moyenne_et_ic(y, N, alpha)
    
    # Estimation de Hajek pour la moyenne et le total
    hajek_moyenne = estimateur_Hajek(y, pik, N, "moyenne", alpha)  # Estimation de la moyenne avec Hajek
    hajek_total = estimateur_Hajek(y, pik, N, "total", alpha)      # Estimation du total avec Hajek
    
    # Estimation de l'Horvitz-Thompson
    ht = estimateur_HT_IC_exact(y, pikl, method=1, alpha=alpha)

    # Construction du dataframe avec tous les résultats
    df = pd.DataFrame({
        "Estimateur": [
            "Moyenne empirique", 
            "Total empirique", 
            "Hajek (moyenne)", 
            "Hajek (total)", 
            "Horvitz-Thompson"
        ],
        "Estimation": [
            moyenne, 
            total, 
            hajek_moyenne["estimation"], 
            hajek_total["estimation"], 
            ht["HT"]
        ],
        "Erreur standard": [
            "-", 
            "-", 
            hajek_moyenne["erreur_standard"], 
            hajek_total["erreur_standard"], 
            "-"
        ],
        "IC min": [
            ic_m[0], 
            ic_t[0], 
            hajek_moyenne["borne_inferieure_IC"], 
            hajek_total["borne_inferieure_IC"], 
            "-"
        ],
        "IC max": [
            ic_m[1], 
            ic_t[1], 
            hajek_moyenne["borne_superieure_IC"], 
            hajek_total["borne_superieure_IC"], 
            "-"
        ]
    })

    # Retourner le tableau avec les résultats arrondis à trois chiffres
    return df.round(3)
