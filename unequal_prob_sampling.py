import pandas as pd
import numpy as np
import random
from typing import Dict, List, Union

df=pd.read_csv("Base.csv", sep=";")

def piar_defaut(df: pd.DataFrame, n: int, col_id: str, col_poids: str, random_state: int=222) -> pd.DataFrame:
    """
    Sélectionne n lignes d'un DataFrame selon une distribution pondérée par les poids dans col_poids.

    Args:
        df (pd.DataFrame): Le DataFrame source.
        n (int): Le nombre de lignes à sélectionner.
        col_id (str): Nom de la colonne identifiant les lignes.
        col_poids (str): Nom de la colonne contenant les poids P_i.

    Returns:
        pd.DataFrame: Un DataFrame contenant les n lignes sélectionnées.
    """
    # Initialise la graine aléatoire si donné
    if random_state is not None:
        np.random.seed(random_state)

    # Copie pour ne pas modifier l'original
    df = df.copy()

    # Vérifie que les colonnes existent
    if col_id not in df.columns or col_poids not in df.columns:
        raise ValueError("Les colonnes spécifiées n'existent pas dans le DataFrame.")
    
    # Normalisation des poids pour que la somme soit 1
    df['P_normalisé'] = df[col_poids] / df[col_poids].sum()

    # Calcul des F_i (cumulés)
    df['F_i'] = df['P_normalisé'].cumsum()
    df['F_i-1'] = df['F_i'].shift(fill_value=0)

    # Tirages aléatoires et sélection
    u = np.random.uniform(0, 1, size=n)
    sélection = []
    
    for val in u:
        ligne = df[(df['F_i-1'] < val) & (val <= df['F_i'])].iloc[0]
        sélection.append(ligne)

    return pd.DataFrame(sélection)

def piar_lahiri(df: pd.DataFrame, n: int, col_id: str, col_poids: str, random_state: int=222) -> pd.DataFrame:
    """
    Sélectionne n lignes d'un DataFrame selon l'algorithme de rejet basé sur les poids.
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        n (int): Le nombre d’unités à sélectionner.
        col_id (str): Nom de la colonne identifiant chaque ligne.
        col_poids (str): Nom de la colonne contenant les poids P_j.
        
    Returns:
        pd.DataFrame: Un DataFrame contenant les lignes sélectionnées.
    """
    # Initialise la graine aléatoire si donné
    if random_state is not None:
        np.random.seed(random_state)
    # Vérification des colonnes
    if col_id not in df.columns or col_poids not in df.columns:
        raise ValueError("Les colonnes spécifiées n'existent pas dans le DataFrame.")

    df = df.copy()
    N = len(df)
    P_0 = df[col_poids].max()  # Le P_0 est le max des P_j

    sélection = []
    k = 0

    while k < n:
        j = np.random.randint(0, N - 1)  # Tirage aléatoire d’un index
        u = np.random.uniform(0, 1)   # Génération d’un u ~ U[0,1]
        
        Pj = df.iloc[j][col_poids]
        if u * P_0 <= Pj:
            sélection.append(df.iloc[j])
            k += 1

    return pd.DataFrame(sélection)

def pisr_poisson(df: pd.DataFrame, col_id: str, col_pi: str, random_state: int=222) -> pd.DataFrame:
    """
    Effectue un échantillonnage Bernoulli basé sur les probabilités d'inclusion π_i.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        col_id (str): Nom de la colonne identifiant chaque ligne.
        col_pi (str): Nom de la colonne contenant les probabilités d’inclusion π_i (comprises entre 0 et 1).

    Returns:
        pd.DataFrame: Un DataFrame contenant les lignes échantillonnées.
    """
    # Initialise la graine aléatoire si donné
    if random_state is not None:
        np.random.seed(random_state)

    # Vérification des colonnes
    if col_id not in df.columns or (col_pi not in df.columns):
        raise ValueError("Les colonnes spécifiées n'existent pas dans le DataFrame.")

    df = df.copy()
    N = len(df)

    # Génération de N réalisations u_i ~ U[0,1]
    u = np.random.uniform(0, 1, size=N)
    
    # Sélection des lignes où u_i < π_i
    df['u_i'] = u
    échantillon = df[df['u_i'] < df[col_pi]]

    return pd.DataFrame(échantillon)

def pisr_systematique(df: pd.DataFrame, n: int, col_id: str, col_pi: str, random_state: int=222) -> pd.DataFrame:
    """
    Effectue un échantillonnage systématique pondéré à probabilités proportionnelles aux tailles (PPS) avec taille fixe n.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        col_id (str): Nom de la colonne identifiant chaque ligne.
        col_pi (str): Nom de la colonne contenant les π_i, telles que leur somme vaut n (taille de l'échantillon).
        
    Returns:
        pd.DataFrame: Un DataFrame contenant les n lignes sélectionnées.
    """
    # Initialise la graine aléatoire si donné
    if random_state is not None:
        np.random.seed(random_state)

    if col_id not in df.columns or col_pi not in df.columns:
        raise ValueError("Les colonnes spécifiées n'existent pas dans le DataFrame.")

    df = df.copy()
    
    # Calcul des cumuls V_i
    df['V'] = df[col_pi].cumsum()
    df['V_shift'] = df['V'].shift(fill_value=0)

    u = np.random.uniform(0, 1)  # Point de départ aléatoire

    # Calcul des positions de sélection u + k pour k = 0 à n-1
    positions = [u + k for k in range(n)]

    # Sélection des lignes correspondantes aux intervalles V_{i-1} < pos <= V_i
    sélection = pd.DataFrame()
    for pos in positions:
        ligne = df[(df['V_shift'] < pos) & (pos <= df['V'])].head(1)
        sélection = pd.concat([sélection, ligne], axis=0)

    return pd.DataFrame(sélection).reset_index(drop=True)

def pisr_sunter(df: pd.DataFrame, n: int, col_id: str, col_pi: str, random_state=None) -> pd.DataFrame:
    """
    Implémente la méthode de sélection-rejet généralisée pour tirer un échantillon de taille fixe n.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        col_id (str): Nom de la colonne identifiant chaque ligne.
        col_pi (str): Nom de la colonne contenant les probabilités d’inclusion π_i.

    Returns:
        pd.DataFrame: Le DataFrame contenant les unités sélectionnées.
    """

    # Initialise la graine aléatoire si donné
    if random_state is not None:
        np.random.seed(random_state)
    if col_id not in df.columns or col_pi not in df.columns:
        raise ValueError("Les colonnes spécifiées n'existent pas dans le DataFrame.")

    df = df.copy().reset_index(drop=True)
    N = len(df)

    i = 0  # index sur la population
    j = 0  # compteur d'unités sélectionnées
    V = 0  # somme cumulative des π_i
    échantillon = []

    while j < n and i < N:
        pi_i = df.loc[i, col_pi]
        u = np.random.uniform(0, 1)

        if n-V != 0:  # éviter division par 0
            seuil = pi_i * ((n - j) / (n - V))

        if u < seuil:
            échantillon.append(df.loc[i])
            j += 1

        V += pi_i
        i += 1

    return pd.DataFrame(échantillon).reset_index(drop=True)

def unequal_prob_sampling(df, n: Union[int, None], col_id: str, col_pi: Union[int, None], methode: str, random_state: int=222, appliquer_piar: bool = True) -> pd.DataFrame:

    """
    Applique la méthode d'échantillonnage choisie parmi les 5 disponibles.

    Args:
        df (pd.DataFrame): Données sources.
        col_id (str): Colonne identifiant les unités.
        col_pi (str): Colonne des poids ou probabilités.
        méthode (str): Nom de la méthode à appliquer (doit être l'un des 5 noms de fonction).
        appliquer_piar (bool): Si True, la méthode doit être 'piar_defaut' ou 'piar_lahiri'.

    Returns:
        pd.DataFrame: Le résultat de l'échantillonnage.
    """
    if random_state is not None:
        np.random.seed(random_state)
    # Dictionnaire des fonctions disponibles
    fonctions = {
        "piar_defaut": piar_defaut,
        "piar_lahiri": piar_lahiri,
        "pisr_poisson": pisr_poisson,
        "pisr_systematique": pisr_systematique,
        "pisr_sunter": pisr_sunter,
    }

    if methode not in fonctions:
        raise ValueError(f"Méthode '{methode}' non reconnue. Choisissez parmi : {list(fonctions.keys())}")

    if appliquer_piar:
        if methode not in ["piar_defaut", "piar_lahiri"]:
            raise ValueError("Quand 'appliquer_piar=True', la méthode doit être 'piar_defaut' ou 'piar_lahiri'.")
    
    if col_pi is None :
        col_pi="col_pi"
        freq   = df[col_id].value_counts().reset_index(0)
        freq.columns = [col_id, 'effectif']
        if methode in ['piar_defaut', 'piar_lahiri', 'pisr_poisson']:
            freq[col_pi] = freq['effectif']/freq['effectif'].sum()
        else: 
            if n is None:
                raise ValueError("Veuillez fournir la taille n de l'échantillon")
            freq[col_pi] = (freq['effectif']/freq['effectif'].sum())*n
        df_copy=freq[[col_id, col_pi]]

    # Appel de la bonne fonction avec les arguments
    fonction_choisie = fonctions[methode]
    if methode in ['pisr_poisson']:
        return fonction_choisie(df_copy, col_id, col_pi, random_state)
    else :
        if n is None:
            raise ValueError("Veuillez fournir la taille n de l'échantillon")
        return fonction_choisie(df_copy, n, col_id, col_pi, random_state)

sampling=unequal_prob_sampling(df, n=5, col_id="Grappe", col_pi=None, methode="pisr_sunter", appliquer_piar=False)
sampling
