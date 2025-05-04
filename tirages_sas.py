import random
import numpy as np
import pandas as pd
from typing import Dict

# --------------------------------------------------
# SAS sans remise (fonction de base avec vecteur unique)
def sas_sans_remise_base(N, n, random_state=None):
    if random_state is not None:
        random.seed(random_state)
    numero = list(range(1, N+1))
    echantillon = random.sample(numero, n)  # tirage sans remise
    return echantillon

# --------------------------------------------------
# SAS avec remise (fonction de base)
def sas_avec_remise_base(N, n, random_state=None):
    if random_state is not None:
        random.seed(random_state)
    numero = list(range(1, N+1))
    echantillon = [random.choice(numero) for _ in range(n)]
    return echantillon

# --------------------------------------------------
# Tirage draw-by-draw sans remise
def draw_by_draw(N, n, random_state=None):
    if random_state is not None:
        random.seed(random_state)
    numero = list(range(1, N+1))
    echantillon = []
    for _ in range(n):
        choix = random.choice([x for x in numero if x not in echantillon])
        echantillon.append(choix)
    return echantillon

# --------------------------------------------------
# Tirage bernoullien
def tirage_bernoulli(N, n, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    seuil = n / N
    unif = np.random.uniform(0, 1, N)
    echantillon = [i+1 for i in range(N) if unif[i] < seuil]
    return echantillon

# --------------------------------------------------
# Tirage par tri aléatoire
def tri_aleatoire(N, n, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    unif = np.random.uniform(0, 1, N)
    indices_tries = np.argsort(unif)[:n]
    echantillon = [i+1 for i in indices_tries]
    return echantillon

# --------------------------------------------------
# Sélection-rejet
def selection_rejet(N, n, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    echantillon = []
    k = 1
    j = 0
    while j < n and k <= N:
        u = np.random.uniform()
        if u < (n - j) / (N - k + 1):
            echantillon.append(k)
            j += 1
        k += 1
    return echantillon

# --------------------------------------------------
# Mise à jour d'échantillon (type Reservoir Sampling)
def reservoir_sampling(N, n, random_state=None):
    if random_state is not None:
        random.seed(random_state)
    selection = list(range(1, n+1))
    for k in range(n+1, N+1):
        u = random.uniform(0, 1)
        if u < n / k:
            r = random.randint(0, n-1)
            selection[r] = k
    return selection

# --------------------------------------------------
# STRATIFICATION AVEC TOUTES LES MÉTHODES DE TIRAGE
def STRATIFICATION(db, n_par_strate, mode="sas_sans_remise", random_state=None):
    if 'Strate' not in db.columns:
        raise ValueError("La colonne 'Strate' est requise dans la base.")

    def tirage(N, taille, seed):
        if mode == "sas_sans_remise":
            return sas_sans_remise_base(N, taille, random_state=seed)
        elif mode == "sas_avec_remise":
            return sas_avec_remise_base(N, taille, random_state=seed)
        elif mode == "bernoulli":
            return tirage_bernoulli(N, taille, random_state=seed)
        elif mode == "tri_aleatoire":
            return tri_aleatoire(N, taille, random_state=seed)
        elif mode == "selection_rejet":
            return selection_rejet(N, taille, random_state=seed)
        elif mode == "draw_by_draw":
            return draw_by_draw(N, taille, random_state=seed)
        elif mode == "reservoir":
            return reservoir_sampling(N, taille, random_state=seed)
        else:
            raise ValueError("Mode de tirage inconnu : " + mode)

    strates = db['Strate'].unique()
    stratified_sample = pd.DataFrame()
    base_seed = random_state if random_state is not None else np.random.randint(0, 10000)

    for i, strate in enumerate(strates):
        sous_base = db[db['Strate'] == strate].reset_index(drop=True)
        taille = n_par_strate.get(strate, 0) if isinstance(n_par_strate, dict) else n_par_strate
        
        # Vérifier que la taille de l'échantillon n'excède pas le nombre d'éléments dans la strate
        if taille > len(sous_base):
            print(f"Avertissement : La taille de l'échantillon pour la strate '{strate}' ({taille}) est supérieure à la taille de la strate ({len(sous_base)}). Ajustement.")
            taille = len(sous_base)

        indices = tirage(len(sous_base), taille, seed=base_seed + i)
        
        # Ajuster les indices (passer de 1-based à 0-based)
        indices = [x - 1 for x in indices]
        
        print(f"Tirage pour la strate '{strate}': Indices {indices}")

        if indices:  # Vérifier que les indices sont valides
            extrait = sous_base.iloc[indices]
            stratified_sample = pd.concat([stratified_sample, extrait], ignore_index=True)
        else:
            print(f"Aucun indice tiré pour la strate '{strate}'")

    return stratified_sample

# --------------------------------------------------
# Allocation proportionnelle

def allocations_proportionnelles(db: pd.DataFrame, n_total: int) -> Dict[str, int]:
    if 'Strate' not in db.columns:
        raise ValueError("La colonne 'Strate' est requise dans la base.")

    effectifs = db['Strate'].value_counts().to_dict()
    N_total = sum(effectifs.values())
    allocations = {}
    for strate, N_h in effectifs.items():
        allocations[strate] = max(1, round(n_total * N_h / N_total))

    total_alloc = sum(allocations.values())
    if total_alloc != n_total:
        strate_ajust = max(allocations.items(), key=lambda x: x[1])[0]
        allocations[strate_ajust] += n_total - total_alloc

    return allocations

# --------------------------------------------------
# Répartition de Neyman

def repartition_neyman(db: pd.DataFrame, n_total: int, variable: str) -> Dict[str, int]:
    if 'Strate' not in db.columns or variable not in db.columns:
        raise ValueError(f"Les colonnes 'Strate' et '{variable}' sont requises.")

    stats = db.groupby('Strate')[variable].agg(['size', 'var']).to_dict('index')

    for strate in stats:
        if np.isnan(stats[strate]['var']) or stats[strate]['var'] == 0:
            stats[strate]['var'] = db[variable].var()

    numerateurs = {strate: stats[strate]['size'] * np.sqrt(stats[strate]['var']) for strate in stats}
    denominateur = sum(numerateurs.values())

    allocations = {}
    for strate in stats:
        n_h = max(1, round(n_total * numerateurs[strate] / denominateur))
        allocations[strate] = min(n_h, stats[strate]['size'])

    total_alloc = sum(allocations.values())
    if total_alloc != n_total:
        ecarts = {s: (allocations[s] / stats[s]['size']) for s in allocations}
        strate_ajust = max(ecarts.items(), key=lambda x: x[1])[0]
        allocations[strate_ajust] += n_total - total_alloc

    return allocations

