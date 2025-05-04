import pandas as pd
import numpy as np
import random

# Fonctions de tirage
def tirage_sas_sans_remise(N, n):
    population = list(range(1, N + 1))
    return random.sample(population, n)

def tirage_sas_avec_remise(N, n):
    population = list(range(1, N + 1))
    return [random.choice(population) for _ in range(n)]

def tirage_draw_by_draw(N, n):
    population = list(range(1, N + 1))
    echantillon = [random.choice(population)]
    while len(echantillon) < n:
        candidat = random.choice(population)
        if candidat not in echantillon:
            echantillon.append(candidat)
    return echantillon

def tirage_bernoulli(N, n):
    unif = np.random.uniform(0, 1, N)
    echantillon = [i + 1 for i in range(N) if unif[i] < n / N]
    return echantillon[:n]

def tirage_tri_aleatoire(N, n):
    unif = np.random.uniform(0, 1, N)
    indices_trie = np.argsort(unif)
    return (indices_trie[:n] + 1).tolist()

def tirage_selection_rejet(N, n):
    echantillon = []
    k = 1
    j = 0
    while j < n and k <= N:
        u = random.uniform(0, 1)
        if u < (n - j) / (N - k + 1):
            echantillon.append(k)
            j += 1
        k += 1
    return echantillon

def tirage_mise_a_jour(N, n):
    selection = list(range(1, n + 1))
    for k in range(n + 1, N + 1):
        u = random.uniform(0, 1)
        if u < n / k:
            c = random.choice(selection)
            r = selection.index(c)
            selection[r] = k
    return selection

# Dictionnaire des méthodes
methodes_tirage = {
    1: ("SAS sans remise", tirage_sas_sans_remise),
    2: ("SAS avec remise", tirage_sas_avec_remise),
    3: ("Draw by draw", tirage_draw_by_draw),
    4: ("Tirage Bernoullien", tirage_bernoulli),
    5: ("Tri aléatoire", tirage_tri_aleatoire),
    6: ("Sélection-Rejet", tirage_selection_rejet),
    7: ("Mise à jour échantillon", tirage_mise_a_jour)
}

def sondage_par_grappes():
    # Fichier source
    chemin = input("Chemin vers le fichier CSV : ")
    df = pd.read_csv(chemin, sep=';')

    print("\n=== Base de données chargée ===")
    print(df.head())

    if 'Num' not in df.columns or 'Grappe' not in df.columns:
        raise ValueError("Le fichier doit contenir les colonnes 'Num' et 'Grappe'.")

    # Correction : Suppression des valeurs manquantes dans 'Num'
    df = df.dropna(subset=['Num'])  # enlève les NaN de 'Num'
    df['Num'] = df['Num'].astype(int)  # Convertit 'Num' en entier
    df['Grappe'] = df['Grappe'].astype(str)  # Assure que 'Grappe' est de type str

    grappes_uniques = sorted(df['Grappe'].unique())
    N = len(grappes_uniques)

    # Affichage des méthodes
    print("\nMéthodes de tirage disponibles :")
    for num, (nom, _) in methodes_tirage.items():
        print(f"{num}: {nom}")

    # Choix de méthode
    try:
        choix = int(input("Choisissez la méthode de tirage (défaut = 1) : ") or 1)
        nom_methode, fonction = methodes_tirage.get(choix, methodes_tirage[1])
    except (ValueError, KeyError):
        nom_methode, fonction = methodes_tirage[1]

    # Nombre de grappes à tirer
    n_grappes = int(input(f"Nombre de grappes à tirer (max = {N}) : "))
    if n_grappes > N:
        raise ValueError("Le nombre de grappes demandées est supérieur à ce qui est disponible.")

    # Tirage des indices
    indices_grappes_tirees = fonction(N, n_grappes)
    grappes_tirees = [grappes_uniques[i - 1] for i in indices_grappes_tirees if i <= N]

    # Sous-échantillon correspondant
    echantillon = df[df['Grappe'].isin(grappes_tirees)]

    # Nombre d’individus par grappe
    individus_par_grappe = echantillon.groupby('Grappe')['Num'].count()

    # Résultats
    individus_tires = echantillon['Num'].tolist()
    individus_tires = [int(i) for i in individus_tires]  # convertit en entiers

    print("\n=== RÉSULTATS ===")
    print("Méthode :", nom_methode)
    print("Grappes tirées :", grappes_tirees)
    print("\nNombre d’individus par grappe :")
    print(individus_par_grappe)
    print("\nNuméros des individus tirés :")
    print(individus_tires)
    print("\nTaille totale de l’échantillon :", len(individus_tires))

    return grappes_tirees, individus_par_grappe, individus_tires

# Exécution
if __name__ == "__main__":
    sondage_par_grappes()
