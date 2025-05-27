# Les matrices : de simples tableaux aux réseaux de neurones ! 🔢

## Visualisons les matrices !

Imaginez un jeu vidéo rétro en noir et blanc. Chaque pixel est soit allumé (1) soit éteint (0). L'écran entier n'est rien d'autre qu'un grand tableau de nombres :

```python
# Une petite image 4x4 représentant un "L"
L = [
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 1, 1]
]
```

Voilà, vous venez de voir votre première matrice !

### 1. Qu'est-ce qu'une matrice ?

Une matrice est simplement un tableau rectangulaire de nombres. En mathématiques, on l'écrit souvent entre crochets ou parenthèses :

```python
import numpy as np

# Une matrice 2x3 (2 lignes, 3 colonnes)
matrice = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
```

🤔 **Point de réflexion** : En quoi est-ce différent d'un vecteur ? Un vecteur est comme une matrice avec une seule colonne (ou une seule ligne). On peut voir une matrice comme une collection de vecteurs !

### 2. Les matrices dans la vraie vie

Les matrices sont partout autour de nous :

- Images numériques : chaque pixel est un nombre
- Jeux vidéo : positions des objets dans une grille
- Réseaux sociaux : qui est ami avec qui ?
- Données tabulaires : tableaux Excel

Prenons un exemple concret :

```python
# Notes d'élèves pour différentes matières
notes = np.array([
    #  Math  Phys  Info
    [  18,   15,   17 ],  # Alice
    [  13,   14,   16 ],  # Bob
    [  16,   12,   15 ]   # Charlie
])
```

### 3. Anatomie d'une matrice

Une matrice a plusieurs caractéristiques importantes :

```python
# Création d'une matrice avec numpy
M = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Dimensions
print(f"Forme : {M.shape}")         # (2, 3)
print(f"Nombre de lignes : {M.shape[0]}")    # 2
print(f"Nombre de colonnes : {M.shape[1]}")  # 3

# Accès aux éléments
print(f"Élément (1,2) : {M[1,2]}")  # 6 (2ème ligne, 3ème colonne)
```

### 4. Types spéciaux de matrices

Il existe des matrices avec des propriétés particulières :

```python
# Matrice carrée (autant de lignes que de colonnes)
carree = np.array([
    [1, 2],
    [3, 4]
])

# Matrice identité (1 sur la diagonale, 0 ailleurs)
identite = np.eye(3)

# Matrice nulle (que des zéros)
nulle = np.zeros((2, 3))

# Matrice diagonale (valeurs uniquement sur la diagonale)
diagonale = np.diag([1, 2, 3])
```

### 5. Les matrices en Machine Learning

En ML, les matrices sont fondamentales :

```python
# Un jeu de données : chaque ligne est un exemple
donnees = np.array([
    # taille  poids  age
    [  170,    65,   25 ],  # Personne 1
    [  180,    80,   30 ],  # Personne 2
    [  165,    55,   20 ]   # Personne 3
])

# Une image en niveaux de gris (valeurs de 0 à 255)
image = np.array([
    [128, 255, 128],
    [255,   0, 255],
    [128, 255, 128]
])
```

### ✍️ Mini exercice

Créons une matrice représentant une grille de jeu Tic-tac-toe :

```python
# 0 : case vide, 1 : X, -1 : O
tic_tac_toe = np.array([
    [ 1,  0, -1],
    [ 0,  1,  0],
    [-1,  0,  1]
])

# Vérifions qui est sur la diagonale
diagonale = np.diag(tic_tac_toe)
print("Diagonale :", diagonale)  # [1, 1, 1] -> X gagne !
```

### 🎯 Points clés à retenir

1. Une matrice est un tableau 2D de nombres
2. Les dimensions sont importantes (lignes × colonnes)
3. Il existe des matrices spéciales (carrée, identité...)
4. Les matrices sont partout en ML
5. NumPy est l'outil idéal pour manipuler les matrices

### 🚀 Prochaines étapes

Dans la prochaine leçon, nous verrons :

- Comment combiner les matrices (addition, multiplication)
- Les transformations géométriques avec les matrices
- Les opérations spéciales (transposée, inverse...)
