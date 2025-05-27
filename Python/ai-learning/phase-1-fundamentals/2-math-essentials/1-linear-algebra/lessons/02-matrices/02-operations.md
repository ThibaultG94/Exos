# Les opérations sur les matrices : donnons-leur vie ! 🎮

## Des opérations qui ont du sens

Dans la partie précédente, nous avons vu ce qu'étaient les matrices. Maintenant, découvrons comment les manipuler ! Pour rendre ça concret, imaginons que nous développons un jeu 2D avec des transformations graphiques.

### 1. Addition et soustraction

Comme pour les vecteurs, on additionne les matrices élément par élément :

```python
import numpy as np

# Position initiale des objets dans notre jeu
positions = np.array([
    [10, 20],  # Position objet 1
    [30, 40]   # Position objet 2
])

# Déplacement à appliquer
deplacement = np.array([
    [5, -3],   # Déplacement objet 1
    [2,  4]    # Déplacement objet 2
])

# Nouvelles positions
nouvelles_positions = positions + deplacement
print("Nouvelles positions :\n", nouvelles_positions)
```

🤔 **Point de réflexion** : Les dimensions doivent correspondre ! On ne peut pas additionner une matrice 2×3 avec une matrice 3×2.

### 2. Multiplication par un scalaire

Pour changer l'échelle de tous les éléments :

```python
# Image en niveaux de gris (0-255)
image = np.array([
    [100, 150, 200],
    [120, 170, 220]
])

# Assombrir l'image (multiplier par 0.5)
image_sombre = 0.5 * image

# Éclaircir l'image (multiplier par 1.5)
image_claire = 1.5 * image
```

### 3. La multiplication matricielle

C'est l'opération la plus puissante, mais aussi la plus complexe :

```python
# Rotation 2D de 45 degrés
angle = np.pi / 4  # 45° en radians
rotation = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle),  np.cos(angle)]
])

# Point à faire tourner
point = np.array([1, 0])

# Application de la rotation
point_tourne = np.dot(rotation, point)
```

La règle pour multiplier deux matrices :

- Le nombre de colonnes de A doit égaler le nombre de lignes de B
- Chaque élément est la somme des produits des lignes et colonnes

### 4. La transposée

Échanger lignes et colonnes :

```python
# Données de joueurs : lignes = joueurs, colonnes = stats
joueurs = np.array([
    [100, 80, 90],  # PV, Force, Défense du joueur 1
    [120, 75, 85]   # PV, Force, Défense du joueur 2
])

# Transposée : lignes = stats, colonnes = joueurs
stats = joueurs.T
print("Stats par catégorie :\n", stats)
```

### 5. Transformations géométriques

Les matrices sont parfaites pour les transformations 2D/3D :

```python
def creer_rotation(angle_degres):
    """Crée une matrice de rotation 2D"""
    angle = np.deg2rad(angle_degres)
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

def creer_echelle(sx, sy):
    """Crée une matrice de mise à l'échelle"""
    return np.array([
        [sx,  0],
        [0, sy]
    ])

# Application à un polygone
polygone = np.array([
    [0, 0],   # Point 1
    [1, 0],   # Point 2
    [1, 1]    # Point 3
])

# Rotation de 45° puis doublement de la taille
rotation = creer_rotation(45)
echelle = creer_echelle(2, 2)

resultat = polygone @ rotation @ echelle  # @ est l'opérateur de multiplication matricielle
```

### 6. Opérations spéciales

#### La trace (somme de la diagonale)

```python
matrice = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
trace = np.trace(matrice)  # 1 + 5 + 9 = 15
```

#### Le déterminant (pour les matrices carrées)

```python
A = np.array([
    [1, 2],
    [3, 4]
])
det = np.linalg.det(A)  # Pour vérifier si la matrice est inversible
```

### ✍️ Exercice guidé

Créons un système de transformation d'images :

```python
def transformer_image(image, luminosite=1.0, contraste=1.0):
    """Applique des transformations à une image."""
    # Ajustement du contraste autour de la moyenne
    moyenne = np.mean(image)
    resultat = contraste * (image - moyenne) + moyenne

    # Ajustement de la luminosité
    resultat = luminosite * resultat

    # Limiter les valeurs entre 0 et 255
    return np.clip(resultat, 0, 255)

# Test avec une petite image
image = np.array([
    [100, 150, 200],
    [120, 170, 220]
])

transformee = transformer_image(image, luminosite=1.2, contraste=1.5)
```

### 🎯 Points clés à retenir

1. Addition/soustraction : élément par élément
2. Multiplication matricielle : règles spécifiques
3. Transposée : échange lignes et colonnes
4. Transformations géométriques : rotations, échelles...
5. Opérations spéciales : trace, déterminant

### 🚀 Prochaines étapes

Dans la dernière partie, nous verrons :

- Les applications en ML
- La décomposition matricielle
- Les systèmes d'équations linéaires
