# Les opérations sur les vecteurs : donnons vie aux nombres ! 🎮

## De la théorie à la pratique

Dans la partie précédente, nous avons vu ce qu'était un vecteur. Maintenant, découvrons comment les manipuler ! Pour rendre ça concret, imaginons que nous développons un petit jeu de combat.

### 1. L'addition de vecteurs

Imaginez que votre personnage se déplace. Sa nouvelle position est la somme de sa position actuelle et de son déplacement :

```python
import numpy as np

# Position initiale (x, y)
position = np.array([10, 20])

# Déplacement
deplacement = np.array([5, -3])  # 5 vers la droite, 3 vers le bas

# Nouvelle position
nouvelle_position = position + deplacement
print(f"Le personnage est maintenant en {nouvelle_position}")  # [15, 17]
```

🤔 **Point de réflexion** : L'addition se fait composante par composante. C'est comme si on ajoutait séparément les x et les y !

### 2. La multiplication par un scalaire

Votre personnage boit une potion de vitesse qui double sa vitesse :

```python
# Vitesse initiale
vitesse = np.array([3, 2])  # 3 unités/s en x, 2 unités/s en y

# Double vitesse (multiplication par un scalaire)
vitesse_boost = 2 * vitesse
print(f"Nouvelle vitesse : {vitesse_boost}")  # [6, 4]

# Demi-vitesse
vitesse_ralentie = 0.5 * vitesse
print(f"Vitesse ralentie : {vitesse_ralentie}")  # [1.5, 1]
```

### 3. Le produit scalaire (dot product)

Le produit scalaire est une opération magique qui mesure à quel point deux vecteurs vont dans la même direction :

```python
# Direction du personnage
direction = np.array([1, 0])  # Regarde vers la droite

# Direction d'un ennemi
direction_ennemi = np.array([1, 0])    # Même direction
direction_ennemi2 = np.array([-1, 0])  # Direction opposée

# Produit scalaire
alignement = np.dot(direction, direction_ennemi)   # = 1 (même direction)
alignement2 = np.dot(direction, direction_ennemi2) # = -1 (directions opposées)
```

### 4. La norme (longueur) d'un vecteur

La norme mesure la "longueur" d'un vecteur. Très utile pour calculer des distances :

```python
# Position d'un ennemi
position_ennemi = np.array([15, 25])

# Calcul du vecteur de distance
vecteur_distance = position_ennemi - position

# Distance réelle (norme du vecteur)
distance = np.linalg.norm(vecteur_distance)
print(f"L'ennemi est à {distance:.2f} unités")
```

### 5. La normalisation

Transformer un vecteur pour qu'il ait une longueur de 1 tout en gardant sa direction :

```python
# Direction quelconque
direction = np.array([3, 4])

# Normalisation
direction_normalisee = direction / np.linalg.norm(direction)

# Vérifions que la longueur est bien 1
print(f"Longueur : {np.linalg.norm(direction_normalisee):.2f}")  # ≈ 1.0
```

### 6. Les opérations pratiques

Dans un jeu, on combine souvent ces opérations :

```python
def se_deplacer_vers(position_actuelle, cible, vitesse):
    """Déplace un objet vers une cible à une certaine vitesse"""
    # Calculer la direction
    direction = cible - position_actuelle

    # Normaliser et ajuster par la vitesse
    if np.linalg.norm(direction) > 0:  # Éviter division par zéro
        deplacement = vitesse * direction / np.linalg.norm(direction)
        return position_actuelle + deplacement
    return position_actuelle

# Exemple
position = np.array([0, 0])
cible = np.array([10, 10])
nouvelle_position = se_deplacer_vers(position, cible, 2)
```

### ✍️ Mini-exercice guidé

Créons une fonction qui vérifie si deux personnages peuvent se voir (pas trop loin et face à face) :

```python
def peut_voir(pos1, dir1, pos2, distance_max=10):
    """
    Vérifie si pos1 peut voir pos2

    Args:
        pos1: position du premier personnage
        dir1: direction dans laquelle il regarde (normalisée)
        pos2: position du deuxième personnage
        distance_max: distance maximale de vision
    """
    # Calcul du vecteur entre les personnages
    vers_cible = pos2 - pos1
    distance = np.linalg.norm(vers_cible)

    if distance > distance_max:
        return False

    # Direction normalisée vers la cible
    dir_cible = vers_cible / distance

    # Produit scalaire pour vérifier l'angle
    angle = np.dot(dir1, dir_cible)

    # angle > 0.7 signifie moins de ~45 degrés
    return angle > 0.7
```

### 🎯 Points clés à retenir

1. L'addition déplace des points
2. La multiplication par un scalaire change l'intensité
3. Le produit scalaire mesure l'alignement
4. La norme donne la distance
5. La normalisation donne la direction pure

### 🎮 Prochaines étapes

Dans la prochaine partie, nous verrons :

- Des applications concrètes
- Comment utiliser tout ça en ML
- Des exercices plus complexes
