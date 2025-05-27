# Les listes et tuples en Python 📦

## Pourquoi utiliser des listes et des tuples ?

Imaginez que vous développez un jeu et que vous devez stocker :

- Les scores des joueurs
- L'inventaire d'un personnage
- Une séquence de mouvements

Une variable simple ne suffit plus... Il nous faut des collections !

## 1. Les listes : collections modifiables

### Création d'une liste

```python
# Liste vide
scores = []

# Liste avec des valeurs
scores = [100, 95, 88, 72]
inventaire = ["épée", "potion", "bouclier"]
mixte = [1, "texte", True, 3.14]  # Types différents possibles
```

### Accéder aux éléments

Les indices commencent à 0 :

```python
scores = [100, 95, 88, 72]
#         0    1   2   3

premier = scores[0]    # 100
dernier = scores[-1]   # 72
```

### Modifier les éléments

```python
inventaire = ["épée", "potion", "bouclier"]
inventaire[1] = "potion de vie"  # Remplace "potion"
```

### Méthodes principales

```python
scores = [100, 95]

# Ajouter des éléments
scores.append(88)      # [100, 95, 88]
scores.insert(1, 98)   # [100, 98, 95, 88]

# Retirer des éléments
dernier = scores.pop()     # Retire et renvoie 88
scores.remove(95)          # Retire le premier 95
del scores[0]             # Supprime l'élément d'index 0

# Autres opérations utiles
longueur = len(scores)         # Nombre d'éléments
maximum = max(scores)          # Plus grande valeur
minimum = min(scores)          # Plus petite valeur
scores.sort()                  # Trie la liste
scores.reverse()              # Inverse l'ordre
```

### Tranches (slices)

Comme pour les strings :

```python
nombres = [0, 1, 2, 3, 4, 5]
debut = nombres[0:3]     # [0, 1, 2]
fin = nombres[3:]        # [3, 4, 5]
pas = nombres[::2]       # [0, 2, 4] (un élément sur deux)
```

## 2. Les tuples : collections immuables

### Création d'un tuple

```python
# Tuple vide
point = ()

# Tuple avec valeurs
coordonnees = (10, 20)
joueur = ("Alice", 100, True)

# Tuple d'un seul élément (notez la virgule !)
singleton = (42,)
```

### Différences avec les listes

```python
# Les tuples sont immuables
coordonnees = (10, 20)
coordonnees[0] = 15  # ❌ Erreur !

# Mais on peut créer un nouveau tuple
coordonnees = (15, 20)  # ✅ OK
```

### Pourquoi utiliser des tuples ?

1. Protection des données (immuables)
2. Plus légers en mémoire
3. Peuvent être utilisés comme clés de dictionnaire
4. Parfaits pour les valeurs qui vont ensemble :
   - Coordonnées (x, y)
   - Couleur RGB (r, g, b)

### Conversion liste ↔ tuple

```python
# Liste vers tuple
liste = [1, 2, 3]
tuple_conv = tuple(liste)    # (1, 2, 3)

# Tuple vers liste
tuple_orig = (1, 2, 3)
liste_conv = list(tuple_orig)  # [1, 2, 3]
```

## Cas d'utilisation courants

### 1. Parcours avec for

```python
inventaire = ["épée", "potion", "bouclier"]

# Simple parcours
for item in inventaire:
    print(item)

# Avec index
for i, item in enumerate(inventaire):
    print(f"{i}: {item}")
```

### 2. Compréhension de liste

Une façon élégante de créer des listes :

```python
# Sans compréhension
carres = []
for n in range(5):
    carres.append(n ** 2)

# Avec compréhension
carres = [n ** 2 for n in range(5)]  # [0, 1, 4, 9, 16]

# Avec condition
pairs = [n for n in range(10) if n % 2 == 0]  # [0, 2, 4, 6, 8]
```

### 3. Retour multiple de fonction

```python
def stats_joueur():
    return "Alice", 100, True  # Retourne un tuple

# Déballage de tuple
nom, score, actif = stats_joueur()
```

## Points clés à retenir

1. Liste = collection modifiable `[]`
2. Tuple = collection immuable `()`
3. Les indices commencent à 0
4. Tranches similaires aux strings
5. For pour parcourir
6. Compréhension de liste pour créer/filtrer

## 🎯 Choisir entre liste et tuple

- Liste : quand les données peuvent changer

  - Scores dans un jeu
  - Inventaire d'items
  - Liste de tâches

- Tuple : quand les données sont fixes
  - Coordonnées (x, y)
  - Configuration
  - Retour multiple de fonction
