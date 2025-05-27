# Les listes en compréhension 🎯

## Pourquoi les listes en compréhension ?

Imaginez que vous devez :

- Créer une liste des carrés des nombres de 0 à 9
- Filtrer une liste pour ne garder que les nombres pairs
- Transformer tous les mots d'une phrase en majuscules

On pourrait le faire avec des boucles... mais Python offre une solution plus élégante !

## 1. Structure de base

### La syntaxe simple

```python
# Sans compréhension
carrés = []
for x in range(10):
    carrés.append(x ** 2)

# Avec compréhension
carrés = [x ** 2 for x in range(10)]
```

Structure : `[expression for élément in itérable]`

### Exemples simples

```python
# Liste des nombres pairs
pairs = [n for n in range(10) if n % 2 == 0]  # [0, 2, 4, 6, 8]

# Conversion en majuscules
mots = ["python", "est", "super"]
majuscules = [mot.upper() for mot in mots]  # ["PYTHON", "EST", "SUPER"]
```

## 2. Filtrage avec conditions

### Condition simple

```python
nombres = [-2, -1, 0, 1, 2]
positifs = [n for n in nombres if n > 0]  # [1, 2]
```

### Conditions multiples

```python
# Nombres pairs et positifs
nombres = range(-5, 6)
pairs_positifs = [n for n in nombres if n > 0 and n % 2 == 0]  # [2, 4]
```

## 3. Transformations complexes

### Expressions élaborées

```python
# Tuple (nombre, carré, cube)
nombres = range(3)
puissances = [(n, n**2, n**3) for n in nombres]
# [(0, 0, 0), (1, 1, 1), (2, 4, 8)]
```

### Avec méthodes de string

```python
phrase = "Python est génial"
analyse = [(mot, len(mot)) for mot in phrase.split()]
# [("Python", 6), ("est", 3), ("génial", 6)]
```

## 4. Compréhensions imbriquées

### Boucles multiples

```python
# Table de multiplication
table = [(x, y, x*y) for x in range(1, 3) for y in range(1, 3)]
# [(1, 1, 1), (1, 2, 2), (2, 1, 2), (2, 2, 4)]
```

### Matrices

```python
# Création d'une matrice 3x3
matrice = [[0 for _ in range(3)] for _ in range(3)]
# [[0, 0, 0],
#  [0, 0, 0],
#  [0, 0, 0]]
```

## 5. Cas d'utilisation pratiques

### Traitement de données

```python
données = [("Alice", 15), ("Bob", 12), ("Charlie", 18)]
# Extraire les noms des élèves ayant la moyenne
admis = [nom for nom, note in données if note >= 15]
```

### Nettoyage de texte

```python
texte = "Python  est    super !"
# Nettoyer les espaces multiples
propre = " ".join([mot for mot in texte.split() if mot])
```

## 6. Conseils et bonnes pratiques

### ✅ Quand les utiliser

- Pour des transformations simples
- Quand le code devient plus lisible
- Pour des opérations sur tous les éléments

### ❌ Quand éviter

- Opérations trop complexes
- Plusieurs conditions imbriquées
- Quand une boucle classique est plus claire

## Points clés à retenir

1. Structure : `[expression for item in iterable if condition]`
2. Plus concis que les boucles traditionnelles
3. Peut inclure des conditions de filtrage
4. Fonctionne avec tous types d'itérables
5. Attention à la lisibilité !

## 🎯 Exercice rapide

Pouvez-vous réécrire ce code avec une compréhension ?

```python
mots = ["Python", "est", "un", "langage"]
longs_mots = []
for mot in mots:
    if len(mot) > 3:
        longs_mots.append(mot.lower())
```

Solution : `longs_mots = [mot.lower() for mot in mots if len(mot) > 3]`
