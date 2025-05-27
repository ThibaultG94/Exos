# Les dictionnaires et ensembles en Python 🗂️

## Les dictionnaires

### Concept de base

Un dictionnaire est une collection de paires clé-valeur :

```python
joueur = {
    "nom": "Alice",
    "niveau": 5,
    "points": 100,
    "items": ["épée", "potion"]
}
```

### Création et manipulation

```python
# Création
scores = {}  # Dictionnaire vide
scores = dict()  # Autre façon

# Ajout/modification
scores["Alice"] = 100
scores["Bob"] = 95

# Accès
niveau = joueur["niveau"]  # 5
points = joueur.get("points", 0)  # Avec valeur par défaut

# Suppression
del scores["Bob"]
dernier = scores.pop("Alice")
```

### Méthodes utiles

```python
# Récupérer les clés/valeurs
cles = scores.keys()
valeurs = scores.values()
items = scores.items()  # Tuples (clé, valeur)

# Vérification
if "Alice" in scores:
    print("Score trouvé !")

# Fusion de dictionnaires
dict1 = {"a": 1, "b": 2}
dict2 = {"c": 3}
dict1.update(dict2)
```

## Les ensembles (sets)

### Concept de base

Un ensemble est une collection non ordonnée d'éléments uniques :

```python
# Création
couleurs = {"rouge", "vert", "bleu"}
nombres = set([1, 2, 2, 3])  # {1, 2, 3}

# Ajout/suppression
couleurs.add("jaune")
couleurs.remove("vert")  # Erreur si absent
couleurs.discard("vert")  # Pas d'erreur si absent
```

### Opérations ensemblistes

```python
A = {1, 2, 3}
B = {3, 4, 5}

# Union (|)
union = A | B  # {1, 2, 3, 4, 5}

# Intersection (&)
commun = A & B  # {3}

# Différence (-)
diff = A - B  # {1, 2}

# Différence symétrique (^)
sym_diff = A ^ B  # {1, 2, 4, 5}
```

## Cas d'utilisation

### Dictionnaires

1. Données structurées :

```python
contact = {
    "nom": "Dupont",
    "email": "dupont@email.com",
    "tel": "0123456789"
}
```

2. Cache/Mémo :

```python
cache = {}
def fibonacci(n):
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    cache[n] = fibonacci(n-1) + fibonacci(n-2)
    return cache[n]
```

### Ensembles

1. Élimination des doublons :

```python
emails = ["user@mail.com", "admin@mail.com", "user@mail.com"]
emails_uniques = set(emails)
```

2. Appartenance rapide :

```python
mots_interdits = {"spam", "virus", "hack"}
message = "Voici un message"
est_safe = not any(mot in message for mot in mots_interdits)
```

## Points clés à retenir

1. Dictionnaire = collection clé-valeur

   - Accès rapide par clé
   - Clés uniques
   - Valeurs modifiables

2. Ensemble = collection d'éléments uniques
   - Pas d'ordre
   - Pas de doublons
   - Opérations ensemblistes

## Quand utiliser quoi ?

- Dictionnaire :

  - Association clé-valeur
  - Données structurées
  - Cache/mémorisation

- Ensemble :
  - Liste sans doublons
  - Tests d'appartenance rapides
  - Opérations ensemblistes
