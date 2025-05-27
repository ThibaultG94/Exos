# Les chaînes de caractères (strings) en Python

## Apprivoiser le texte ! 📝

### À quoi servent les chaînes de caractères ?

En programmation, on manipule souvent du texte : noms d'utilisateurs, messages, données... En Python, ce texte est géré avec les "strings" (chaînes de caractères).

### 1. Créer une chaîne de caractères

```python
# Avec des guillemets simples
prenom = 'Alice'

# Avec des guillemets doubles
message = "Bonjour tout le monde !"

# Sur plusieurs lignes avec triple guillemets
description = """Ceci est un texte
qui peut s'étendre sur
plusieurs lignes !"""
```

🤔 **Point de réflexion** : Quand utiliser des guillemets simples ou doubles ?

- Les deux fonctionnent pareil
- Pratique d'avoir le choix quand le texte contient des guillemets :

```python
texte1 = "L'apostrophe est facile ici"
texte2 = 'Il a dit "Bonjour" poliment'
```

### 2. Opérations de base

```python
# Concaténation (addition de texte)
prenom = "Alice"
nom = "Dupont"
nom_complet = prenom + " " + nom  # "Alice Dupont"

# Répétition
rire = "ha" * 3  # "hahaha"

# Longueur d'une chaîne
message = "Bonjour"
taille = len(message)  # 7 caractères
```

### 3. Accéder aux caractères

En Python, chaque caractère a une position (index) qui commence à 0 :

```python
mot = "Python"
#      012345  <- positions

# Un caractère spécifique
premiere_lettre = mot[0]    # "P"
derniere_lettre = mot[5]    # "n"

# On peut aussi compter depuis la fin avec -1
derniere_lettre = mot[-1]   # "n"
avant_derniere = mot[-2]    # "o"
```

### 4. Les tranches (slices)

On peut extraire une partie d'une chaîne :

```python
message = "Bonjour Python"
#         0123456789...

# Structure : texte[début:fin]  (fin non incluse)
debut = message[0:7]     # "Bonjour"
fin = message[8:14]      # "Python"

# Raccourcis utiles
debut = message[:7]      # Depuis le début
fin = message[8:]        # Jusqu'à la fin
copie = message[:]       # Toute la chaîne
```

### 5. Méthodes utiles

Les strings ont plein de méthodes pratiques :

```python
texte = "Bonjour le monde"

# Majuscules/minuscules
print(texte.upper())      # "BONJOUR LE MONDE"
print(texte.lower())      # "bonjour le monde"
print(texte.title())      # "Bonjour Le Monde"

# Recherche
print(texte.count("o"))   # 3 (nombre de "o")
print(texte.find("le"))   # 8 (position de "le")

# Remplacement
nouveau = texte.replace("monde", "python")  # "Bonjour le python"

# Nettoyage
texte = "  du texte  "
print(texte.strip())      # "du texte" (sans espaces)

# Découpage
phrase = "Python,Java,C++"
langages = phrase.split(",")  # ["Python", "Java", "C++"]
```

### 6. Les f-strings (formatage moderne)

La meilleure façon de mélanger texte et variables :

```python
nom = "Alice"
age = 25
message = f"{nom} a {age} ans"  # "Alice a 25 ans"

# On peut même mettre des expressions
prix = 19.99
message = f"Prix TTC : {prix * 1.2:.2f}€"
```

🎮 **Mini-exercice** : Créez une variable avec votre prénom et affichez "Bonjour, [votre prénom] !"

### ✍️ Exercice pratique guidé

```python
# Créons un petit formateur de carte de visite
prenom = "Marie"
nom = "Durand"
metier = "développeuse"
entreprise = "TechCorp"

# 1. Mettons le nom en majuscules
nom = nom.upper()

# 2. Créons la carte
carte = f"""
{prenom} {nom}
{metier.title()}
{entreprise}
"""

print(carte)
```

### 🎯 Points clés à retenir

1. Les strings sont créés avec ' ou "
2. On peut accéder aux caractères avec []
3. Les strings ont plein de méthodes utiles
4. Les f-strings sont super pratiques pour le formatage

### 🏃‍♂️ Prochaines étapes

- Essayez les exemples de code
- Expérimentez avec vos propres textes
- Prêt(e) à faire les exercices ?
