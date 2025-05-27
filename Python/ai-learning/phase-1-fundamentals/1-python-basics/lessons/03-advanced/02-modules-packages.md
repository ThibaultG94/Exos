# Les modules et packages en Python 📦

## Pourquoi utiliser des modules ?

Imaginez que vous développez un gros jeu avec :

- Des personnages
- Un système de combat
- Un système d'inventaire
- Une interface utilisateur
- Une sauvegarde

Mettre tout ce code dans un seul fichier deviendrait vite ingérable !

## 1. Les modules : l'unité de base

### Création d'un module

Un module est simplement un fichier Python (.py) :

```python
# personnages.py
class Personnage:
    def __init__(self, nom):
        self.nom = nom

# Fonction du module
def creer_heros(nom):
    return Personnage(nom)

# Variable du module
VERSION = "1.0"
```

### Utilisation d'un module

```python
# jeu.py
import personnages

# Utilisation du module
hero = personnages.creer_heros("Alice")
print(personnages.VERSION)

# Import spécifique
from personnages import Personnage, VERSION
hero2 = Personnage("Bob")

# Import avec alias
import personnages as perso
hero3 = perso.creer_heros("Charlie")
```

## 2. Les packages : organiser vos modules

Un package est un dossier contenant des modules, avec un fichier spécial `__init__.py`

### Structure type d'un package

```
mon_jeu/
    __init__.py
    personnages/
        __init__.py
        guerrier.py
        mage.py
        monstre.py
    inventaire/
        __init__.py
        items.py
        gestion.py
    combat/
        __init__.py
        attaques.py
        defense.py
```

### Le fichier **init**.py

```python
# mon_jeu/__init__.py
from .personnages.guerrier import Guerrier
from .personnages.mage import Mage

__version__ = "1.0.0"
```

### Importation depuis un package

```python
# Import du package
import mon_jeu
guerrier = mon_jeu.Guerrier("Conan")

# Import spécifique
from mon_jeu.personnages import Mage
mage = Mage("Gandalf")

# Import avec chemin complet
from mon_jeu.combat.attaques import lancer_sort
```

## 3. Bonnes pratiques

### Organisation des imports

```python
# D'abord les imports de la bibliothèque standard
import os
import sys
import json

# Ensuite les imports de bibliothèques tierces
import pygame
import requests

# Enfin les imports locaux
from . import utils
from .personnages import Guerrier
```

### Éviter les imports circulaires

❌ À éviter :

```python
# fichier1.py
from fichier2 import fonction2

# fichier2.py
from fichier1 import fonction1
```

✅ Préférer :

```python
# utils.py
def fonction_commune():
    pass

# fichier1.py et fichier2.py importent utils.py
```

## 4. Modules standards utiles

```python
# Gestion des chemins
import os
chemin = os.path.join('dossier', 'fichier.txt')

# Date et heure
from datetime import datetime
maintenant = datetime.now()

# Nombres aléatoires
import random
degats = random.randint(1, 20)

# Manipulation de JSON
import json
with open('sauvegarde.json', 'r') as f:
    donnees = json.load(f)
```

## 5. Créer son propre package distribuable

### Structure minimale

```
mon_package/
    setup.py
    mon_jeu/
        __init__.py
        core.py
```

### Configuration setup.py

```python
from setuptools import setup, find_packages

setup(
    name="mon-super-jeu",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'pygame>=2.0.0',
    ],
)
```

## Points clés à retenir

1. Un module = un fichier .py
2. Un package = un dossier avec **init**.py
3. Différentes façons d'importer
4. Organisation importante du code
5. Attention aux imports circulaires

## 🌟 Conseils

- Organisez votre code de manière logique
- Utilisez des noms explicites
- Documentez vos modules
- Évitez les imports \*
- Pensez à la réutilisabilité

Prêt pour les exercices ? 💪
