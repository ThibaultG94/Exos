# Documentation et Bonnes Pratiques en Python 📚

## Les docstrings : l'art de documenter son code

Une bonne documentation commence par de bons docstrings. Imaginons documenter notre classe Personnage :

```python
class Personnage:
    """
    Représente un personnage dans notre jeu.

    Cette classe gère les attributs et comportements de base
    de tous les personnages jouables ou non-jouables.

    Attributes:
        nom (str): Le nom du personnage
        niveau (int): Niveau actuel (1-100)
        points_de_vie (int): Points de vie actuels

    Example:
        >>> hero = Personnage("Alice", niveau=1)
        >>> print(hero.nom)
        'Alice'
    """

    def attaquer(self, cible):
        """
        Effectue une attaque sur une cible.

        Args:
            cible (Personnage): Le personnage à attaquer

        Returns:
            int: Les dégâts infligés

        Raises:
            PersonnageError: Si la cible est déjà vaincue
        """
```

## Les commentaires efficaces

Les commentaires doivent expliquer le "pourquoi", pas le "quoi" :

```python
# ❌ Mauvais : décrit ce que fait le code
x = x + 1  # On incrémente x

# ✅ Bon : explique pourquoi on le fait
x = x + 1  # Compensation du décalage de l'index en base 0
```

## La PEP 8 : le style qui compte

### Nommage des variables et fonctions

```python
# ✅ Variables et fonctions en snake_case
nombre_de_vies = 3
def calculer_degats():
    pass

# ✅ Classes en PascalCase
class JoueurPrincipal:
    pass

# ✅ Constantes en MAJUSCULES
VITESSE_MAX = 100
```

### Espacement et indentation

```python
# ✅ Bon espacement
def fonction_exemple(param1, param2):
    # 4 espaces d'indentation
    resultat = param1 + param2

    # Ligne vide avant le return
    return resultat

# ✅ Bonnes pratiques pour les opérateurs
x = 5
y = x * 2 + 1  # Espaces autour des opérateurs
```

## Type Hints : la clarté avant tout

Les type hints rendent le code plus explicite :

```python
from typing import List, Optional

def ajouter_points(
    joueur: str,
    points: int,
    bonus: Optional[float] = None
) -> int:
    """
    Ajoute des points au score du joueur.

    Args:
        joueur: Nom du joueur
        points: Nombre de points à ajouter
        bonus: Multiplicateur de points optionnel

    Returns:
        Le nouveau score total
    """
```

## Organisation du projet

### Structure recommandée

```
mon_projet/
├── README.md           # Documentation principale
├── requirements.txt    # Dépendances
├── setup.py           # Configuration du package
├── docs/              # Documentation détaillée
├── tests/             # Tests unitaires
│   ├── __init__.py
│   └── test_jeu.py
└── mon_jeu/           # Code source
    ├── __init__.py
    ├── core.py
    └── utils.py
```

### Le README essentiel

````markdown
# Mon Super Jeu

## Installation

```bash
pip install -r requirements.txt
```
````

## Utilisation

```python
from mon_jeu import Personnage
hero = Personnage("Alice")
```

## Contribuer

Voir CONTRIBUTING.md

````

## Outils pratiques

1. **pylint** pour vérifier le style :
```bash
pylint mon_jeu/*.py
````

2. **black** pour formater automatiquement :

```bash
black mon_jeu/
```

3. **sphinx** pour générer la documentation :

```bash
sphinx-quickstart docs
```

## Conseils pour un code propre

1. **Principe DRY** (Don't Repeat Yourself)

```python
# ❌ Répétition
def attaquer_epee():
    degats = force * 2
    points_de_vie -= degats

def attaquer_hache():
    degats = force * 2
    points_de_vie -= degats

# ✅ Factorisation
def attaquer(arme):
    degats = calculer_degats(arme)
    appliquer_degats(degats)
```

2. **KISS** (Keep It Simple, Stupid)

```python
# ❌ Trop complexe
def valider_niveau(n):
    return True if n >= 1 and n <= 100 else False

# ✅ Simple et clair
def valider_niveau(n):
    return 1 <= n <= 100
```

## Exercices pratiques recommandés

1. Documenter complètement une classe
2. Mettre en place un projet avec la structure recommandée
3. Configurer les outils d'analyse de code
4. Écrire un README complet
