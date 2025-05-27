# Exercices : Les modules et packages en Python 📦

## Structure du projet

```
02-modules-packages/
    ├── exercise.py          # Fichier principal d'exercice
    ├── solution.py          # Solutions complètes
    └── mon_jeu/            # Package du jeu
        ├── __init__.py      # Initialisation du package
        ├── personnages/     # Module des personnages
        │   ├── __init__.py
        │   └── base.py      # Classes de personnages
        ├── inventaire/      # Module d'inventaire
        │   ├── __init__.py
        │   └── gestion.py   # Classes d'items et inventaire
        └── combat/          # Module de combat
            ├── __init__.py
            └── systeme.py   # Fonctions de combat
```

## Exercice 1 : Structure de base

```python
"""
Objectif : Créer la structure de base du package mon_jeu

1. Créez les dossiers et fichiers selon la structure ci-dessus
2. Dans base.py, créez la classe Personnage
3. Dans gestion.py, créez les classes Item et Inventaire
4. Dans systeme.py, créez une fonction calculer_degats
5. Dans chaque __init__.py, importez les éléments pertinents
"""
```

## Exercice 2 : Utilisation du package

```python
"""
Objectif : Utiliser le package créé

1. Dans exercise.py, importez les classes et fonctions :
   - from mon_jeu.personnages.base import Personnage
   - from mon_jeu.inventaire.gestion import Item, Inventaire
   - from mon_jeu.combat.systeme import calculer_degats

2. Créez une fonction test_jeu() qui :
   - Crée un personnage avec inventaire
   - Ajoute quelques items
   - Simule un combat
"""
```

## Exercice 3 : Module utilitaire

```python
"""
Objectif : Ajouter un module utilitaire

1. Ajoutez un module utils/ avec :
   - config.py : fonctions de configuration
   - saves.py : gestion des sauvegardes
   - logger.py : système de logs

2. Dans chaque fichier, utilisez les modules standards Python :
   - json pour config et saves
   - datetime pour le logger
   - os.path pour la gestion des chemins
"""
```

## Conseils

- Testez les imports à chaque étape
- Utilisez des imports relatifs dans les modules
- Documentez chaque module avec des docstrings
- Gérez les erreurs d'import et de fichiers
