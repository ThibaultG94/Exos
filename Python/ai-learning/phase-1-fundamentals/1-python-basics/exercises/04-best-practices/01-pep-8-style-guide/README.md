# Exercices : Les bonnes pratiques Python 📚

## Objectifs pédagogiques

À la fin de ces exercices, vous serez capable de :

- Écrire une documentation claire et utile
- Appliquer les conventions PEP 8
- Structurer un projet Python correctement
- Utiliser les type hints efficacement

## Exercice 1 : Documentation et docstrings

```python
"""
Objectif : Améliorer la documentation d'un projet existant

1. Reprenez le code suivant et améliorez sa documentation :

class Inventaire:
    def __init__(self):
        self.items = []

    def ajouter(self, item):
        self.items.append(item)

    def retirer(self, item):
        self.items.remove(item)

    def afficher(self):
        print(self.items)

2. Pour chaque méthode, ajoutez :
   - Une description claire
   - Les paramètres et leur type
   - La valeur de retour
   - Au moins un exemple d'utilisation
   - Les exceptions possibles

3. Ajoutez une documentation de module
4. Utilisez les type hints appropriés
"""
```

## Exercice 2 : Réorganisation selon PEP 8

```python
"""
Objectif : Corriger le style d'un module selon PEP 8

1. Reprenez ce code et corrigez tous les problèmes de style :

class gestionnaire_combat:
    def __init__(self,joueur1,joueur2):
        self.j1=joueur1
        self.j2=joueur2
        self.tour=1

    def ATTAQUER(self):
        degats=10
        self.j2.points_de_vie-=degats

    def finDuCombat(self):
        return self.j1.points_de_vie<=0 or self.j2.points_de_vie<=0

2. Corrigez :
   - Le nommage des classes et méthodes
   - Les espaces autour des opérateurs
   - L'indentation
   - Les conventions de nommage
   - Les lignes trop longues
"""
```

## Exercice 3 : Structure de projet

```python
"""
Objectif : Créer la structure d'un nouveau projet

1. Créez une structure de projet pour un jeu de rôle avec :
   - Un package principal 'rpg_game'
   - Des modules pour : personnages, combat, inventaire
   - Des tests unitaires
   - Une documentation
   - Un fichier de configuration

2. Dans chaque module, créez :
   - Une documentation de module
   - Au moins une classe ou fonction
   - Des tests correspondants

3. Créez les fichiers standards :
   - README.md
   - requirements.txt
   - setup.py
   - .gitignore
"""
```

## Points à évaluer

- Clarté et utilité de la documentation
- Respect des conventions PEP 8
- Organisation logique du projet
- Utilisation appropriée des type hints
- Gestion des erreurs et exceptions

## Bonus

- Ajoutez des tests automatisés pour vérifier le style
- Créez une documentation avec Sphinx
- Implémentez un système de logging
- Ajoutez des exemples interactifs dans la documentation
