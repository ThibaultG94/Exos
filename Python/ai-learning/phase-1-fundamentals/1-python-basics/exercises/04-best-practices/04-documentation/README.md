# Exercices : Documentation et bonnes pratiques 📚

## Exercice 1 : Documentation d'une classe

```python
"""
Objectif : Documenter une classe de jeu de rôle

1. Reprenez le code suivant et ajoutez une documentation complète :

class Personnage:
    def __init__(self, nom, classe, niveau=1):
        self.nom = nom
        self.classe = classe
        self.niveau = niveau
        self.experience = 0
        self.points_de_vie = niveau * 10
        self.force = niveau * 2

    def gagner_experience(self, montant):
        self.experience += montant
        while self.experience >= self.niveau * 100:
            self.monter_niveau()

    def monter_niveau(self):
        self.niveau += 1
        self.points_de_vie = self.niveau * 10
        self.force = self.niveau * 2
        self.experience -= (self.niveau - 1) * 100

2. Pour chaque élément, ajoutez :
   - Docstring de classe avec description, attributs et exemple
   - Docstring pour chaque méthode avec Args, Returns et Raises
   - Type hints pour tous les paramètres et retours
   - Commentaires pertinents expliquant la logique métier
"""
```

## Exercice 2 : Structuration d'un module

```python
"""
Objectif : Créer un module bien structuré

1. Créez un module 'combat.py' avec :
   - En-tête de module (description, auteur, date)
   - Imports organisés (standard, third-party, local)
   - Constants et configuration clairement documentées
   - Classes et fonctions avec docstrings complètes
   - Types hints et gestion d'erreurs

2. Le module doit contenir :
   - Une classe GestionnaireCombat
   - Une fonction calculer_degats
   - Des exceptions personnalisées
   - Des constantes de configuration

3. Documentez chaque élément selon les standards PEP 257
"""
```

## Exercice 3 : Documentation de projet

```python
"""
Objectif : Mettre en place la documentation complète d'un projet

1. Créez la structure suivante :
   rpg_game/
   ├── docs/
   │   ├── api.md
   │   ├── guide.md
   │   └── examples.md
   ├── rpg_game/
   │   ├── __init__.py
   │   ├── personnage.py
   │   └── combat.py
   ├── tests/
   │   └── test_combat.py
   ├── README.md
   ├── requirements.txt
   └── setup.py

2. Écrivez une documentation complète :
   - README avec badges, installation, utilisation rapide
   - Guide d'utilisation détaillé
   - Documentation de l'API
   - Exemples de code commentés
   - Docstrings pour tous les modules

3. Configuration de la documentation :
   - Ajoutez un fichier .pylintrc
   - Configurez sphinx
   - Ajoutez des tests de documentation
"""
```

## Points à évaluer

- Respect des conventions PEP 257
- Clarté et exhaustivité des docstrings
- Organisation logique des fichiers
- Qualité des exemples
- Utilisation correcte des type hints

## Bonus

- Générez la documentation HTML avec sphinx
- Ajoutez des diagrammes avec mermaid
- Créez une documentation multilingue
- Intégrez des tests de documentation automatisés
