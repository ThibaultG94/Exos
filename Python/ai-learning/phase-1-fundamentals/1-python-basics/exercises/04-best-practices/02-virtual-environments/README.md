# Exercices : Les environnements virtuels 🌍

## Niveau 1 : Création et gestion

### Exercice 1 : Configuration d'un projet Python

```python
"""
Objectif : Mettre en place un nouveau projet avec son environnement

1. Créez une structure de projet :
   mon_projet/
   ├── src/
   │   └── main.py
   ├── tests/
   │   └── test_main.py
   └── requirements.txt

2. Configurez l'environnement virtuel :
   - Créez un environnement nommé 'venv'
   - Activez-le
   - Installez : pytest, requests
   - Créez le requirements.txt
   - Ajoutez le .gitignore approprié

3. Testez votre configuration :
   - Désactivez l'environnement
   - Supprimez-le
   - Recréez-le avec requirements.txt
"""
```

### Exercice 2 : Gestion de dépendances avancée

```python
"""
Objectif : Gérer différents environnements pour un même projet

1. Créez trois fichiers de requirements :
   - requirements-dev.txt  (outils de développement)
   - requirements-test.txt (outils de test)
   - requirements-prod.txt (dépendances minimales)

2. Dans requirements-dev.txt ajoutez :
   - black
   - pylint
   - ipython
   - debug-tools

3. Dans requirements-test.txt ajoutez :
   - pytest
   - pytest-cov
   - pytest-mock

4. Dans requirements-prod.txt mettez :
   - Les dépendances minimales pour l'exécution

5. Créez un script bash/batch pour :
   - Créer l'environnement
   - Installer les dépendances selon le mode
   - Activer l'environnement
"""
```

### Exercice 3 : Compatibilité Python

```python
"""
Objectif : Gérer la compatibilité entre versions Python

1. Créez deux environnements :
   - Un avec Python 3.7
   - Un avec Python 3.9

2. Dans chaque environnement :
   - Installez numpy et pandas
   - Créez un script utilisant des f-strings
   - Utilisez le match case (Python 3.10+)
   - Testez les différences

3. Documentez :
   - Les incompatibilités trouvées
   - Les solutions possibles
   - Les bonnes pratiques de compatibilité
"""
```

## Points à évaluer

- Isolation correcte des environnements
- Structure des requirements
- Gestion des dépendances
- Documentation du projet
- Reproductibilité de l'installation

## Bonus

- Créez un script pour sauvegarder l'état de l'environnement
- Implémentez la détection automatique de la version Python
- Ajoutez des tests de compatibilité des dépendances
- Créez un Makefile pour automatiser la gestion
