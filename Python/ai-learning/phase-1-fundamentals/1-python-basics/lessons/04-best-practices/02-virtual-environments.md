# Les environnements virtuels en Python 🌍

## Pourquoi utiliser des environnements virtuels ?

Imaginez ce scénario :

- Projet A : besoin de Django 2.2
- Projet B : besoin de Django 4.0
- Projet C : besoin de Django 3.0

Comment gérer ces différentes versions sur une même machine ? Les environnements virtuels sont la solution !

## 1. Les bases des environnements virtuels

### Création d'un environnement

```bash
# Création
python -m venv mon-env

# Structure créée :
mon-env/
    ├── bin/           # Scripts d'activation (Linux/Mac)
    ├── Scripts/       # Scripts d'activation (Windows)
    ├── Lib/          # Librairies Python
    └── pyvenv.cfg    # Configuration
```

### Activation/Désactivation

```bash
# Linux/Mac
source mon-env/bin/activate

# Windows
mon-env\Scripts\activate

# Pour désactiver (tous OS)
deactivate
```

## 2. Gestion des packages

### Installation de packages

```bash
# Dans l'environnement activé
pip install requests
pip install Django==4.0.0  # Version spécifique

# Installation depuis un fichier
pip install -r requirements.txt
```

### Création du requirements.txt

```bash
# Sauvegarder les dépendances
pip freeze > requirements.txt

# Exemple de contenu :
Django==4.0.0
requests==2.28.1
pytest==7.1.1
```

## 3. Bonnes pratiques

### Organisation des projets

```
mon-projet/
    ├── venv/          # Environnement virtuel
    ├── src/           # Code source
    ├── tests/         # Tests
    ├── requirements.txt
    └── README.md
```

### Différents environnements

```bash
# Environnement de développement
python -m venv dev-env
pip install -r requirements-dev.txt

# Environnement de production
python -m venv prod-env
pip install -r requirements-prod.txt
```

## 4. Cas d'usage avancés

### Environnements pour différentes versions Python

```bash
# Avec Python 3.7
python3.7 -m venv py37-env

# Avec Python 3.9
python3.9 -m venv py39-env
```

### Environnements conda (Alternative)

```bash
# Création
conda create --name mon-env python=3.9

# Activation
conda activate mon-env

# Installation packages
conda install pandas numpy
```

## Points clés à retenir

1. Un environnement par projet
2. Toujours activer avant utilisation
3. Maintenir requirements.txt à jour
4. Ne pas versionner l'environnement
5. Documenter la configuration

## 🎯 Conseils pratiques

1. Dans le .gitignore :

```
venv/
env/
.venv/
```

2. Dans le README :

```markdown
## Installation

python -m venv venv
source venv/bin/activate # ou venv\Scripts\activate sur Windows
pip install -r requirements.txt
```

## ⚠️ Erreurs courantes

1. Oublier d'activer l'environnement
2. Installer globalement au lieu de dans l'env
3. Versionner l'environnement
4. Oublier de mettre à jour requirements.txt

## 🔄 Workflow recommandé

1. Créer le projet
2. Créer l'environnement
3. Activer l'environnement
4. Installer les dépendances
5. Développer
6. Figer les dépendances
7. Documenter

Prêt pour les exercices ? 💪
