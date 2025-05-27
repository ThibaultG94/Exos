# La manipulation des fichiers en Python 📁

## Ouvrir et fermer des fichiers

```python
# Ouverture simple
fichier = open('donnees.txt', 'r')  # 'r' pour lecture
contenu = fichier.read()
fichier.close()

# Meilleure pratique avec with
with open('donnees.txt', 'r') as fichier:
    contenu = fichier.read()  # Fermeture automatique
```

## Modes d'ouverture

- `'r'` : lecture seule
- `'w'` : écriture (efface le contenu existant)
- `'a'` : ajout à la fin du fichier
- `'r+'` : lecture et écriture

## Lire un fichier

```python
# Lire tout le fichier
with open('texte.txt', 'r') as f:
    contenu = f.read()

# Lire ligne par ligne
with open('texte.txt', 'r') as f:
    for ligne in f:
        print(ligne.strip())

# Lire dans une liste
with open('texte.txt', 'r') as f:
    lignes = f.readlines()
```

## Écrire dans un fichier

```python
# Écriture simple
with open('sortie.txt', 'w') as f:
    f.write('Bonjour\n')
    f.write('Python')

# Écrire plusieurs lignes
lignes = ['Ligne 1', 'Ligne 2', 'Ligne 3']
with open('sortie.txt', 'w') as f:
    f.writelines(ligne + '\n' for ligne in lignes)
```

## Gérer l'encodage

```python
# Spécifier l'encodage
with open('texte.txt', 'r', encoding='utf-8') as f:
    contenu = f.read()

# Gérer les erreurs d'encodage
with open('texte.txt', 'r', encoding='utf-8', errors='ignore') as f:
    contenu = f.read()
```

## Travailler avec des chemins

```python
from pathlib import Path

# Créer un chemin
chemin = Path('dossier/sous-dossier/fichier.txt')

# Vérifier l'existence
if chemin.exists():
    print("Le fichier existe")

# Créer des dossiers
chemin.parent.mkdir(parents=True, exist_ok=True)

# Lire/écrire avec Path
chemin.write_text('Contenu')
contenu = chemin.read_text()
```
