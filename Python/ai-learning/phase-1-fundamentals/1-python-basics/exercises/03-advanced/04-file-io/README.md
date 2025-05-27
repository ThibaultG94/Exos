# Exercices : Manipulation des fichiers en Python 📁

## Exercice 1 : Carnet de notes

```python
"""
Objectif : Créer un petit carnet de notes

1. Écrire une fonction ajouter_note(texte) qui :
   - Ouvre un fichier 'notes.txt' en mode append
   - Ajoute la date et l'heure avant le texte
   - Écrit la note sur une nouvelle ligne

2. Écrire une fonction lire_notes() qui :
   - Lit et affiche toutes les notes
   - Numérote chaque note
   - Gère le cas où le fichier n'existe pas

3. Test avec plusieurs notes
"""
```

## Exercice 2 : Compteur de mots

```python
"""
Objectif : Analyser un fichier texte

1. Créer un fichier texte avec plusieurs paragraphes
2. Écrire une fonction analyser_texte(nom_fichier) qui donne :
   - Le nombre total de mots
   - Le nombre de lignes
   - Les 5 mots les plus fréquents
   - La longueur moyenne des mots

3. Gérer proprement :
   - Les erreurs de fichier
   - La ponctuation
   - La casse (majuscules/minuscules)
"""
```

## Exercice 3 : Mini base de données

```python
"""
Objectif : Gérer un système de scores

1. Créer des fonctions pour :
   - ajouter_score(joueur, score)
   - voir_top_scores(n=5)
   - effacer_scores()

2. Format du fichier 'scores.txt' :
   Jean,100
   Marie,85
   Pierre,95

3. Fonctionnalités :
   - Tri automatique des scores
   - Pas de doublons de joueurs
   - Validation des données
"""
```

## Points à évaluer

- Gestion correcte ouverture/fermeture des fichiers
- Utilisation de 'with'
- Gestion des erreurs
- Validation des entrées
- Format des données cohérent

## Bonus

- Ajouter une recherche dans les notes
- Compter les mots par paragraphe
- Ajouter une date aux scores
