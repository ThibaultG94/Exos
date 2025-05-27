# Exercices : Les listes en compréhension

## Exercice 1 : Transformations de base

```python
"""
Objectif : Pratiquer les transformations simples avec les list comprehensions

1. À partir de la liste nombres = range(10) :
   - Créez une liste des carrés
   - Créez une liste des nombres pairs
   - Créez une liste des nombres divisibles par 3

2. À partir de la liste mots = ["Python", "est", "UN", "langage", "SUPER"] :
   - Mettez tous les mots en minuscules
   - Créez une liste des longueurs de chaque mot
   - Gardez uniquement les mots de plus de 3 lettres
"""
```

## Exercice 2 : Filtrage et transformation

```python
"""
Objectif : Combiner filtrage et transformation

1. Créez une liste notes avec ces tuples :
   notes = [("math", 15), ("physique", 12), ("histoire", 18), ("sport", 8)]

2. Avec des list comprehensions :
   - Extraire les matières avec une note ≥ 15
   - Créer des tuples (matière, mention) où :
     * note >= 16 : "Très bien"
     * note >= 14 : "Bien"
     * note >= 12 : "Assez bien"
     * sinon : "Passable"
   - Créer une liste des matières en majuscules où la note est paire
"""
```

## Exercice 3 : Compréhensions imbriquées

```python
"""
Objectif : Utiliser des list comprehensions imbriquées

1. Générateur de coordonnées :
   - Créez une liste de tuples (x, y) pour x et y allant de 0 à 2
   - Gardez uniquement les points où x > y

2. Matrice :
   - Créez une matrice 3x3 où chaque élément est x*y
   - Aplatissez la matrice en une seule liste des valeurs > 0

3. Bonus - Traitement de phrases :
   phrases = [
       "Python est génial",
       "Les list comprehensions sont utiles",
       "Coder est amusant"
   ]
   - Créez une liste de listes contenant les mots de chaque phrase
   - Extrayez tous les mots de plus de 4 lettres de toutes les phrases
"""
```

## Critères d'évaluation

- Utilisation correcte de la syntaxe des list comprehensions
- Code concis mais lisible
- Gestion correcte des conditions
- Résultats exacts
- Bonus pour l'élégance des solutions

## Conseils

- Commencez par écrire la version avec une boucle classique
- Identifiez clairement l'expression, l'itérable et les conditions
- Vérifiez vos résultats avec des cas simples
- N'hésitez pas à diviser une compréhension complexe en plusieurs étapes
