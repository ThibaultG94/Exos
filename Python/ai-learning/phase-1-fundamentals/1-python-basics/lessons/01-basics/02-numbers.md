# Leçon 1 (Suite) : Les nombres en Python

## Allons un peu plus loin ! 🚀

### Rappel de la partie précédente

Nous avons vu :

- Les variables (nos "boîtes" pour stocker des valeurs)
- Les nombres entiers (1, 2, 3...)

### 1. Les différents types de nombres

En Python, il existe principalement deux types de nombres :

- Les entiers (integers ou `int`) : 1, 2, 3...
- Les nombres à virgule (floating point ou `float`) : 1.5, 3.14, -2.8...

```python
age = 25              # Un entier (int)
temperature = 37.5    # Un nombre à virgule (float)
```

🤔 **Point de réflexion** : Dans la vie quotidienne, quand utilisez-vous des nombres entiers et quand utilisez-vous des nombres à virgule ?

### 2. Les opérations de base

Vous pouvez faire des calculs avec vos variables :

```python
# Addition (+)
score = 10 + 5        # score vaut 15

# Soustraction (-)
vies = 3 - 1         # vies vaut 2

# Multiplication (*)
points = 5 * 2       # points vaut 10

# Division (/)
parts = 10 / 2       # parts vaut 5.0 (notez que c'est un float !)
```

🎮 **Mini-exercice** : Créez deux variables `prix_unitaire` et `quantite`, puis calculez le total !

### 3. Un peu plus sur la division

Python propose deux types de division :

```python
# Division classique (/) - donne toujours un float
resultat1 = 10 / 2   # resultat1 vaut 5.0
resultat2 = 5 / 2    # resultat2 vaut 2.5

# Division entière (//) - garde seulement la partie entière
resultat3 = 5 // 2   # resultat3 vaut 2
```

💡 **Exemple concret** : Imaginez que vous avez 5 bonbons à partager entre 2 personnes.

- Avec /, vous obtenez 2.5 bonbons par personne (mathématiquement correct)
- Avec //, vous obtenez 2 bonbons par personne (plus réaliste !)

### 4. Le reste de la division (%)

Le symbole % donne le reste d'une division :

```python
reste = 5 % 2    # reste vaut 1 (car 5 ÷ 2 = 2 reste 1)
```

🎯 **Application** : Pratique pour savoir si un nombre est pair !

```python
# Si le reste de la division par 2 est 0, le nombre est pair
nombre = 4
est_pair = nombre % 2 == 0    # est_pair vaut True
```

### 5. Les arrondis avec les floats

Les nombres à virgule peuvent parfois nous surprendre :

```python
calcul = 0.1 + 0.2    # On pourrait penser que ça fait 0.3...
print(calcul)         # Mais on obtient 0.30000000000000004 !
```

Pour arrondir proprement :

```python
prix = 9.99999
prix_arrondi = round(prix, 2)    # Arrondi à 2 décimales : 10.00
```

### ✍️ Exercice pratique guidé

Créons une petite caisse enregistreuse :

```python
# 1. Prix des articles
prix_pomme = 0.50
prix_banane = 0.75

# 2. Quantités
nb_pommes = 3
nb_bananes = 2

# 3. Calcul du total
total_pommes = prix_pomme * nb_pommes
total_bananes = prix_banane * nb_bananes
total = total_pommes + total_bananes

# 4. Affichage
print("Total pommes :", total_pommes)
print("Total bananes :", total_bananes)
print("Total général :", total)
```

### 🎯 Vérifions votre compréhension

Répondez mentalement :

1. Quelle est la différence entre `int` et `float` ?
2. Que donne `17 / 5` ? Et `17 // 5` ?
3. À quoi sert l'opérateur `%` ?
4. Pourquoi utiliser `round()` avec les floats ?

### 🏃‍♂️ Pratiquons !

Quelques idées d'exercices :

1. Calculez le prix TTC d'un article (prix HT × 1.20 pour 20% de TVA)
2. Partagez équitablement une addition entre amis
3. Calculez la moyenne de plusieurs notes
4. Vérifiez si un nombre est divisible par un autre

### 🎯 Projet mini-calculatrice

```python
# Prix d'un article
prix_ht = 100

# Calculs
tva = prix_ht * 0.20
prix_ttc = prix_ht + tva

# Affichage
print("Prix HT :", prix_ht)
print("TVA :", tva)
print("Prix TTC :", prix_ttc)
```

Prêt à créer vos propres calculs ? 🚀
