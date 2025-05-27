# Leçon 1 : Les variables en Python

## Une introduction tout en douceur 🐣

### À quoi sert cette leçon ?

À la fin de cette leçon, vous saurez :

- Ce qu'est une variable et à quoi ça sert
- Comment créer et utiliser des variables simples
- Comprendre les nombres entiers en Python

### 1. Qu'est-ce qu'une variable ?

Imaginez une boîte... 📦

Une variable, c'est comme une boîte étiquetée dans laquelle vous pouvez ranger une valeur.

- L'étiquette, c'est le nom de votre variable
- Ce qui est dans la boîte, c'est la valeur

```python
age = 25
```

Ici, on crée une boîte étiquetée "age" et on met le nombre 25 dedans.

### 2. Pourquoi utiliser des variables ?

Prenons un exemple concret. Imaginons que vous développez un petit jeu :

```python
# Sans variable (difficile à maintenir)
print("Le joueur a 100 points")
print("Il reste 100 points à gagner")
print("Le niveau maximum est de 100 points")

# Avec une variable (beaucoup mieux !)
score_maximum = 100
print("Le joueur a", score_maximum, "points")
print("Il reste", score_maximum, "points à gagner")
print("Le niveau maximum est de", score_maximum, "points")
```

🤔 **Point de réflexion** : Que se passe-t-il si on veut changer le score maximum à 200 ?
Quelle version sera plus facile à modifier ?

### 3. Comment créer une variable ?

En Python, c'est très simple. On utilise le signe = pour "ranger" une valeur dans une variable :

```python
# On crée une variable "nombre_de_vies" qui contient 3
nombre_de_vies = 3

# On crée une variable "score" qui contient 0
score = 0
```

🎮 **Mini-exercice** : Créez une variable `niveau` et mettez-y le nombre 1

### 4. Les règles pour nommer les variables

Comme pour les étiquettes sur vos boîtes, il y a quelques règles :

✅ Ce qui est permis :

```python
age = 25          # Simple et clair
mon_age = 25      # Avec un underscore
numberOfLives = 3  # Style "camelCase"
SCORE_MAXIMUM = 100  # Constantes en majuscules
```

❌ Ce qui n'est PAS permis :

```python
2age = 25        # Ne peut pas commencer par un chiffre
mon-age = 25     # Le tiret n'est pas permis
mon âge = 25     # Les espaces ne sont pas permis
```

### 5. Utiliser les variables

Une fois créée, vous pouvez utiliser votre variable :

```python
# On crée la variable
niveau = 1

# On l'affiche
print("Vous êtes au niveau", niveau)

# On peut la modifier
niveau = 2
print("Bravo ! Vous passez au niveau", niveau)
```

### 6. Premier pas avec les nombres entiers

En Python, les nombres entiers s'appellent des "integers" (ou "int").
Ce sont les nombres comme : -2, -1, 0, 1, 2, 3...

```python
niveau = 1        # Un nombre positif
vies = 0         # Zéro
temperature = -5  # Un nombre négatif
```

### ✍️ Exercice pratique guidé

Créons un petit programme étape par étape :

```python
# 1. Créez une variable score qui vaut 0
score = 0
print("Score de départ :", score)

# 2. Le joueur gagne 10 points
score = 10
print("Nouveau score :", score)

# 3. Créez une variable vies qui vaut 3
vies = 3
print("Il vous reste", vies, "vies")
```

### 🎯 Vérifions votre compréhension

Prenez un moment pour répondre à ces questions (mentalement) :

1. Une variable, c'est comme une **\_\_\_** avec une étiquette
2. Pour créer une variable en Python, on utilise le signe **\_\_\_**
3. Est-ce qu'une variable peut commencer par un chiffre ? **\_\_\_**
4. Une fois créée, est-ce qu'on peut changer la valeur d'une variable ? **\_\_\_**

### 🏃‍♂️ Prochaines étapes

- Essayez de modifier le code des exemples
- Créez vos propres variables
- Jouez avec l'affichage en utilisant print()
- Prêt(e) à continuer ? Dans la prochaine partie, nous verrons d'autres types de nombres !

❓ Des questions ? N'hésitez pas à les poser avant de continuer !
