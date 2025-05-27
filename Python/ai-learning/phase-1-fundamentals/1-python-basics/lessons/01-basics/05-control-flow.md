# Leçon 5 : Le contrôle de flux 🔀

## Donnons de l'intelligence à nos programmes !

### À quoi sert le contrôle de flux ?

Les structures de contrôle permettent à nos programmes de :

- Prendre des décisions (avec `if/elif/else`)
- Répéter des actions (avec les boucles `while` et `for`)
- Réagir différemment selon les situations

### 1. Les conditions avec if/elif/else

La base du contrôle de flux : exécuter du code selon une condition.

```python
age = 17

if age >= 18:
    print("Vous êtes majeur !")
else:
    print("Vous êtes mineur !")
```

🤔 **Point de réflexion** : Les instructions dans un `if` ne s'exécutent que si la condition est `True`. Rappelez-vous les booléens !

#### La structure elif

Pour gérer plusieurs cas :

```python
note = 15

if note >= 16:
    print("Très bien !")
elif note >= 14:
    print("Bien !")
elif note >= 12:
    print("Assez bien !")
else:
    print("Pas de mention")
```

#### Les conditions imbriquées

On peut mettre un `if` dans un autre `if` :

```python
age = 20
permis = True

if age >= 18:
    if permis:
        print("Vous pouvez conduire !")
    else:
        print("Il vous faut le permis !")
else:
    print("Trop jeune pour conduire")
```

### 2. La boucle while

Pour répéter tant qu'une condition est vraie :

```python
# Compte à rebours
compte = 5

while compte > 0:
    print(compte)
    compte = compte - 1  # ou compte -= 1
print("Décollage ! 🚀")
```

⚠️ **Attention** : Assurez-vous que la condition devient fausse à un moment, sinon boucle infinie !

### 3. La boucle for

Pour parcourir une séquence (nous verrons les listes plus tard) :

```python
# Parcourir une chaîne
mot = "Python"
for lettre in mot:
    print(lettre)

# Parcourir une plage de nombres
for i in range(5):  # 0, 1, 2, 3, 4
    print(f"Tour {i}")
```

### 4. break et continue

Pour contrôler le flux dans les boucles :

```python
# break : sort de la boucle
for i in range(10):
    if i == 5:
        break
    print(i)  # Affiche 0, 1, 2, 3, 4

# continue : passe à l'itération suivante
for i in range(5):
    if i == 2:
        continue
    print(i)  # Affiche 0, 1, 3, 4
```

### ✍️ Exercice pratique guidé

Créons un petit jeu de devinette :

```python
# Le nombre à deviner
secret = 42
tentatives = 0

while True:
    guess = int(input("Devinez le nombre : "))
    tentatives += 1

    if guess == secret:
        print(f"Bravo ! Trouvé en {tentatives} essais !")
        break
    elif guess < secret:
        print("C'est plus !")
    else:
        print("C'est moins !")
```

### 🎯 Points clés à retenir

1. `if/elif/else` pour les décisions
2. `while` pour répéter tant qu'une condition est vraie
3. `for` pour parcourir une séquence
4. `break` pour sortir d'une boucle
5. `continue` pour passer à l'itération suivante
6. L'indentation est cruciale en Python !

### 🚀 Cas d'utilisation courants

1. Validation de données :

```python
age = int(input("Âge : "))
if age < 0:
    print("Âge invalide !")
elif age < 18:
    print("Mineur")
else:
    print("Majeur")
```

2. Traitement jusqu'à condition :

```python
mot_de_passe = ""
while len(mot_de_passe) < 8:
    mot_de_passe = input("Choisissez un mot de passe (min 8 car.) : ")
```

3. Parcours de données :

```python
total = 0
for i in range(1, 11):
    total += i
print(f"Somme de 1 à 10 : {total}")
```

### 🎯 Vérifions votre compréhension

Répondez mentalement :

1. Quelle est la différence entre `if` et `elif` ?
2. Que se passe-t-il si aucune condition n'est vraie et qu'il n'y a pas de `else` ?
3. Quand utiliser `while` plutôt que `for` ?
4. Que fait `break` dans une boucle `for` ?

Prêt(e) pour les exercices ? 💪
