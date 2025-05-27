# Leçon 4 : Les booléens et opérateurs logiques 🔍

## Les bases de la logique en Python !

### À quoi servent les booléens ?

Les booléens sont les briques de base de la logique en programmation. Ce sont des valeurs qui ne peuvent être que vraies (`True`) ou fausses (`False`). Ils sont essentiels pour :

- Prendre des décisions dans notre code
- Vérifier des conditions
- Contrôler le flux d'exécution

### 1. Les valeurs booléennes

En Python, il n'y a que deux valeurs booléennes :

```python
a = True
b = False
```

🤔 **Point de réflexion** : Dans la vie quotidienne, quand utilisez-vous ce type de logique "vrai/faux" ?
(Par exemple : Est-ce qu'il pleut ? Est-ce que j'ai mes clés ?)

### 2. Les opérateurs de comparaison

Pour créer des expressions booléennes, on utilise les opérateurs de comparaison :

```python
age = 25
majeur = age >= 18      # True
temperature = 15
fait_froid = temperature < 10    # False

# Égalité (==) et inégalité (!=)
x = 5
y = 10
sont_egaux = x == y     # False
sont_differents = x != y # True

# Rappel : un seul = est pour l'affectation !
a = 5        # Affectation
a == 5       # Comparaison
```

### 3. Les opérateurs logiques

Pour combiner des conditions, on utilise trois opérateurs logiques principaux :

```python
# and : les deux conditions doivent être vraies
age = 25
solde = 1000
peut_louer = age >= 18 and solde >= 500    # True

# or : au moins une condition doit être vraie
est_etudiant = True
est_senior = False
reduction = est_etudiant or est_senior      # True

# not : inverse la valeur
connecte = True
deconnecte = not connecte                   # False
```

🎮 **Mini-exercice** : Créez une variable `peut_conduire` qui vérifie qu'une personne a plus de 18 ans ET possède un permis !

### 4. Les priorités des opérateurs

Comme en mathématiques, il y a un ordre de priorité :

1. Les comparaisons (`<`, `>`, `==`, etc.)
2. `not`
3. `and`
4. `or`

```python
age = 25
permis = True
solde = 1000
peut_louer_voiture = age >= 18 and permis and solde >= 500    # True
```

### 5. Les conversions implicites en booléen

En Python, certaines valeurs sont considérées comme `False` :

- `0` (entier nul)
- `0.0` (float nul)
- `""` (chaîne vide)
- `[]` (liste vide)
- `None` (valeur nulle)

Tout le reste est considéré comme `True` !

```python
# Exemples pratiques
message = ""
if message:  # équivalent à if message != ""
    print("Message non vide")

score = 0
if score:  # équivalent à if score != 0
    print("Score non nul")
```

### ✍️ Exercice pratique guidé

Créons un petit système de vérification pour un jeu vidéo :

```python
# Conditions du joueur
niveau = 5
points = 1000
vies = 3
bonus_actif = True

# Vérifications
peut_acceder_niveau_boss = niveau >= 5 and points >= 500
partie_terminee = vies <= 0 or points >= 2000
super_joueur = niveau >= 10 or (points >= 1000 and bonus_actif)

# Affichage
print(f"Peut accéder au boss : {peut_acceder_niveau_boss}")
print(f"Partie terminée : {partie_terminee}")
print(f"Super joueur : {super_joueur}")
```

### 🎯 Vérifions votre compréhension

Répondez mentalement :

1. Quelle est la différence entre `=` et `==` ?
2. Que donne `True and False` ? Et `True or False` ?
3. Que donne `not True` ?
4. Si `age = 15`, que donne `age >= 18 and age <= 30` ?

### 🏃‍♂️ Pour aller plus loin

Voici quelques cas d'utilisation courants :

1. Vérification de plage de valeurs :

```python
age = 25
est_adulte_actif = age >= 18 and age <= 65
```

2. Validation de formulaire :

```python
email = "user@example.com"
password = "12345"
formulaire_valide = len(email) > 0 and len(password) >= 5
```

3. Système de permission :

```python
est_admin = True
est_connecte = True
peut_modifier = est_connecte and est_admin
```

### 🎯 Points clés à retenir

1. Un booléen ne peut être que `True` ou `False`
2. Les opérateurs de comparaison (`>`, `<`, `==`, etc.) renvoient des booléens
3. `and`, `or` et `not` permettent de combiner des conditions
4. Attention à la différence entre `=` (affectation) et `==` (comparaison)
5. Certaines valeurs sont naturellement considérées comme `False`

### 🚀 À vous de jouer !

N'oubliez pas :

- La pratique est la clé
- Commencez simple
- Testez différentes combinaisons
- N'hésitez pas à utiliser `print()` pour vérifier vos résultats

Prêt(e) pour les exercices ? 💪
