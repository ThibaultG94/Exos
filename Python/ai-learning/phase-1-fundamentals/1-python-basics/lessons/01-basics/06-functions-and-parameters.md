# Leçon 6 : Les fonctions en Python 🔧

## Réutilisons notre code !

### 1. Qu'est-ce qu'une fonction ?

Une fonction est un bloc de code réutilisable qui :

- A un nom
- Peut prendre des paramètres
- Peut retourner un résultat

```python
def dire_bonjour(nom):
    return f"Bonjour {nom} !"

# Utilisation
message = dire_bonjour("Alice")  # "Bonjour Alice !"
```

### 2. Définir une fonction

```python
def calculer_moyenne(note1, note2):
    total = note1 + note2
    moyenne = total / 2
    return moyenne

# Sans paramètres
def afficher_menu():
    print("1. Nouveau jeu")
    print("2. Charger")
    print("3. Quitter")

# Sans retour
def saluer(nom):
    print(f"Salut {nom} !")  # utilise print au lieu de return
```

### 3. Les paramètres

#### Paramètres positionnels

```python
def calculer_prix(prix_ht, tva=20):
    return prix_ht * (1 + tva/100)

# Utilisation
prix1 = calculer_prix(100)      # TVA par défaut (20%)
prix2 = calculer_prix(100, 10)  # TVA spécifiée (10%)
```

#### Paramètres nommés

```python
def creer_profil(nom, age, ville="Paris"):
    return f"{nom}, {age} ans, de {ville}"

# Différentes façons d'appeler
profil1 = creer_profil("Alice", 25)
profil2 = creer_profil(age=30, nom="Bob", ville="Lyon")
```

### 4. Les valeurs de retour

Une fonction peut retourner :

- Une seule valeur
- Plusieurs valeurs (tuple)
- Rien (None par défaut)

```python
# Une valeur
def doubler(n):
    return n * 2

# Plusieurs valeurs
def calculer_stats(nombres):
    mini = min(nombres)
    maxi = max(nombres)
    moyenne = sum(nombres) / len(nombres)
    return mini, maxi, moyenne

# Utilisation
minimum, maximum, avg = calculer_stats([1, 2, 3, 4])
```

### 5. Variables locales et globales

```python
score = 0  # variable globale

def ajouter_points(points):
    global score  # déclare qu'on utilise la variable globale
    score += points

def nouvelle_partie():
    # score est une nouvelle variable locale
    score = 100
    print(f"Score initial : {score}")
```

### 6. Les fonctions lambda

Fonctions courtes et anonymes :

```python
# Fonction classique
def carre(x):
    return x ** 2

# Équivalent en lambda
carre = lambda x: x ** 2

# Utile pour le tri
etudiants = [("Alice", 15), ("Bob", 12), ("Charlie", 18)]
triés = sorted(etudiants, key=lambda x: x[1])  # tri par note
```

### ✍️ Exercice pratique guidé

```python
def valider_mot_de_passe(mdp):
    # Vérifie la longueur
    if len(mdp) < 8:
        return False, "Trop court"

    # Vérifie présence de chiffres
    if not any(c.isdigit() for c in mdp):
        return False, "Doit contenir un chiffre"

    # Vérifie présence de majuscules
    if not any(c.isupper() for c in mdp):
        return False, "Doit contenir une majuscule"

    return True, "Mot de passe valide"

# Test
valide, message = valider_mot_de_passe("Bonjour123")
print(message)  # "Mot de passe valide"
```

### 🎯 Points clés à retenir

1. `def` pour définir une fonction
2. Les paramètres sont optionnels
3. `return` pour renvoyer des valeurs
4. Une fonction sans `return` renvoie `None`
5. Les paramètres peuvent avoir des valeurs par défaut

### 🚀 Conseils de conception

1. Une fonction = une tâche précise
2. Noms clairs et descriptifs
3. Documentation avec docstrings
4. Gestion propre des erreurs
5. Tests des cas limites

Prêt pour les exercices ? 💪
