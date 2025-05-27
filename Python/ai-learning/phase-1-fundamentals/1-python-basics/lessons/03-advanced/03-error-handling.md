# La gestion des erreurs en Python 🛠️

## Les bases des exceptions

### Les erreurs communes

```python
# IndexError
liste = [1, 2, 3]
element = liste[10]  # IndexError: list index out of range

# TypeError
resultat = "2" + 2  # TypeError: can only concatenate str to str

# ValueError
nombre = int("abc")  # ValueError: invalid literal for int()

# KeyError
dict = {"a": 1}
valeur = dict["b"]  # KeyError: 'b'
```

## Try/Except : la base

```python
try:
    age = int(input("Âge : "))
except ValueError:
    print("Veuillez entrer un nombre valide")

# Version plus complète
try:
    resultat = 10 / nombre
except ZeroDivisionError:
    print("Division par zéro impossible")
except TypeError:
    print("Type invalide")
except Exception as e:
    print(f"Erreur inattendue : {e}")
else:
    print(f"Résultat : {resultat}")
finally:
    print("Fin du calcul")
```

## Lever ses propres erreurs

```python
class NiveauInvalideError(Exception):
    pass

def augmenter_niveau(personnage, niveaux):
    if niveaux <= 0:
        raise NiveauInvalideError("Le nombre de niveaux doit être positif")
    if personnage.niveau + niveaux > 100:
        raise ValueError("Niveau maximum dépassé")
    personnage.niveau += niveaux
```

## Assertions pour le débogage

```python
def attaquer(personnage, cible):
    assert personnage.points_de_vie > 0, "Personnage mort"
    assert cible.points_de_vie > 0, "Cible morte"
    # Logique d'attaque
```

## Contextes de gestion (with)

```python
# Gestion de fichiers
with open('sauvegarde.txt', 'w') as f:
    f.write('données')  # Fermeture automatique

# Gestion de plusieurs ressources
with open('entree.txt') as source, open('sortie.txt', 'w') as destination:
    destination.write(source.read())
```

## Bonnes pratiques

1. Exceptions spécifiques

```python
# ❌ Trop général
try:
    faire_quelque_chose()
except Exception:
    pass

# ✅ Spécifique
try:
    faire_quelque_chose()
except ValueError:
    # Gestion spécifique
```

2. Clean-up avec finally

```python
fichier = None
try:
    fichier = open('donnees.txt')
    # Traitement
except IOError:
    print("Erreur d'accès fichier")
finally:
    if fichier:
        fichier.close()
```

3. Contextes adaptés

```python
from contextlib import contextmanager

@contextmanager
def session_jeu():
    print("Démarrage session")
    try:
        yield
    finally:
        print("Sauvegarde et fermeture")
```

## Cas d'utilisation courants

```python
# Validation de données
def creer_personnage(nom, niveau):
    if not isinstance(nom, str):
        raise TypeError("Le nom doit être une chaîne")
    if not 1 <= niveau <= 100:
        raise ValueError("Niveau invalide (1-100)")

# Gestion de ressources
def sauvegarder_partie(personnage):
    try:
        with open('save.json', 'w') as f:
            json.dump(personnage.__dict__, f)
    except IOError as e:
        raise SauvegardeError("Échec sauvegarde") from e
```

Prêt pour les exercices ? 💪
