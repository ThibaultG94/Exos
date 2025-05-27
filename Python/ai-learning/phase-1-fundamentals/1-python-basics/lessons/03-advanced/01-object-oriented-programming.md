# La Programmation Orientée Objet (POO) en Python 🎮

## Pourquoi la POO ?

Imaginons que vous créez un jeu avec différents personnages. Chaque personnage a :

- Des caractéristiques (points de vie, force, niveau...)
- Des actions (attaquer, se défendre, utiliser un objet...)

Comment organiser tout ça proprement ? La POO est la réponse !

## 1. Les classes : le concept de base

### Définition simple

Une classe est comme un "plan" pour créer des objets :

```python
class Personnage:
    def __init__(self, nom, pv):
        self.nom = nom
        self.points_de_vie = pv

    def est_vivant(self):
        return self.points_de_vie > 0
```

### Création d'objets (instances)

```python
# Création de personnages
hero = Personnage("Alice", 100)
monstre = Personnage("Dragon", 200)

print(hero.nom)           # "Alice"
print(monstre.est_vivant())  # True
```

## 2. Les attributs et méthodes

### Attributs : les données de l'objet

```python
class Joueur:
    # Attribut de classe (partagé par tous les joueurs)
    nombre_joueurs = 0

    def __init__(self, nom):
        # Attributs d'instance (propres à chaque joueur)
        self.nom = nom
        self.score = 0
        Joueur.nombre_joueurs += 1
```

### Méthodes : les actions de l'objet

```python
class Guerrier:
    def __init__(self, nom, force):
        self.nom = nom
        self.force = force
        self.vie = 100

    def attaquer(self, cible):
        degats = self.force * 2
        cible.vie -= degats
        return f"{self.nom} inflige {degats} dégâts à {cible.nom}"

    def boire_potion(self):
        self.vie += 20
        self.vie = min(self.vie, 100)  # Max 100 PV
```

## 3. L'héritage : réutiliser et spécialiser

### Concept de base

```python
class Personnage:
    def __init__(self, nom, vie):
        self.nom = nom
        self.vie = vie

    def se_presenter(self):
        return f"Je suis {self.nom}"

class Magicien(Personnage):  # Hérite de Personnage
    def __init__(self, nom, vie, mana):
        super().__init__(nom, vie)  # Appelle le init du parent
        self.mana = mana

    def lancer_sort(self):
        if self.mana >= 10:
            self.mana -= 10
            return "Boule de feu !"
        return "Pas assez de mana..."
```

### Polymorphisme

```python
class Guerrier(Personnage):
    def attaquer(self):
        return "Coup d'épée !"

class Archer(Personnage):
    def attaquer(self):
        return "Tir à l'arc !"

# Même méthode, comportement différent
personnages = [Guerrier("Conan", 100), Archer("Legolas", 80)]
for p in personnages:
    print(p.attaquer())  # Chaque classe a sa propre version
```

## 4. Encapsulation et propriétés

### Protection des attributs

```python
class CompteBancaire:
    def __init__(self, solde_initial):
        self._solde = solde_initial  # Convention : protégé
        self.__pin = "1234"          # Vraiment privé

    @property
    def solde(self):
        return self._solde

    @solde.setter
    def solde(self, nouveau_solde):
        if nouveau_solde >= 0:
            self._solde = nouveau_solde
        else:
            raise ValueError("Le solde ne peut pas être négatif")
```

## 5. Méthodes spéciales

### Personnalisation du comportement

```python
class Carte:
    def __init__(self, valeur, couleur):
        self.valeur = valeur
        self.couleur = couleur

    def __str__(self):
        return f"{self.valeur} de {self.couleur}"

    def __lt__(self, autre):
        return self.valeur < autre.valeur

    def __eq__(self, autre):
        return self.valeur == autre.valeur and self.couleur == autre.couleur
```

## Points clés à retenir

1. Une classe est un plan pour créer des objets
2. Les objets ont des attributs (données) et des méthodes (actions)
3. L'héritage permet de réutiliser du code
4. L'encapsulation protège les données
5. Les méthodes spéciales personnalisent le comportement

## Cas d'utilisation courants

1. Modélisation d'entités du monde réel

   - Joueurs, Personnages, Véhicules...
   - Comptes bancaires, Produits...

2. Organisation du code

   - Regrouper données et comportements
   - Créer des hiérarchies logiques

3. Création d'interfaces
   - Définir des comportements standards
   - Assurer la cohérence
