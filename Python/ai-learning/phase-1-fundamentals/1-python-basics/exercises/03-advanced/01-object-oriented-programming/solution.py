"""
Solutions des exercices de Programmation Orientée Objet
Author: Claude
Date: 2024-11-26
"""

def exercise_1():
    """
    Solution de l'exercice 1 : Création d'une classe Personnage
    """
    class Personnage:
        def __init__(self, nom, points_de_vie):
            self.nom = nom
            self.points_de_vie = points_de_vie
            self.niveau = 1

        def afficher_statut(self):
            return f"{self.nom} (Niv.{self.niveau}) - PV: {self.points_de_vie}"

        def est_vivant(self):
            return self.points_de_vie > 0

        def subir_degats(self, degats):
            self.points_de_vie = max(0, self.points_de_vie - degats)

    # Tests
    hero = Personnage("Alice", 100)
    monstre = Personnage("Dragon", 150)

    print(hero.afficher_statut())
    print(monstre.afficher_statut())

    # Simulation de combat
    hero.subir_degats(30)
    monstre.subir_degats(50)

    print(f"\nAprès le combat :")
    print(hero.afficher_statut(), f"- En vie : {hero.est_vivant()}")
    print(monstre.afficher_statut(), f"- En vie : {monstre.est_vivant()}")

def exercise_2():
    """
    Solution de l'exercice 2 : Gestion d'un inventaire
    """
    class Item:
        def __init__(self, nom, type_item, valeur):
            self.nom = nom
            self.type = type_item
            self.valeur = valeur

        def decrire(self):
            return f"{self.nom} ({self.type}) - {self.valeur} pièces d'or"

    class Inventaire:
        def __init__(self):
            self.items = []

        def ajouter_item(self, item):
            self.items.append(item)
            print(f"Ajout de : {item.nom}")

        def retirer_item(self, nom_item):
            for item in self.items:
                if item.nom == nom_item:
                    self.items.remove(item)
                    print(f"Retrait de : {nom_item}")
                    return True
            print(f"Item non trouvé : {nom_item}")
            return False

        def calculer_valeur_totale(self):
            return sum(item.valeur for item in self.items)

        def lister_items(self):
            if not self.items:
                print("Inventaire vide")
                return
            print("\nContenu de l'inventaire :")
            for item in self.items:
                print(f"- {item.decrire()}")
            print(f"Valeur totale : {self.calculer_valeur_totale()} pièces d'or")

    # Tests
    inv = Inventaire()
    
    # Création et ajout d'items
    items = [
        Item("Épée en fer", "arme", 50),
        Item("Potion de vie", "potion", 30),
        Item("Bouclier en bois", "armure", 40)
    ]
    
    for item in items:
        inv.ajouter_item(item)
    
    inv.lister_items()
    
    # Test de retrait
    inv.retirer_item("Potion de vie")
    inv.lister_items()

def exercise_3():
    """
    Solution de l'exercice 3 : Système de combat
    """
    class Personnage:
        def __init__(self, nom, points_de_vie):
            self.nom = nom
            self.points_de_vie = points_de_vie
            self.niveau = 1

        def est_vivant(self):
            return self.points_de_vie > 0

    class Combattant(Personnage):
        def __init__(self, nom, points_de_vie, force, defense):
            super().__init__(nom, points_de_vie)
            self.force = force
            self.defense = defense
            self.defense_active = False

        def attaquer(self, cible):
            degats = self.force
            if cible.defense_active:
                degats = max(0, degats - cible.defense)
            cible.subir_degats(degats)
            return f"{self.nom} inflige {degats} dégâts à {cible.nom}"

        def defendre(self):
            self.defense_active = True
            return f"{self.nom} se met en position de défense"

        def subir_degats(self, degats):
            if self.defense_active:
                degats = max(0, degats - self.defense)
            self.points_de_vie = max(0, self.points_de_vie - degats)
            self.defense_active = False

    class Guerrier(Combattant):
        def coup_puissant(self, cible):
            degats = self.force * 2
            cible.subir_degats(degats)
            return f"{self.nom} assène un coup puissant et inflige {degats} dégâts à {cible.nom}"

    class Mage(Combattant):
        def __init__(self, nom, points_de_vie, force, defense, mana):
            super().__init__(nom, points_de_vie, force, defense)
            self.mana = mana
            self.mana_max = mana

        def boule_de_feu(self, cible):
            cout_mana = 20
            if self.mana >= cout_mana:
                self.mana -= cout_mana
                degats = self.force * 3
                cible.subir_degats(degats)
                return f"{self.nom} lance une boule de feu et inflige {degats} dégâts à {cible.nom}"
            return f"{self.nom} n'a pas assez de mana"

    # Test du système de combat
    guerrier = Guerrier("Conan", 150, 15, 10)
    mage = Mage("Gandalf", 100, 8, 5, 100)

    print("=== Début du combat ===")
    print(f"{guerrier.nom}: {guerrier.points_de_vie} PV")
    print(f"{mage.nom}: {mage.points_de_vie} PV")

    # Simulation d'un combat
    print("\nTour 1:")
    print(mage.boule_de_feu(guerrier))
    print(guerrier.coup_puissant(mage))

    print("\nStatut après le tour 1:")
    print(f"{guerrier.nom}: {guerrier.points_de_vie} PV")
    print(f"{mage.nom}: {mage.points_de_vie} PV")

if __name__ == "__main__":
    print("\n=== Exercise 1 : Création d'une classe Personnage ===")
    exercise_1()
    
    print("\n=== Exercise 2 : Gestion d'un inventaire ===")
    exercise_2()
    
    print("\n=== Exercise 3 : Système de combat ===")
    exercise_3()