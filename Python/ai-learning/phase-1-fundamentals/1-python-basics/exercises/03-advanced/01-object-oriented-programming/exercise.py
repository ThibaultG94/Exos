def exercise_1():
    """
    Création d'une classe Personnage
    """
    class Personnage:
        def __init__(self, nom, points_de_vie):
            # À compléter
            self.nom = nom
            self.points_de_vie = points_de_vie
            self.niveau = 1
            pass

        def afficher_statut(self):
            # À compléter
            print(f"{self.nom} (niveau {self.niveau}) a {self.points_de_vie} points de vie.")
            pass

        def est_vivant(self):
            # À compléter
            if self.points_de_vie > 0:
                print(f"{self.nom} est vivant.")
            else:
                print(f"{self.nom} est mort.")
            pass

        def attaquer(self, cible):
            # À compléter
            print(f"{self.nom} attaque {cible.nom} !")
            cible.points_de_vie -= 20
            pass

    hero = Personnage("Azael", 100)
    monstre = Personnage("Goblin", 50)
    hero.afficher_statut()
    monstre.afficher_statut()
    hero.est_vivant()
    monstre.est_vivant()

    # Faire combattre les deux personnages
    hero.attaquer(monstre)
    monstre.attaquer(hero)
    hero.attaquer(monstre)
    monstre.attaquer(hero)
    hero.attaquer(monstre)

    hero.afficher_statut()
    monstre.afficher_statut()
    hero.est_vivant()
    monstre.est_vivant()

def exercise_2():
    """
    Gestion d'un inventaire
    """
    class Item:
        def __init__(self, nom, type_item, valeur):
            # À compléter
            self.nom = nom
            self.type_item = type_item
            self.valeur = valeur
            pass

        def decrire(self):
            # À compléter la description de l'item (arme/potion/armure)
            print(f"{self.nom} ({self.type_item}) vaut {self.valeur} pièces d'or.")
            pass

    class Inventaire:
        def __init__(self):
            # À compléter
            self.items = []
            pass

        def ajouter_item(self, item):
            # À compléter
            self.items.append(item)
            print(f"{item.nom} a été ajouté à l'inventaire.")
            pass

        def retirer_item(self, nom_item):
            # À compléter
            for item in self.items:
                if item.nom == nom_item:
                    self.items.remove(item)
                    break
            print(f"{nom_item} a été retiré de l'inventaire.")
            pass

        def calculer_valeur_totale(self):
            # À compléter
            total = 0
            for item in self.items:
                total += item.valeur
            print(f"La valeur totale de l'inventaire est de {total} pièces d'or.")
        
        def lister_items(self):
            # À compléter
            for item in self.items:
                item.decrire()
            pass
    
    # Création des items
    epee = Item("Épée", "Arme", 50)
    potion = Item("Potion de soin", "Potion", 10)
    armure = Item("Armure de cuir", "Armure", 30)

    # Création de l'inventaire
    inventaire = Inventaire()
    inventaire.ajouter_item(epee)
    inventaire.ajouter_item(potion)
    inventaire.ajouter_item(armure)
    inventaire.calculer_valeur_totale()
    inventaire.lister_items()
    inventaire.retirer_item("Potion de soin")
    inventaire.calculer_valeur_totale()
    inventaire.lister_items()

def exercise_3():
    """
    Système de combat
    """
    class Personnage:
        def __init__(self, nom, points_de_vie):
            self.nom = nom
            self.points_de_vie = points_de_vie
            self.niveau = 1

    class Combattant(Personnage):
        def __init__(self, nom, points_de_vie, force, defense):
            # À compléter
            self.force = force
            self.defense = defense
            self.nom = nom
            self.points_de_vie = points_de_vie
            pass

        def attaquer(self, cible):
            # À compléter
            print(f"{self.nom} attaque {cible.nom} !")
            cible.points_de_vie -= self.force - cible.defense
            print(f"{cible.nom} a {cible.points_de_vie} points de vie.")
            pass

        def defendre(self):
            # À compléter
            print(f"{self.nom} se défend !")
            self.defense += 10
            pass

    class Guerrier(Combattant):
        def coup_puissant(self, cible):
            # À compléter
            print(f"{self.nom} utilise Coup puissant sur {cible.nom} !")
            cible.points_de_vie -= self.force * 2 - cible.defense
            print(f"{cible.nom} a {cible.points_de_vie} points de vie.")
            pass

    class Mage(Combattant):
        def __init__(self, nom, points_de_vie, force, defense, mana):
            # À compléter
            self.mana = mana
            self.nom = nom
            self.points_de_vie = points_de_vie
            self.force = force
            self.defense = defense
            pass

        def boule_de_feu(self, cible):
            # À compléter
            print(f"{self.nom} lance Boule de feu sur {cible.nom} !")
            cible.points_de_vie -= self.force * 3 - cible.defense
            print(f"{cible.nom} a {cible.points_de_vie} points de vie.")
            pass

    # Création des personnages
    guerrier = Guerrier("Conan", 100, 20, 10)
    mage = Mage("Merlin", 80, 10, 5, 50)
    # Faire combattre les deux personnages
    guerrier.attaquer(mage)
    mage.boule_de_feu(guerrier)
    guerrier.coup_puissant(mage)
    mage.attaquer(guerrier)
    guerrier.defendre()
    mage.boule_de_feu(guerrier)
    guerrier.attaquer(mage)
    mage.boule_de_feu(guerrier)
    guerrier.attaquer(mage)

if __name__ == "__main__":
    # Tests pour chaque exercice
    print("\n=== Exercise 1 : Création d'une classe Personnage ===")
    exercise_1()
    
    print("\n=== Exercise 2 : Gestion d'un inventaire ===")
    exercise_2()
    
    print("\n=== Exercise 3 : Système de combat ===")
    exercise_3()