"""
Script principal de test
"""

from mon_jeu.personnages.base import Personnage
from mon_jeu.inventaire.gestion import Item, Inventaire
from mon_jeu.combat.system import calculer_degats

def test_jeu():
    # À implémenter
    # Créer un personnage
    hero = Personnage("Hero")
    # Créer un inventaire
    inventaire = Inventaire()
    # Créer un item
    item = Item("Épée", "Arme")
    # Ajouter l'item à l'inventaire
    inventaire.ajouter_item(item)
    # Afficher le personnage, l'inventaire et l'item
    print('Personnage:', hero, 'Inventaire:', inventaire.items, 'Item:', item)
    # Créer un ennemi
    ennemi = Personnage("Ennemi")
    # Calculer les dégâts
    print(calculer_degats(hero, ennemi))
    # Afficher les points de vie de l'ennemi
    print('Points de vie de l\'ennemi:', ennemi.points_de_vie)
    pass

if __name__ == "__main__":
    test_jeu()