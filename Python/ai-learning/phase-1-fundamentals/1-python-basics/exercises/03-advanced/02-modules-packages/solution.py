"""
Script principal de test
"""

from mon_jeu.personnages.base import Personnage
from mon_jeu.inventaire.gestion import Item, Inventaire
from mon_jeu.combat.system import calculer_degats
from mon_jeu.utils.logger import log_evenement
from mon_jeu.utils.config import charger_config

def test_jeu():
    # Chargement de la configuration
    config = charger_config()
    log_evenement("Démarrage du jeu")

    # Création des personnages
    hero = Personnage("Alice")
    hero.inventaire = Inventaire()
    monstre = Personnage("Gobelin")

    # Test de l'inventaire
    epee = Item("Épée en fer", "arme")
    potion = Item("Potion de vie", "consommable")
    hero.inventaire.ajouter_item(epee)
    hero.inventaire.ajouter_item(potion)

    # Test de combat
    log_evenement("Début du combat")
    resultat = calculer_degats(hero, monstre)
    log_evenement(resultat)

    print("Test terminé avec succès!")

if __name__ == "__main__":
    test_jeu()