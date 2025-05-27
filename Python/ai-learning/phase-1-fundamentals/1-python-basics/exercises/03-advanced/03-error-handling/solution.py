"""
Solutions des exercices sur la gestion des erreurs
Author: Claude
Date: 2024-11-28
"""

import json
from contextlib import contextmanager
from datetime import datetime
import logging
import os

# Configuration du logging
logging.basicConfig(
    filename='game.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Exercice 1 : Validation d'entrées
def creer_personnage(nom, niveau, points_de_vie):
    """Crée un personnage avec validation des données"""
    if not isinstance(nom, str) or not nom.strip():
        raise ValueError("Le nom doit être une chaîne non vide")
    
    if not isinstance(niveau, int) or not 1 <= niveau <= 100:
        raise ValueError("Le niveau doit être entre 1 et 100")
    
    if not isinstance(points_de_vie, int) or points_de_vie <= 0:
        raise ValueError("Les points de vie doivent être positifs")
    
    return {"nom": nom, "niveau": niveau, "points_de_vie": points_de_vie}

def ajouter_item(inventaire, nom, type_item):
    """Ajoute un item à l'inventaire avec validation"""
    types_valides = {"arme", "potion", "armure"}
    
    if len(inventaire) >= 10:
        raise InventaireError("Inventaire plein")
    
    if type_item not in types_valides:
        raise ValueError(f"Type d'item invalide. Types acceptés : {types_valides}")
    
    inventaire.append({"nom": nom, "type": type_item})
    logging.info(f"Item ajouté : {nom} ({type_item})")

# Exercice 2 : Gestion de ressources
@contextmanager
def SauvegardePartie(nom_fichier, mode='w'):
    """Gestionnaire de contexte pour la sauvegarde"""
    fichier = None
    try:
        fichier = open(nom_fichier, mode)
        logging.debug(f"Ouverture du fichier {nom_fichier}")
        yield fichier
    except IOError as e:
        logging.error(f"Erreur d'accès fichier : {e}")
        raise
    finally:
        if fichier:
            fichier.close()
            logging.debug(f"Fermeture du fichier {nom_fichier}")

def sauvegarder_personnage(personnage):
    """Sauvegarde un personnage dans un fichier JSON"""
    nom_fichier = f"save_{personnage['nom']}.json"
    try:
        with SauvegardePartie(nom_fichier) as f:
            json.dump(personnage, f, indent=2)
            logging.info(f"Personnage sauvegardé : {personnage['nom']}")
    except Exception as e:
        logging.error(f"Erreur de sauvegarde : {e}")
        raise SauvegardeError("Échec de la sauvegarde") from e

def charger_personnage(nom_fichier):
    """Charge un personnage depuis un fichier JSON"""
    try:
        with SauvegardePartie(nom_fichier, 'r') as f:
            personnage = json.load(f)
            logging.info(f"Personnage chargé : {personnage['nom']}")
            return personnage
    except FileNotFoundError:
        raise ChargeError(f"Sauvegarde introuvable : {nom_fichier}")
    except json.JSONDecodeError:
        raise ChargeError("Fichier de sauvegarde corrompu")

# Exercice 3 : Exceptions personnalisées
class GameError(Exception):
    """Classe de base pour les exceptions du jeu"""
    pass

class InventaireError(GameError):
    """Erreurs liées à l'inventaire"""
    pass

class PersonnageError(GameError):
    """Erreurs liées aux personnages"""
    pass

class CombatError(GameError):
    """Erreurs liées au combat"""
    pass

class AttaqueImpossibleError(CombatError):
    """Erreur quand une attaque ne peut pas être effectuée"""
    pass

def attaquer(attaquant, cible):
    """Effectue une attaque avec gestion des erreurs"""
    try:
        if attaquant['points_de_vie'] <= 0:
            raise AttaqueImpossibleError("L'attaquant est mort")
        if cible['points_de_vie'] <= 0:
            raise AttaqueImpossibleError("La cible est morte")
            
        # Simulation de dégâts
        cible['points_de_vie'] -= 10
        logging.info(f"{attaquant['nom']} attaque {cible['nom']}")
        
    except Exception as e:
        logging.error(f"Erreur durant l'attaque : {e}")
        raise

if __name__ == "__main__":
    try:
        # Test de création de personnage
        hero = creer_personnage("Alice", 1, 100)
        inventaire = []
        
        # Test d'ajout d'items
        ajouter_item(inventaire, "Épée", "arme")
        ajouter_item(inventaire, "Potion", "potion")
        
        # Test de sauvegarde/chargement
        sauvegarder_personnage(hero)
        personnage_charge = charger_personnage(f"save_{hero['nom']}.json")
        
        # Test de combat
        monstre = creer_personnage("Gobelin", 1, 50)
        attaquer(hero, monstre)
        
        print("Tous les tests réussis !")
        
    except GameError as e:
        logging.error(f"Erreur de jeu : {e}")
        print(f"Erreur de jeu : {e}")
    except Exception as e:
        logging.critical(f"Erreur inattendue : {e}")
        print(f"Erreur inattendue : {e}")