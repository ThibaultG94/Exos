"""
Exercices sur la gestion des erreurs
"""

# Exercice 1 : Validation d'entrées
def creer_personnage(nom, niveau, points_de_vie):
    # À implémenter

    # Vérifie que nom est une chaîne non vide
    if not isinstance(nom, str) or len(nom) == 0:
        raise ValueError("Le nom doit être une chaîne non vide")
    
    # Vérifie que niveau est entre 1 et 100
    if not isinstance(niveau, int) or niveau < 1 or niveau > 100:
        raise ValueError("Le niveau doit être un entier entre 1 et 100")
    
    # Vérifie que points_de_vie est positif
    if not isinstance(points_de_vie, int) or points_de_vie < 0:
        raise ValueError("Les points de vie doivent être un entier positif")
    
    # Retourne un objet Personnage si tout est valide
    return (nom, niveau, points_de_vie)

    pass

def ajouter_item(inventaire, nom, type_item):
    # À implémenter

    # Vérifie que l'inventaire n'est pas plein (max 10 items)
    if len(inventaire) >= 10:
        raise ValueError("L'inventaire est plein")
    
    # Vérifie que le type_item est valide (arme/potion/armure)
    if type_item not in ["arme", "potion", "armure"]:
        raise ValueError("Le type d'item doit être 'arme', 'potion' ou 'armure'")
    
    pass

# Exercice 2 : Gestion de ressources
from contextlib import contextmanager

@contextmanager
def SauvegardePartie(nom_fichier, mode='w'):
    # À implémenter

    # Ouvre un fichier de sauvegarde en mode écriture
    # Gère les erreurs d'écriture/lecture
    # Assure la fermeture du fichier même en cas d'erreur
    # Permet d'utiliser le with statement
    with open(nom_fichier, mode) as f:
        f.write("Sauvegarde de la partie")

    pass

def sauvegarder_personnage(personnage, type_item):
    # À implémenter
    # Implémentez les fonctions :
        # sauvegarder_personnage(personnage)
        # charger_personnage(nom_fichier) 
        if type_item not in ["arme", "potion", "armure"]:
            raise ValueError("Le type d'item doit être 'arme', 'potion' ou 'armure'")
        
        with SauvegardePartie("sauvegarde.txt") as f:
            f.write(f"{personnage} - {type_item}")

def charger_personnage(nom_fichier):
    # À implémenter
    pass

# Exercice 3 : Exceptions personnalisées
class GameError(Exception):
    """Classe de base pour les exceptions du jeu"""
    pass

class InventaireError(GameError):
    # À implémenter
    pass

class PersonnageError(GameError):
    # À implémenter
    pass

def attaquer(attaquant, cible):
    # À implémenter
    pass

if __name__ == "__main__":
    # Tests des différentes fonctions
    try:
        # À implémenter : tests de vos fonctions
        pass
    except Exception as e:
        print(f"Une erreur est survenue : {e}")