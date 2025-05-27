"""
Template pour les exercices de documentation
"""
from typing import Optional, List

# Exercice 1 : Documentation de classe
class Personnage:
    """
    Représente un personnage de jeu de rôle.

    Attributs:
        nom (str): Le nom du personnage.
        classe (str): La classe du personnage (ex: Guerrier, Mage).
        niveau (int): Le niveau du personnage, par défaut 1.
        experience (int): Points d'expérience accumulés.
        points_de_vie (int): Points de vie calculés en fonction du niveau.
        force (int): Force du personnage calculée en fonction du niveau.

    Exemple:
        >>> perso = Personnage("Arthas", "Guerrier")
        >>> perso.gagner_experience(150)
        >>> perso.niveau
        2
    """
    def __init__(self, nom: str, classe: str, niveau: int = 1):
        """
        Initialise un personnage avec les attributs de base.

        Args:
            nom (str): Le nom du personnage.
            classe (str): La classe du personnage.
            niveau (int, optional): Le niveau initial du personnage. Par défaut 1.
        """
        self.nom = nom
        self.classe = classe
        self.niveau = niveau
        self.experience = 0
        self.points_de_vie = niveau * 10
        self.force = niveau * 2

    def gagner_experience(self, montant: int) -> None:
        """
        Ajoute de l'expérience au personnage et monte le niveau si nécessaire.

        Args:
            montant (int): Points d'expérience gagnés.

        Raises:
            ValueError: Si le montant est négatif.
        """
        if montant < 0:
            raise ValueError("Le montant d'expérience doit être positif.")
        
        self.experience += montant
        while self.experience >= self.niveau * 100:
            self.monter_niveau()

    def monter_niveau(self) -> None:
        """
        Monte le niveau du personnage et met à jour ses attributs.
        """
        self.niveau += 1
        self.points_de_vie = self.niveau * 10
        self.force = self.niveau * 2
        self.experience -= (self.niveau - 1) * 100
    def __init__(self, nom: str, classe: str, niveau: int = 1):
        self.nom = nom
        self.classe = classe
        self.niveau = niveau
        self.experience = 0
        self.points_de_vie = niveau * 10
        self.force = niveau * 2

    def gagner_experience(self, montant: int) -> None:
        self.experience += montant
        while self.experience >= self.niveau * 100:
            self.monter_niveau()

    def monter_niveau(self) -> None:
        self.niveau += 1
        self.points_de_vie = self.niveau * 10
        self.force = self.niveau * 2
        self.experience -= (self.niveau - 1) * 100

# Exercice 2 : Module de combat
"""
Module de gestion des combats dans le jeu RPG.

Auteur: TonReuf
Date: Aujourd'hui
"""

from typing import Dict

class GestionnaireCombat:
    """
    Gère les combats entre deux personnages.

    Méthodes:
        lancer_combat: Initialise un combat entre deux personnages.
    """

    def lancer_combat(self, attaquant: Dict, defenseur: Dict) -> int:
        """
        Calcule les dégâts infligés pendant un combat.

        Args:
            attaquant (Dict): Stats de l'attaquant.
            defenseur (Dict): Stats du défenseur.

        Returns:
            int: Dégâts infligés.
        """
        force = attaquant.get("force", 0)
        defense = defenseur.get("defense", 0)
        bonus = attaquant.get("bonus", 0)

        degats = max(force - defense + bonus, 0)
        return degats


# Exercice 3 : Documentation de projet
# TODO: Créer la structure de projet et la documentation