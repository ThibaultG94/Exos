"""
Solution des exercices sur la documentation Python
Author: Claude
Date: 2024-11-30
Version: 1.0

Ce module implémente les fonctionnalités de base d'un jeu de rôle,
notamment la gestion des personnages et du système de combat.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class ClassePersonnage(Enum):
    """Énumération des classes de personnage disponibles."""
    GUERRIER = "guerrier"
    MAGE = "mage"
    ARCHER = "archer"

class Personnage:
    """
    Représente un personnage jouable dans le jeu.

    Cette classe gère les attributs et le système de progression
    des personnages, incluant les points de vie, la force et 
    l'expérience.

    Attributes:
        nom (str): Nom du personnage
        classe (ClassePersonnage): Classe du personnage
        niveau (int): Niveau actuel (1-100)
        experience (int): Points d'expérience accumulés
        points_de_vie (int): Points de vie actuels
        force (int): Force du personnage

    Example:
        >>> hero = Personnage("Gandalf", ClassePersonnage.MAGE)
        >>> hero.gagner_experience(150)
        >>> print(hero.niveau)
        2
    """

    def __init__(self, nom: str, classe: ClassePersonnage, niveau: int = 1):
        """
        Initialise un nouveau personnage.

        Args:
            nom: Nom du personnage
            classe: Classe du personnage
            niveau: Niveau de départ (défaut: 1)

        Raises:
            ValueError: Si le niveau est inférieur à 1
        """
        if niveau < 1:
            raise ValueError("Le niveau doit être positif")

        self.nom = nom
        self.classe = classe
        self.niveau = niveau
        self.experience = 0
        self.points_de_vie = niveau * 10
        self.force = niveau * 2

    def gagner_experience(self, montant: int) -> None:
        """
        Ajoute de l'expérience au personnage et gère la montée de niveau.

        Args:
            montant: Quantité d'expérience à ajouter

        Raises:
            ValueError: Si le montant est négatif
        """
        if montant < 0:
            raise ValueError("Le gain d'expérience doit être positif")

        self.experience += montant
        # Tant que le personnage a assez d'XP, il monte de niveau
        while self.experience >= self.niveau * 100:
            self.monter_niveau()

    def monter_niveau(self) -> None:
        """
        Fait progresser le personnage d'un niveau.

        Cette méthode met à jour les statistiques du personnage
        et ajuste l'expérience en conséquence.
        """
        self.niveau += 1
        # Mise à jour des statistiques
        self.points_de_vie = self.niveau * 10
        self.force = self.niveau * 2
        # Soustraction du coût en XP du niveau
        self.experience -= (self.niveau - 1) * 100

class CombatError(Exception):
    """Classe de base pour les erreurs liées au combat."""
    pass

class GestionnaireCombat:
    """
    Gère le système de combat entre personnages.

    Cette classe implémente les mécaniques de combat,
    incluant les calculs de dégâts et la gestion des tours.

    Attributes:
        BONUS_CRITIQUE (float): Multiplicateur de dégâts critiques
        chances_critique (float): Probabilité de coup critique
    """

    BONUS_CRITIQUE: float = 1.5
    
    def __init__(self, chances_critique: float = 0.1):
        self.chances_critique = chances_critique

    def calculer_degats(self, 
                       attaquant: Personnage, 
                       defenseur: Personnage,
                       modificateur: float = 1.0) -> int:
        """
        Calcule les dégâts infligés lors d'une attaque.

        Args:
            attaquant: Personnage qui attaque
            defenseur: Personnage qui défend
            modificateur: Multiplicateur de dégâts

        Returns:
            Montant des dégâts infligés

        Raises:
            CombatError: Si un des personnages est KO
        """
        if attaquant.points_de_vie <= 0:
            raise CombatError("L'attaquant est KO")
        if defenseur.points_de_vie <= 0:
            raise CombatError("Le défenseur est KO")

        degats_base = attaquant.force * modificateur
        return max(1, int(degats_base))  # Minimum 1 dégât

if __name__ == "__main__":
    # Exemple d'utilisation
    hero = Personnage("Aragorn", ClassePersonnage.GUERRIER)
    monstre = Personnage("Gobelin", ClassePersonnage.GUERRIER)
    
    combat = GestionnaireCombat()
    degats = combat.calculer_degats(hero, monstre)
    print(f"{hero.nom} inflige {degats} points de dégâts à {monstre.nom}")