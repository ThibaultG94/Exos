"""
Solutions des exercices sur les bonnes pratiques Python
Author: Claude
Date: 2024-11-29
"""

from typing import Any, List, Optional
from dataclasses import dataclass

def exercice_1():
    """
    Solution de l'exercice sur la documentation
    """
    @dataclass
    class Item:
        """Représente un item dans l'inventaire."""
        nom: str
        type: str
        quantite: int = 1

    class Inventaire:
        """
        Gère une collection d'items pour un personnage.

        Cette classe permet d'ajouter, retirer et visualiser les items
        d'un inventaire de jeu.

        Attributes:
            items (List[Item]): Liste des items dans l'inventaire

        Example:
            >>> inv = Inventaire()
            >>> inv.ajouter(Item("Épée", "Arme"))
            >>> inv.afficher()
            ['Épée (Arme)']
        """

        def __init__(self) -> None:
            """Initialise un inventaire vide."""
            self.items: List[Item] = []

        def ajouter(self, item: Item) -> None:
            """
            Ajoute un item à l'inventaire.

            Args:
                item (Item): L'item à ajouter

            Raises:
                ValueError: Si l'item est None
            """
            if not item:
                raise ValueError("Item invalide")
            self.items.append(item)

        def retirer(self, item: Item) -> None:
            """
            Retire un item de l'inventaire.

            Args:
                item (Item): L'item à retirer

            Raises:
                ValueError: Si l'item n'est pas dans l'inventaire
            """
            try:
                self.items.remove(item)
            except ValueError:
                raise ValueError(f"Item {item.nom} non trouvé dans l'inventaire")

        def afficher(self) -> None:
            """
            Affiche le contenu de l'inventaire.

            Example:
                >>> inv = Inventaire()
                >>> inv.ajouter(Item("Potion", "Consommable"))
                >>> inv.afficher()
                Inventaire (1 items):
                - Potion (Consommable)
            """
            print(f"Inventaire ({len(self.items)} items):")
            for item in self.items:
                print(f"- {item.nom} ({item.type})")

def exercice_2():
    """
    Solution de l'exercice sur le style PEP 8
    """
    class GestionnaireCombat:
        """Gère un combat entre deux joueurs."""

        def __init__(self, joueur1: str, joueur2: str) -> None:
            """
            Initialise un nouveau combat.

            Args:
                joueur1: Premier combattant
                joueur2: Second combattant
            """
            self.joueur1 = joueur1
            self.joueur2 = joueur2
            self.tour = 1

        def attaquer(self) -> int:
            """
            Effectue une attaque du joueur actif.

            Returns:
                int: Montant des dégâts infligés
            """
            degats = 10
            self.joueur2.points_de_vie -= degats
            return degats

        def est_termine(self) -> bool:
            """
            Vérifie si le combat est terminé.

            Returns:
                bool: True si un des joueurs est vaincu
            """
            return (self.joueur1.points_de_vie <= 0 or 
                   self.joueur2.points_de_vie <= 0)

def exercice_3():
    """
    Solution pour la structure de projet
    """
    # Création de la structure (à implémenter avec os/pathlib)
    project_structure = {
        'rpg_game': {
            '__init__.py': '',
            'personnages': {
                '__init__.py': '',
                'base.py': 'class Personnage:\n    pass',
            },
            'combat': {
                '__init__.py': '',
                'systeme.py': 'class Combat:\n    pass',
            },
            'inventaire': {
                '__init__.py': '',
                'gestion.py': 'class Inventaire:\n    pass',
            },
        },
        'tests': {
            '__init__.py': '',
            'test_personnages.py': '',
            'test_combat.py': '',
            'test_inventaire.py': '',
        },
        'docs': {
            'conf.py': '',
            'index.rst': '',
        },
        'README.md': '# RPG Game\n\nDescription du projet...',
        'requirements.txt': 'pytest>=7.0.0\nsphinx>=4.0.0',
        'setup.py': 'from setuptools import setup\n\nsetup(\n    name="rpg_game"\n)',
        '.gitignore': '*.pyc\n__pycache__\n.pytest_cache',
    }

if __name__ == "__main__":
    print("\n=== Exercice 1 : Documentation ===")
    exercice_1()
    
    print("\n=== Exercice 2 : Style PEP 8 ===")
    exercice_2()
    
    print("\n=== Exercice 3 : Structure de projet ===")
    exercice_3()