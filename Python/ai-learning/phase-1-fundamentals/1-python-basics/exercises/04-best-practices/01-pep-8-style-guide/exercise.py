"""
Exercices sur les bonnes pratiques Python
"""

from typing import List

def exercice_1():
    """
    Documentation et docstrings
    """
    class Inventaire:
        """
        Classe représentant un inventaire pour gérer des objets.

        Attributes:
            items (List[str]): La liste des objets dans l'inventaire.

        Example:
            >>> inventaire = Inventaire()
            >>> inventaire.ajouter("épée")
            >>> inventaire.afficher()
            ['épée']
        """
         
        def __init__(self):
            """Initialise un inventaire vide."""
            self.items = []
        
        def ajouter(self, item) -> None:
            """
            Ajoute un objet à l'inventaire.

            Args:
                item (str): Le nom de l'objet à ajouter.

            Raises:
                ValueError: Si l'objet est vide.
            """
            if not item:
                raise ValueError("L'objet ne peut pas être vide.")
            self.items.append(item)
        
        def retirer(self, item) -> None:
            """
            Retire un objet de l'inventaire.

            Args:
                item (str): Le nom de l'objet à retirer.

            Raises:
                ValueError: Si l'objet n'est pas dans l'inventaire.
            """
            if item not in self.items:
                raise ValueError(f"L'objet '{item}' n'est pas dans l'inventaire.")
            self.items.remove(item)
        
        def afficher(self) -> List[str]:
            """
            Affiche le contenu de l'inventaire.

            Returns:
                List[str]: La liste des objets dans l'inventaire.
            """
            print(self.items)
            return self.items

def exercice_2():
    """
    Style PEP 8
    """
    class gestionnaire_combat:
        """
        Classe pour gérer les combats entre deux joueurs.
        
        Attributes:
            joueur1: Le premier joueur (instance de Joueur).
            joueur2: Le deuxième joueur (instance de Joueur).
            tour: Le numéro du tour actuel.
        """
        def __init__(self,joueur1,joueur2):
            """
            Initialise le combat avec deux joueurs.

            Args:
                joueur1: Le premier joueur.
                joueur2: Le deuxième joueur.
            """
            self.j1=joueur1
            self.j2=joueur2
            self.tour=1
        
        def ATTAQUER(self):
            """Effectue une attaque du joueur 1 sur le joueur 2."""
            degats=10
            self.j2.points_de_vie-=degats
        
        def finDuCombat(self):
            """
            Vérifie si le combat est terminé.

            Returns:
                bool: True si l'un des joueurs a 0 points de vie ou moins.
            """
            return self.j1.points_de_vie<=0 or self.j2.points_de_vie<=0

def exercice_3():
    """
    Structure de projet - À implémenter dans un dossier séparé
    """
    # Créez la structure de projet décrite dans le README
    #     rpg_game/
    # ├── __init__.py
    # ├── personnages.py
    # ├── combat.py
    # ├── inventaire.py
    # tests/
    # ├── __init__.py
    # ├── test_personnages.py
    # ├── test_combat.py
    # ├── test_inventaire.py
    # README.md
    # requirements.txt
    # setup.py
    # .gitignore

    pass

if __name__ == "__main__":
    print("\n=== Exercice 1 : Documentation ===")
    exercice_1()
    
    print("\n=== Exercice 2 : Style PEP 8 ===")
    exercice_2()
    
    print("\n=== Exercice 3 : Structure de projet ===")
    exercice_3()