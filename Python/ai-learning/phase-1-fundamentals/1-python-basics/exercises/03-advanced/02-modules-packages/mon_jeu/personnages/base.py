"""
Module de base pour les personnages
"""

class Personnage:
    def __init__(self, nom):
        self.nom = nom
        self.points_de_vie = 100
        self.niveau = 1
        self.inventaire = None

    def __str__(self):
        return f"{self.nom} (Niveau {self.niveau})"