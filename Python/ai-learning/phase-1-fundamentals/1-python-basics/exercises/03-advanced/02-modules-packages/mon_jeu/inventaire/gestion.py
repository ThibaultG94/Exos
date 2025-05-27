"""
Module de gestion d'inventaire
"""

class Item:
    def __init__(self, nom, type_item):
        self.nom = nom
        self.type = type_item

    def __str__(self):
        return f"{self.nom} ({self.type})"

class Inventaire:
    def __init__(self):
        self.items = []

    def ajouter_item(self, item):
        self.items.append(item)
        return f"{item} ajouté à l'inventaire"