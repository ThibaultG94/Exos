"""
Module de gestion des combats
"""
import random

def calculer_degats(attaquant, defenseur):
    degats = random.randint(5, 15)
    defenseur.points_de_vie -= degats
    return f"{attaquant.nom} inflige {degats} dégâts à {defenseur.nom}"