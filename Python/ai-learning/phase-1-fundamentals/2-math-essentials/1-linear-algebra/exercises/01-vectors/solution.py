"""
Solution des exercices sur les vecteurs et calculs vectoriels
Author: Claude
Date: 2024-12-04
"""

import numpy as np
from typing import Optional, Tuple, List
from math import cos, sin, pi

class Personnage:
    def __init__(self, position: Tuple[float, float]):
        self.position = np.array(position, dtype=float)
        self.direction = np.array([1.0, 0.0])  # Regarde vers la droite
        self.vitesse = 5.0  # Unités par seconde
        self.points_de_vie = 100

    def se_deplacer(self, delta_temps: float) -> None:
        """Déplace le personnage selon sa direction et sa vitesse"""
        deplacement = self.direction * self.vitesse * delta_temps
        self.position += deplacement

    def tourner(self, angle_degres: float) -> None:
        """Fait tourner le personnage d'un certain angle"""
        # Conversion en radians
        angle_rad = angle_degres * pi / 180.0
        
        # Matrice de rotation 2D
        cos_angle = cos(angle_rad)
        sin_angle = sin(angle_rad)
        rotation = np.array([[cos_angle, -sin_angle],
                           [sin_angle, cos_angle]])
        
        # Application de la rotation
        self.direction = np.dot(rotation, self.direction)
        # Renormalisation pour éviter l'accumulation d'erreurs
        self.direction /= np.linalg.norm(self.direction)

    def distance_vers(self, autre_personnage: 'Personnage') -> float:
        """Calcule la distance jusqu'à un autre personnage"""
        vecteur = autre_personnage.position - self.position
        return np.linalg.norm(vecteur)

    def regarder_vers(self, position_cible: Tuple[float, float]) -> None:
        """Oriente le personnage vers une position cible"""
        vecteur = np.array(position_cible) - self.position
        norme = np.linalg.norm(vecteur)
        
        if norme > 0:  # Évite la division par zéro
            self.direction = vecteur / norme

class GestionnaireCombat:
    def __init__(self, portee_attaque: float = 5.0, angle_vision: float = 90.0):
        self.portee_attaque = portee_attaque
        self.angle_vision = angle_vision

    def peut_attaquer(self, attaquant: Personnage, cible: Personnage) -> bool:
        """Vérifie si l'attaquant peut atteindre la cible"""
        # Vérification de la distance
        distance = attaquant.distance_vers(cible)
        if distance > self.portee_attaque:
            return False

        # Vérification de l'angle
        vers_cible = cible.position - attaquant.position
        vers_cible = vers_cible / np.linalg.norm(vers_cible)
        
        angle_rad = np.arccos(np.clip(np.dot(attaquant.direction, vers_cible), -1.0, 1.0))
        angle_deg = angle_rad * 180.0 / pi
        
        return angle_deg <= self.angle_vision / 2

    def calculer_degats(self, attaquant: Personnage, cible: Personnage) -> float:
        """Calcule les dégâts infligés par une attaque"""
        if not self.peut_attaquer(attaquant, cible):
            return 0.0

        distance = attaquant.distance_vers(cible)
        degats_base = 20.0  # Dégâts de base
        
        # Réduction des dégâts avec la distance
        facteur_distance = 1.0 - (distance / self.portee_attaque)
        
        # Bonus d'attaque de dos
        vers_cible = cible.position - attaquant.position
        vers_cible = vers_cible / np.linalg.norm(vers_cible)
        angle_dos = np.dot(cible.direction, vers_cible)
        bonus_dos = 1.5 if angle_dos > 0.5 else 1.0  # +50% de dos
        
        return degats_base * facteur_distance * bonus_dos

class AnalyseurMouvement:
    def __init__(self):
        self.historique_positions = []  # [(position, temps), ...]
        
    def enregistrer_position(self, position: np.ndarray, temps: float) -> None:
        """Enregistre une position avec son temps"""
        self.historique_positions.append((position.copy(), temps))

    def calculer_statistiques(self) -> dict:
        """Calcule les statistiques de mouvement"""
        if len(self.historique_positions) < 2:
            return {
                "vitesse_moyenne": 0.0,
                "acceleration_moyenne": 0.0,
                "distance_totale": 0.0
            }

        # Calcul de la vitesse moyenne
        vitesses = []
        distance_totale = 0.0
        
        for i in range(1, len(self.historique_positions)):
            pos_prev, t_prev = self.historique_positions[i-1]
            pos_curr, t_curr = self.historique_positions[i]
            
            delta_t = t_curr - t_prev
            if delta_t > 0:
                deplacement = pos_curr - pos_prev
                distance = np.linalg.norm(deplacement)
                vitesses.append(distance / delta_t)
                distance_totale += distance

        vitesse_moyenne = np.mean(vitesses)
        
        # Calcul de l'accélération
        accelerations = []
        for i in range(1, len(vitesses)):
            delta_t = (self.historique_positions[i+1][1] - 
                      self.historique_positions[i][1])
            if delta_t > 0:
                acc = (vitesses[i] - vitesses[i-1]) / delta_t
                accelerations.append(acc)

        acceleration_moyenne = np.mean(accelerations) if accelerations else 0.0

        return {
            "vitesse_moyenne": vitesse_moyenne,
            "acceleration_moyenne": acceleration_moyenne,
            "distance_totale": distance_totale
        }

    def predire_position_future(self, delta_temps: float) -> np.ndarray:
        """Prédit la position future basée sur l'historique"""
        if len(self.historique_positions) < 2:
            return self.historique_positions[-1][0] if self.historique_positions else np.zeros(2)

        # Calcul du vecteur de vitesse moyen récent
        pos_last, t_last = self.historique_positions[-1]
        pos_prev, t_prev = self.historique_positions[-2]
        
        vitesse_vector = (pos_last - pos_prev) / (t_last - t_prev)
        
        # Prédiction linéaire simple
        return pos_last + vitesse_vector * delta_temps

# Tests de validation
if __name__ == "__main__":
    # Test du personnage
    hero = Personnage((0.0, 0.0))
    hero.tourner(45)  # Tourne de 45 degrés
    hero.se_deplacer(1.0)  # Avance pendant 1 seconde
    print(f"Position après déplacement : {hero.position}")

    # Test du combat
    combat = GestionnaireCombat()
    ennemi = Personnage((3.0, 3.0))
    peut_attaquer = combat.peut_attaquer(hero, ennemi)
    degats = combat.calculer_degats(hero, ennemi)
    print(f"Peut attaquer : {peut_attaquer}, Dégâts : {degats}")

    # Test de l'analyseur
    analyseur = AnalyseurMouvement()
    for t in range(5):
        pos = hero.position.copy()
        hero.se_deplacer(1.0)
        analyseur.enregistrer_position(pos, float(t))
    
    stats = analyseur.calculer_statistiques()
    print("Statistiques :", stats)