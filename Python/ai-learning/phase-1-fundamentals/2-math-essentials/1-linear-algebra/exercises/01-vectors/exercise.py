"""
Template d'exercices sur les vecteurs et calculs vectoriels
Author: Claude
Date: 2024-12-04
"""

import numpy as np
from typing import Optional, Tuple, List

class Personnage:
    def __init__(self, position: Tuple[float, float]):
        """
        Initialise un personnage avec une position de départ.
        
        Args:
            position: Tuple (x, y) de la position initiale
        """
        self.position = np.array(position, dtype=float)
        self.direction = np.array([1.0, 0.0])  # Regarde vers la droite par défaut
        self.vitesse = 5.0  # Unités par seconde
        self.points_de_vie = 100

    def se_deplacer(self, delta_temps: float) -> None:
        """
        Déplace le personnage selon sa direction et sa vitesse.
        
        Args:
            delta_temps: Temps écoulé depuis le dernier déplacement en secondes
        """
        self.position += self.direction * self.vitesse * delta_temps

    def tourner(self, angle_degres: float) -> None:
        """
        Fait tourner le personnage d'un certain angle.
        
        Args:
            angle_degres: Angle de rotation en degrés (positif = sens horaire)
        """
        angle_radians = np.deg2rad(angle_degres)
        rotation = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians), np.cos(angle_radians)]
        ])
        self.direction = np.dot(rotation, self.direction)

    def distance_vers(self, autre_personnage: 'Personnage') -> float:
        """
        Calcule la distance jusqu'à un autre personnage.
        
        Args:
            autre_personnage: Le personnage cible
            
        Returns:
            Distance en unités de jeu
        """
        return np.linalg.norm(self.position - autre_personnage.position)

    def regarder_vers(self, position_cible: Tuple[float, float]) -> None:
        """
        Oriente le personnage vers une position cible.
        
        Args:
            position_cible: Tuple (x, y) de la position à regarder
        """
        direction = np.array(position_cible) - self.position
        if np.linalg.norm(direction) > 0.0:
            self.direction = direction / np.linalg.norm(direction)

class GestionnaireCombat:
    def __init__(self, portee_attaque: float = 5.0, angle_vision: float = 90.0):
        """
        Initialise le gestionnaire de combat avec des paramètres.
        
        Args:
            portee_attaque: Distance maximale d'attaque
            angle_vision: Angle de vision en degrés
        """
        self.portee_attaque = portee_attaque
        self.angle_vision = angle_vision

    def peut_attaquer(self, attaquant: Personnage, cible: Personnage) -> bool:
        """
        Vérifie si l'attaquant peut atteindre la cible.
        
        Args:
            attaquant: Personnage qui attaque
            cible: Personnage ciblé
            
        Returns:
            True si l'attaque est possible
        """
        vecteur_vers_cible = cible.position - attaquant.position
        distance = np.linalg.norm(vecteur_vers_cible)
        if distance > self.portee_attaque:
            return False
        
        direction_normalisee = vecteur_vers_cible / distance
        angle_cos = np.dot(attaquant.direction, direction_normalisee)
        return angle_cos >= np.cos(np.deg2rad(self.angle_vision / 2))

    def calculer_degats(self, attaquant: Personnage, cible: Personnage) -> float:
        """
        Calcule les dégâts infligés par une attaque.
        
        Args:
            attaquant: Personnage qui attaque
            cible: Personnage ciblé
            
        Returns:
            Montant des dégâts
        """
        vecteur_vers_cible = cible.position - attaquant.position
        distance = np.linalg.norm(vecteur_vers_cible)
        base_degats = 20
        if distance > self.portee_attaque:
            return 0
        facteur_distance = 1 - (distance / self.portee_attaque)
        bonus_dos = 1.5 if np.dot(cible.direction, -vecteur_vers_cible / distance) < -0.5 else 1
        return base_degats * facteur_distance * bonus_dos

class AnalyseurMouvement:
    def __init__(self):
        self.historique_positions = []  # [(position, temps), ...]
        
    def enregistrer_position(self, position: np.ndarray, temps: float) -> None:
        """
        Enregistre une position avec son temps.
        
        Args:
            position: Vecteur numpy de position
            temps: Temps en secondes
        """
        self.historique_positions.append((position.copy(), temps))

    def calculer_statistiques(self) -> dict:
        """
        Calcule les statistiques de mouvement.
        
        Returns:
            Dictionnaire avec vitesse moyenne, accélération, distance totale
        """
        if len(self.historique_positions) < 2:
            return {"vitesse_moyenne": 0, "acceleration_moyenne": 0, "distance_totale": 0}
        
        distances = []
        temps_deltas = []
        for i in range(1, len(self.historique_positions)):
            pos1, t1 = self.historique_positions[i - 1]
            pos2, t2 = self.historique_positions[i]
            distances.append(np.linalg.norm(pos2 - pos1))
            temps_deltas.append(t2 - t1)

        distance_totale = sum(distances)
        vitesse_moyenne = distance_totale / sum(temps_deltas)
        return {"vitesse_moyenne": vitesse_moyenne, "distance_totale": distance_totale}

    def predire_position_future(self, delta_temps: float) -> np.ndarray:
        """
        Prédit la position future basée sur l'historique.
        
        Args:
            delta_temps: Temps dans le futur en secondes
            
        Returns:
            Vecteur numpy de la position prédite
        """
        if len(self.historique_positions) < 2:
            return self.historique_positions[-1][0] if self.historique_positions else np.array([0.0, 0.0])
        
        pos1, t1 = self.historique_positions[-2]
        pos2, t2 = self.historique_positions[-1]
        vitesse = (pos2 - pos1) / (t2 - t1)
        return pos2 + vitesse * delta_temps

# Tests
if __name__ == "__main__":
    # Créez ici vos tests pour valider les implémentations
    pass
