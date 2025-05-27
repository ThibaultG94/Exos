"""
Solution des exercices sur les matrices
Date: 2024-12-05
"""

import numpy as np
from typing import List, Tuple, Optional
from math import cos, sin, pi

class TraitementImage:
    def __init__(self, image: np.ndarray):
        self.verifier_image(image)
        self.image = image.astype(float)
    
    def verifier_image(self, image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray) or len(image.shape) != 2:
            raise ValueError("L'image doit être une matrice 2D numpy")
        if image.min() < 0 or image.max() > 255:
            raise ValueError("Les valeurs doivent être entre 0 et 255")

    def rotation_90(self) -> np.ndarray:
        """Rotation de 90 degrés dans le sens horaire."""
        # Utilise la transposée et le flip pour une rotation efficace
        return np.flipud(self.image.T)

    def miroir_horizontal(self) -> np.ndarray:
        """Effet miroir horizontal."""
        return np.fliplr(self.image)

    def inverser(self) -> np.ndarray:
        """Inverse les couleurs (négatif)."""
        return 255 - self.image

    def appliquer_flou(self, taille_noyau: int = 3) -> np.ndarray:
        """Applique un flou (moyenne des voisins)."""
        if taille_noyau % 2 == 0:
            raise ValueError("La taille du noyau doit être impaire")
        
        resultat = np.zeros_like(self.image)
        padding = taille_noyau // 2
        
        # Ajoute du padding pour gérer les bords
        image_padded = np.pad(self.image, padding, mode='reflect')
        
        # Calcul de la moyenne pour chaque pixel
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                voisinage = image_padded[i:i+taille_noyau, j:j+taille_noyau]
                resultat[i, j] = np.mean(voisinage)
                
        return resultat

class TransformationsGeometriques:
    @staticmethod
    def rotation(angle_degres: float) -> np.ndarray:
        """Crée une matrice de rotation 2D."""
        angle = np.deg2rad(angle_degres)
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

    @staticmethod
    def translation(dx: float, dy: float) -> np.ndarray:
        """Crée une matrice de translation 2D."""
        return np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0,  1]
        ])

    @staticmethod
    def echelle(sx: float, sy: float) -> np.ndarray:
        """Crée une matrice de mise à l'échelle 2D."""
        return np.array([
            [sx,  0, 0],
            [ 0, sy, 0],
            [ 0,  0, 1]
        ])

    def combiner_transformations(self, transformations: List[np.ndarray]) -> np.ndarray:
        """Combine plusieurs matrices de transformation."""
        resultat = np.eye(3)  # Matrice identité 3x3
        for t in reversed(transformations):  # Dans l'ordre inverse pour la composition
            resultat = t @ resultat
        return resultat

    def transformer_points(self, points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
        """Applique une transformation à un ensemble de points."""
        # Conversion en coordonnées homogènes
        points_h = np.hstack([points, np.ones((len(points), 1))])
        
        # Application de la transformation
        points_transformes = points_h @ transformation.T
        
        # Retour aux coordonnées cartésiennes
        return points_transformes[:, :2] / points_transformes[:, 2:]

class ClassificateurImages:
    def __init__(self):
        self.caracteristiques_moyennes = {}
        self.classes = set()

    def extraire_caracteristiques(self, image: np.ndarray) -> np.ndarray:
        """Extrait les caractéristiques d'une image."""
        # Divise l'image en 4 zones et calcule la moyenne de chaque zone
        h, w = image.shape
        zones = [
            image[:h//2, :w//2],     # Haut gauche
            image[:h//2, w//2:],     # Haut droite
            image[h//2:, :w//2],     # Bas gauche
            image[h//2:, w//2:]      # Bas droite
        ]
        moyennes_zones = [np.mean(zone) for zone in zones]
        
        # Calcul de l'histogramme simplifié (4 bins)
        hist, _ = np.histogram(image, bins=4, range=(0, 256))
        hist = hist / hist.sum()  # Normalisation
        
        # Combine les caractéristiques
        return np.concatenate([moyennes_zones, hist])

    def entrainer(self, images: List[np.ndarray], labels: List[str]) -> None:
        """Entraîne le classificateur sur un ensemble d'images."""
        self.classes = set(labels)
        
        # Calcul des caractéristiques moyennes par classe
        for classe in self.classes:
            indices = [i for i, l in enumerate(labels) if l == classe]
            caracteristiques = np.array([
                self.extraire_caracteristiques(images[i]) 
                for i in indices
            ])
            self.caracteristiques_moyennes[classe] = np.mean(caracteristiques, axis=0)

    def predire(self, image: np.ndarray) -> str:
        """Prédit la classe d'une nouvelle image."""
        carac = self.extraire_caracteristiques(image)
        
        # Trouve la classe la plus proche (distance euclidienne)
        distances = {
            classe: np.linalg.norm(carac - moyenne)
            for classe, moyenne in self.caracteristiques_moyennes.items()
        }
        return min(distances.items(), key=lambda x: x[1])[0]

# Tests de validation
if __name__ == "__main__":
    # Test du traitement d'image
    image_test = np.random.randint(0, 256, (10, 10))
    traitement = TraitementImage(image_test)
    rotated = traitement.rotation_90()
    blurred = traitement.appliquer_flou()
    
    # Test des transformations géométriques
    trans = TransformationsGeometriques()
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    rot_45 = trans.rotation(45)
    trans_2_3 = trans.translation(2, 3)
    combined = trans.combiner_transformations([rot_45, trans_2_3])
    points_transformes = trans.transformer_points(points, combined)
    
    # Test du classificateur
    classif = ClassificateurImages()
    images_train = [np.random.randint(0, 256, (20, 20)) for _ in range(4)]
    labels_train = ["A", "A", "B", "B"]
    classif.entrainer(images_train, labels_train)
    prediction = classif.predire(np.random.randint(0, 256, (20, 20)))
    print(f"Classe prédite : {prediction}")