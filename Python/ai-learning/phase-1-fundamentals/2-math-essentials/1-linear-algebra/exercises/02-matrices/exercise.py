import numpy as np
from typing import List, Tuple, Optional

class TraitementImage:
    def __init__(self, image: np.ndarray):
        """
        Initialise le système de traitement avec une image en niveaux de gris.
        L'image doit être une matrice 2D avec des valeurs entre 0 et 255.
        """
        self.verifier_image(image)
        self.image = image.astype(float)
    
    def verifier_image(self, image: np.ndarray) -> None:
        """Vérifie que l'image est valide."""
        if not isinstance(image, np.ndarray) or len(image.shape) != 2:
            raise ValueError("L'image doit être une matrice 2D numpy")
        if image.min() < 0 or image.max() > 255:
            raise ValueError("Les valeurs doivent être entre 0 et 255")

    def rotation_90(self) -> np.ndarray:
        """Rotation de 90 degrés dans le sens horaire."""
        return np.rot90(self.image, k=-1)

    def miroir_horizontal(self) -> np.ndarray:
        """Effet miroir horizontal."""
        return np.flip(self.image, axis=1)

    def inverser(self) -> np.ndarray:
        """Inverse les couleurs (négatif)."""
        return 255 - self.image

    def appliquer_flou(self, taille_noyau: int = 3) -> np.ndarray:
        """Applique un flou (moyenne des voisins)."""
        if taille_noyau % 2 == 0:
            raise ValueError("La taille du noyau doit être impaire")
        pad = taille_noyau // 2
        padded_image = np.pad(self.image, pad_width=pad, mode='constant', constant_values=0)
        resultat = np.zeros_like(self.image)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                voisinage = padded_image[i:i + taille_noyau, j:j + taille_noyau]
                resultat[i, j] = np.mean(voisinage)
        return resultat


class TransformationsGeometriques:
    @staticmethod
    def rotation(angle_degres: float) -> np.ndarray:
        """Crée une matrice de rotation 2D."""
        angle_radians = np.deg2rad(angle_degres)
        return np.array([
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians),  np.cos(angle_radians)]
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
            [sx, 0],
            [0, sy]
        ])

    def combiner_transformations(self, transformations: List[np.ndarray]) -> np.ndarray:
        """Combine plusieurs matrices de transformation."""
        resultat = transformations[0]
        for transfo in transformations[1:]:
            resultat = np.dot(resultat, transfo)
        return resultat

    def transformer_points(self, points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
        """Applique une transformation à un ensemble de points."""
        if transformation.shape[1] == 2:
            points = np.hstack((points, np.ones((points.shape[0], 1))))
        return np.dot(points, transformation.T)[:, :2]


class ClassificateurImages:
    def __init__(self):
        """Initialise le classificateur."""
        self.caracteristiques_moyennes = {}
        self.classes = set()

    def extraire_caracteristiques(self, image: np.ndarray) -> np.ndarray:
        """Extrait les caractéristiques d'une image."""
        return np.array([image.mean(), image.std()])

    def entrainer(self, images: List[np.ndarray], labels: List[str]) -> None:
        """Entraîne le classificateur sur un ensemble d'images."""
        for image, label in zip(images, labels):
            self.classes.add(label)
            if label not in self.caracteristiques_moyennes:
                self.caracteristiques_moyennes[label] = []
            self.caracteristiques_moyennes[label].append(self.extraire_caracteristiques(image))
        for label in self.caracteristiques_moyennes:
            self.caracteristiques_moyennes[label] = np.mean(self.caracteristiques_moyennes[label], axis=0)

    def predire(self, image: np.ndarray) -> str:
        """Prédit la classe d'une nouvelle image."""
        caracteristiques = self.extraire_caracteristiques(image)
        distances = {
            label: np.linalg.norm(caracteristiques - moyenne)
            for label, moyenne in self.caracteristiques_moyennes.items()
        }
        return min(distances, key=distances.get)

# Tests
if __name__ == "__main__":
    image_test = np.array([[100, 150], [200, 250]], dtype=float)
    traitement = TraitementImage(image_test)
    print("Rotation 90°:\n", traitement.rotation_90())
    print("Miroir horizontal:\n", traitement.miroir_horizontal())
    print("Inversion des couleurs:\n", traitement.inverser())
    print("Flou:\n", traitement.appliquer_flou(3))

    transformations = TransformationsGeometriques()
    print("Matrice rotation:\n", transformations.rotation(45))
    print("Matrice translation:\n", transformations.translation(5, 5))
    print("Matrice échelle:\n", transformations.echelle(2, 2))

    classificateur = ClassificateurImages()
    images = [np.array([[50, 50], [50, 50]]), np.array([[200, 200], [200, 200]])]
    labels = ["clair", "foncé"]
    classificateur.entrainer(images, labels)
    print("Prédiction classe:\n", classificateur.predire(np.array([[60, 60], [60, 60]])))
