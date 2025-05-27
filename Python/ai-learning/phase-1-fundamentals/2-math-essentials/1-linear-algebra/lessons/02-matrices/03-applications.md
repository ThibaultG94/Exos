# Les matrices en action : du jeu au Machine Learning ! 🚀

## Donnons du sens aux matrices !

Nous avons vu les opérations matricielles, mais à quoi servent-elles concrètement ? Plongeons dans des applications réelles, particulièrement en Machine Learning.

### 1. Les images numériques

Une image en niveaux de gris est une matrice ! Chaque nombre représente l'intensité lumineuse d'un pixel :

```python
import numpy as np

# Une petite image 4x4 (0 = noir, 255 = blanc)
smiley = np.array([
    [0,   255, 255,   0],
    [255,   0,   0, 255],
    [255, 255, 255, 255],
    [0,   255, 255,   0]
])

# Application d'un flou (moyenne avec les voisins)
def flouter_image(image):
    resultat = np.zeros_like(image)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            # Moyenne du pixel et ses 8 voisins
            voisinage = image[i-1:i+2, j-1:j+2]
            resultat[i,j] = np.mean(voisinage)
    return resultat

smiley_floute = flouter_image(smiley)
```

### 2. Les jeux de données en ML

En ML, nos données sont toujours organisées en matrices :

```python
# Base de données de maisons
# Lignes = maisons, Colonnes = caractéristiques
donnees = np.array([
    #  surface  chambres  age  prix
    [   120,      3,     10,  250000],
    [    90,      2,     15,  180000],
    [   150,      4,      5,  350000]
])

# Séparation caractéristiques / cibles
X = donnees[:, :-1]  # Toutes les colonnes sauf la dernière
y = donnees[:, -1]   # Juste la dernière colonne (prix)

# Normalisation des données
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
```

### 3. Les réseaux de neurones

Un réseau de neurones n'est qu'une série de multiplications matricielles :

```python
def couche_neuronale(entrees, poids, biais):
    """Simule une couche de neurones."""
    # entrees : matrice (n_exemples, n_caracteristiques)
    # poids : matrice (n_caracteristiques, n_neurones)
    # biais : vecteur (n_neurones)

    # Multiplication matricielle + biais
    activation = np.dot(entrees, poids) + biais

    # Fonction d'activation ReLU
    return np.maximum(0, activation)

# Exemple simple
entrees = np.array([[1, 2, 3]])  # Un exemple avec 3 caractéristiques
poids = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
])  # 3 entrées -> 2 neurones
biais = np.array([0.1, 0.1])

sortie = couche_neuronale(entrees, poids, biais)
```

### 4. La compression d'images

La décomposition en valeurs singulières (SVD) permet de compresser des images :

```python
def compresser_image(image, n_composantes):
    """Compresse une image en gardant les n composantes principales."""
    # Décomposition SVD
    U, s, Vt = np.linalg.svd(image)

    # Reconstruction avec moins de composantes
    reconstruction = np.dot(
        U[:, :n_composantes],
        np.dot(np.diag(s[:n_composantes]), Vt[:n_composantes, :])
    )

    return reconstruction

# Application sur notre image
image_compressee = compresser_image(smiley, 2)
taux_compression = (1 - 2*len(smiley)/len(smiley)**2) * 100
print(f"Taux de compression : {taux_compression:.1f}%")
```

### 5. Recommandation de films

Les systèmes de recommandation utilisent aussi les matrices :

```python
# Matrice utilisateurs-films (1-5 étoiles, 0 = pas vu)
evaluations = np.array([
    #  Film1  Film2  Film3
    [    5,     3,     0  ],  # Alice
    [    4,     0,     5  ],  # Bob
    [    0,     4,     4  ]   # Charlie
])

def trouver_films_similaires(film_id):
    """Trouve les films similaires basés sur les évaluations."""
    film_evals = evaluations[:, film_id]

    # Calcul de similarité avec tous les autres films
    similarites = []
    for autre_film in range(evaluations.shape[1]):
        if autre_film != film_id:
            autres_evals = evaluations[:, autre_film]
            # Ne considérer que les utilisateurs ayant vu les deux films
            masque = (film_evals != 0) & (autres_evals != 0)
            if np.sum(masque) > 0:
                correlation = np.corrcoef(
                    film_evals[masque],
                    autres_evals[masque]
                )[0,1]
                similarites.append((autre_film, correlation))

    return sorted(similarites, key=lambda x: x[1], reverse=True)
```

### 6. Application complète : Détection d'anomalies

```python
class DetecteurAnomalies:
    def __init__(self, donnees_normales):
        """Initialise avec des données normales."""
        self.moyenne = np.mean(donnees_normales, axis=0)
        self.covariance = np.cov(donnees_normales.T)
        self.inv_covariance = np.linalg.inv(self.covariance)

    def calculer_distance(self, echantillon):
        """Calcule la distance de Mahalanobis."""
        diff = echantillon - self.moyenne
        return np.sqrt(
            diff.T @ self.inv_covariance @ diff
        )

    def est_anomalie(self, echantillon, seuil=3.0):
        """Détecte si un échantillon est anormal."""
        return self.calculer_distance(echantillon) > seuil

# Utilisation
donnees_normales = np.random.multivariate_normal(
    mean=[0, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=1000
)

detecteur = DetecteurAnomalies(donnees_normales)
point_test = np.array([5, 5])
print(f"Anomalie détectée : {detecteur.est_anomalie(point_test)}")
```

### 🎯 Points clés à retenir

1. Les matrices sont fondamentales en traitement d'image
2. Le ML repose sur des opérations matricielles
3. La décomposition matricielle permet la compression
4. Les systèmes de recommandation utilisent des matrices
5. La détection d'anomalies utilise des matrices de covariance

### 🚀 Pour aller plus loin

- Explorez les bibliothèques comme scikit-learn
- Approfondissez l'algèbre linéaire
- Pratiquez avec des projets concrets
- Visualisez les transformations matricielles
