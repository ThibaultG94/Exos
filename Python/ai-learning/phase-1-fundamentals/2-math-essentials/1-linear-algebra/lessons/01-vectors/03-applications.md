# Les vecteurs en action : du jeu au Machine Learning ! 🚀

## Passons à la pratique !

Nous avons vu la théorie et les opérations, mais à quoi ça sert concrètement ? Voyons comment les vecteurs sont utilisés dans le monde réel, particulièrement en Machine Learning.

### 1. Les vecteurs de caractéristiques

En ML, chaque exemple est représenté par un vecteur. Prenons un exemple concret : la prédiction du prix d'une maison.

```python
import numpy as np

# Une maison = un vecteur de caractéristiques
maison = np.array([
    120,    # surface (m²)
    3,      # nombre de chambres
    2,      # nombre de salles de bain
    10,     # âge (années)
    1,      # garage (oui/non)
    800     # distance du centre-ville (m)
])
```

🤔 **Point de réflexion** : Pourquoi utiliser des vecteurs ? Parce qu'ils nous permettent de calculer facilement des similitudes et des distances entre les exemples !

### 2. Calcul de similarité

Comment savoir si deux maisons sont similaires ? La similarité cosinus est souvent utilisée en ML :

```python
def similarite_cosinus(v1, v2):
    """
    Calcule la similarité entre deux vecteurs.
    Retourne un score entre -1 et 1 (1 = très similaire)
    """
    produit_scalaire = np.dot(v1, v2)
    norme_produit = np.linalg.norm(v1) * np.linalg.norm(v2)
    return produit_scalaire / norme_produit

# Comparons deux maisons
maison1 = np.array([120, 3, 2, 10, 1, 800])
maison2 = np.array([125, 3, 2, 12, 1, 850])
score = similarite_cosinus(maison1, maison2)
print(f"Score de similarité : {score:.2f}")  # Proche de 1 = très similaire
```

### 3. Représentation de texte (Word Embeddings)

En traitement du langage naturel, on transforme les mots en vecteurs :

```python
# Version simplifiée d'embeddings
mots_vecteurs = {
    "roi": np.array([0.2, 0.5, 0.1]),
    "reine": np.array([0.2, 0.5, -0.1]),
    "homme": np.array([0.1, 0.2, 0.1]),
    "femme": np.array([0.1, 0.2, -0.1])
}

# Les relations sont préservées !
difference_genre = mots_vecteurs["roi"] - mots_vecteurs["reine"]
print("Différence roi-reine ≈ différence homme-femme")
```

### 4. Classification d'images

Une image peut être vue comme un grand vecteur de pixels :

```python
def charger_image_nb(chemin, taille=(28, 28)):
    """Charge une image en niveaux de gris comme un vecteur"""
    # En pratique, on utiliserait PIL ou cv2
    image = np.random.randint(0, 256, size=taille)
    return image.flatten()  # Transforme la matrice 28x28 en vecteur 784

# Une image devient un vecteur
image_vecteur = charger_image_nb("chat.jpg")
print(f"Dimension du vecteur : {len(image_vecteur)}")  # 784 dimensions
```

### 5. Détection d'anomalies

Les vecteurs nous aident à détecter ce qui est "normal" ou "anormal" :

```python
class DetecteurAnomalies:
    def __init__(self, donnees_normales):
        self.moyenne = np.mean(donnees_normales, axis=0)
        self.seuil = self.calculer_seuil(donnees_normales)

    def calculer_seuil(self, donnees):
        """Calcule un seuil basé sur les distances à la moyenne"""
        distances = [np.linalg.norm(x - self.moyenne) for x in donnees]
        return np.mean(distances) + 2 * np.std(distances)

    def est_anomalie(self, echantillon):
        distance = np.linalg.norm(echantillon - self.moyenne)
        return distance > self.seuil

# Exemple avec des données de capteurs
donnees_normales = np.random.normal(size=(1000, 3))  # 1000 mesures de 3 capteurs
detecteur = DetecteurAnomalies(donnees_normales)

mesure_suspecte = np.array([10, -5, 8])  # Valeurs inhabituelles
print(f"Anomalie détectée : {detecteur.est_anomalie(mesure_suspecte)}")
```

### 6. Application complète : Recommandation de films

Créons un système de recommandation simple basé sur les vecteurs :

```python
class SystemeRecommandation:
    def __init__(self):
        # Vecteurs de caractéristiques des films
        # [action, comédie, romance, fantasy, drame]
        self.films = {
            "Star Wars": np.array([0.9, 0.2, 0.3, 0.9, 0.3]),
            "Titanic": np.array([0.3, 0.0, 0.9, 0.0, 0.9]),
            "La La Land": np.array([0.1, 0.4, 0.9, 0.2, 0.7]),
            "Matrix": np.array([0.9, 0.2, 0.2, 0.9, 0.4])
        }

    def recommander(self, film_reference, n=2):
        """Trouve les n films les plus similaires"""
        if film_reference not in self.films:
            return []

        v_reference = self.films[film_reference]
        scores = []

        for titre, vecteur in self.films.items():
            if titre != film_reference:
                sim = similarite_cosinus(v_reference, vecteur)
                scores.append((titre, sim))

        # Trie par similarité décroissante
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

# Utilisation
recommandeur = SystemeRecommandation()
recommandations = recommandeur.recommander("Star Wars")
for film, score in recommandations:
    print(f"Recommandé : {film} (score : {score:.2f})")
```

### 🎯 Points clés à retenir

1. Les vecteurs permettent de représenter des données complexes
2. La similarité entre vecteurs mesure la ressemblance entre exemples
3. En ML, tout devient vecteur : textes, images, sons...
4. Les opérations sur les vecteurs ont du sens dans le monde réel

### 🚀 Pour aller plus loin

- Explorez les bibliothèques de ML comme scikit-learn
- Essayez de créer vos propres représentations vectorielles
- Expérimentez avec d'autres mesures de similarité
- Combinez les vecteurs avec des algorithmes de clustering
