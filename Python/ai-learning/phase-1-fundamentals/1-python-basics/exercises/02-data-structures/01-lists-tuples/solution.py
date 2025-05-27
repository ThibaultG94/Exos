"""
Solutions des exercices sur les listes et tuples
Author: Claude
Date: 2024-11-15
"""

def gerer_inventaire():
    """
    Solution de l'exercice 1 : Gestionnaire d'inventaire
    """
    # Création de l'inventaire initial
    inventaire = ["épée", "potion", "bouclier"]
    print(f"Inventaire initial : {inventaire}")
    
    # Ajout et insertion d'items
    inventaire.append("arc")
    print(f"Après ajout de l'arc : {inventaire}")
    
    inventaire.insert(1, "anneau")
    print(f"Après insertion de l'anneau : {inventaire}")
    
    # Suppression d'un item
    inventaire.remove("potion")
    
    # Affichage final
    print(f"Inventaire final ({len(inventaire)} items) : {inventaire}")

def analyser_scores():
    """
    Solution de l'exercice 2 : Analyse de scores
    """
    # Liste des scores
    scores = [85, 92, 78, 65, 95, 88]
    print(f"Scores : {scores}")
    
    # Analyse des scores
    score_max = max(scores)
    score_min = min(scores)
    moyenne = sum(scores) / len(scores)
    
    print(f"Score maximum : {score_max}")
    print(f"Score minimum : {score_min}")
    print(f"Moyenne : {moyenne:.2f}")
    
    # Tri et meilleurs scores
    scores.sort(reverse=True)
    print(f"Top 3 : {scores[:3]}")

def geometrie():
    """
    Solution de l'exercice 3 : Coordonnées et points
    """
    import math
    
    # Création des points
    point_a = (0, 0)
    point_b = (3, 4)
    point_c = (-2, 7)
    
    # Liste des points
    points = [point_a, point_b, point_c]
    
    # Affichage des coordonnées x
    for i, point in enumerate(['A', 'B', 'C']):
        print(f"Point {point} - x : {points[i][0]}")
    
    # Calcul de distance (bonus)
    def distance(p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    print(f"\nDistance AB : {distance(point_a, point_b):.2f}")
    print(f"Distance BC : {distance(point_b, point_c):.2f}")
    print(f"Distance AC : {distance(point_a, point_c):.2f}")

if __name__ == "__main__":
    print("\n=== Exercice 1 : Gestionnaire d'inventaire ===")
    gerer_inventaire()
    
    print("\n=== Exercice 2 : Analyse de scores ===")
    analyser_scores()
    
    print("\n=== Exercice 3 : Coordonnées et points ===")
    geometrie()