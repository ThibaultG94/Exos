def exercise_1():
    """
    Gestionnaire d'inventaire
    """
    inventory = ["sword", "potion", "shield", "bow"]
    inventory.append("arrow")
    inventory.insert(0, "ring")
    inventory.remove("potion")
    print(f"Inventory: {inventory} - Length: {len(inventory)}")

def exercise_2():
    """
    Analyse de scores
    """
    scores = [85, 92, 78, 65, 95, 88]
    score_max = max(scores)
    score_min = min(scores)
    score_average = sum(scores) / len(scores)
    print(f"Max: {score_max} - Min: {score_min} - Average: {score_average}")
    sorted_scores = sorted(scores)
    print(f"The thee best scores are: {sorted_scores[-3:]}")

def exercise_3():
    """
    Coordonnées et points
    """
    A = (0, 0)
    B = (3, 4)
    C = (-2, 7)
    list_points = [A, B, C]
    coordonates = [f"Point {i+1} : {point}" for i, point in enumerate(list_points)]
    print("\n".join(coordonates))
    distances = [((A[0] - point[0])**2 + (A[1] - point[1])**2)**0.5 for point in list_points]
    print(f"Distances from A: {distances}")

if __name__ == "__main__":
    print("\n=== Exercise 1 : Gestionnaire d'inventaire ===")
    exercise_1()
    
    print("\n=== Exercise 2 : Analyse de scores ===")
    exercise_2()
    
    print("\n=== Exercise 3 : Coordonnées et points ===")
    exercise_3()