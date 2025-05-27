"""
Solution des exercices sur les variables
Author: Claude
Date: 2024-11-06
"""

def exercise_1():
    """Démonstration de la création de variables basiques"""
    age = 30
    year = 2024
    name = "Thibault"
    print(f"{name} is {age} years old in {year}.")

def exercise_2():
    """Démonstration de la modification de score"""
    score = 0
    print(f"Score: {score}")
    score += 10  # Augmentation du score de 10
    print(f"Score: {score}")
    score += 5   # Augmentation du score de 5
    print(f"Score: {score}")

def exercise_3():
    """Simulation d'un mini jeu avec points et vies"""
    points = 0
    life = 3
    print("Début du jeu !")
    print(f'Points: {points} | Vies: {life}')
    points += 10  # Gain de points
    print(f'Points: {points} | Vies: {life}')
    life -= 1    # Perte d'une vie
    print(f'Points: {points} | Vies: {life}')

if __name__ == "__main__":
    # Exécution de chaque exercice avec séparation claire
    print("\n=== Exercise 1 ===")
    exercise_1()
    
    print("\n=== Exercise 2 ===")
    exercise_2()
    
    print("\n=== Exercise 3 ===")
    exercise_3()