def exercise_1():
    """Pratiquer les opérations mathématiques"""
    nombre1 = 15
    nombre2 = 5
    print(f"Addition: {nombre1 + nombre2}")
    print(f"Soustraction: {nombre1 - nombre2}")
    print(f"Multiplication: {nombre1 * nombre2}")
    print(f"Division: {nombre1 / nombre2}")

def exercise_2():
    """Calcul de moyenne"""
    note1 = 15
    note2 = 12
    note3 = 18
    moyenne = (note1 + note2 + note3) / 3
    print(f"Moyenne: {moyenne}")

def exercise_3():
    """Gérer des calculs de prix"""
    prix_ht = 20
    quantite = 3
    taux_tva = 0.2
    print(f"Prix HT: {prix_ht * quantite}")
    print(f"Montant TVA: {prix_ht * quantite * taux_tva}")
    print(f"Prix TTC: {prix_ht * quantite * (1 + taux_tva)}")

if __name__ == "__main__":
    # Exécution de chaque exercice avec séparation claire
    print("\n=== Exercise 1 ===")
    exercise_1()
    
    print("\n=== Exercise 2 ===")
    exercise_2()
    
    print("\n=== Exercise 3 ===")
    exercise_3()
