"""
Solution des exercices sur les nombres
Author: Claude
Date: 2024-11-06
"""

def exercise_1():
    """Pratiquer les opérations mathématiques"""
    nombre1 = 15
    nombre2 = 5
    
    # Calculs avec formatage plus détaillé
    print(f"{nombre1} + {nombre2} = {nombre1 + nombre2}")
    print(f"{nombre1} - {nombre2} = {nombre1 - nombre2}")
    print(f"{nombre1} × {nombre2} = {nombre1 * nombre2}")
    print(f"{nombre1} ÷ {nombre2} = {nombre1 / nombre2}")

def exercise_2():
    """Calcul de moyenne"""
    note1 = 15
    note2 = 12
    note3 = 18
    
    # Calcul avec variables intermédiaires pour plus de clarté
    somme = note1 + note2 + note3
    moyenne = somme / 3
    
    # Affichage détaillé
    print(f"Note 1 : {note1}")
    print(f"Note 2 : {note2}")
    print(f"Note 3 : {note3}")
    print(f"Moyenne : {moyenne:.2f}")  # Arrondi à 2 décimales

def exercise_3():
    """Gérer des calculs de prix"""
    prix_ht = 20
    quantite = 3
    taux_tva = 0.2  # 20%
    
    # Calculs intermédiaires pour plus de clarté
    total_ht = prix_ht * quantite
    montant_tva = total_ht * taux_tva
    total_ttc = total_ht + montant_tva
    
    # Affichage formaté avec le symbole €
    print(f"Total HT : {total_ht:.2f} €")
    print(f"Montant TVA : {montant_tva:.2f} €")
    print(f"Total TTC : {total_ttc:.2f} €")

if __name__ == "__main__":
    # Exécution de chaque exercice avec séparation claire
    print("\n=== Exercise 1 : Opérations de base ===")
    exercise_1()
    
    print("\n=== Exercise 2 : Calcul de moyenne ===")
    exercise_2()
    
    print("\n=== Exercise 3 : Mini caisse enregistreuse ===")
    exercise_3()