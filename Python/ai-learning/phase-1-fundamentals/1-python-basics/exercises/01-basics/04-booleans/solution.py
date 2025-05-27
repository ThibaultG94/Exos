"""
Solution des exercices sur les booléens
Author: Claude
Date: 2024-11-15
"""

def exercise_1():
    """
    Comparaisons simples
    """
    # Création des variables de base
    x = 5
    y = 10
    
    # Création des booléens de comparaison
    est_plus_petit = x < y
    est_egal = x == y
    est_different = x != y
    
    # Affichage formaté des résultats
    print(f"{x} < {y} : {est_plus_petit}")
    print(f"{x} == {y} : {est_egal}")
    print(f"{x} != {y} : {est_different}")

def exercise_2():
    """
    Opérateurs logiques
    """
    # Données initiales
    age = 20
    a_permis = True
    solde = 1000
    
    # Tests logiques combinés
    peut_conduire = age >= 18 and a_permis
    peut_louer = peut_conduire and solde >= 500
    est_riche = solde >= 5000
    offre_speciale = est_riche or age <= 25
    
    # Affichage des résultats
    print(f"Peut conduire : {peut_conduire}")
    print(f"Peut louer : {peut_louer}")
    print(f"Est riche : {est_riche}")
    print(f"Éligible à l'offre spéciale : {offre_speciale}")

def exercise_3():
    """
    Validation de données
    """
    # Données à valider
    email = "user@example.com"
    mot_de_passe = "12345"
    confirmation = "12345"
    
    # Validations avec conditions combinées
    email_valide = "@" in email and "." in email
    password_valide = len(mot_de_passe) >= 5
    confirmation_valide = mot_de_passe == confirmation
    tout_valide = email_valide and password_valide and confirmation_valide
    
    # Affichage des résultats
    print(f"Email valide : {email_valide}")
    print(f"Mot de passe valide : {password_valide}")
    print(f"Confirmation valide : {confirmation_valide}")
    print(f"Tout valide : {tout_valide}")

if __name__ == "__main__":
    print("\n=== Exercise 1 : Comparaisons simples ===")
    exercise_1()
    
    print("\n=== Exercise 2 : Opérateurs logiques ===")
    exercise_2()
    
    print("\n=== Exercise 3 : Validation de données ===")
    exercise_3()