"""
Solution des exercices sur les chaînes de caractères
Author: Claude
Date: 2024-11-06
"""

def exercise_1():
    """
    Manipulation basique de strings
    """
    # Création et analyse de la phrase
    phrase = "Python est un langage génial"
    longueur = len(phrase)
    print(f"Phrase originale : '{phrase}'")
    print(f"Longueur : {longueur} caractères")
    
    # Transformations
    phrase_maj = phrase.upper()
    phrase_finale = phrase_maj.replace("GÉNIAL", "SUPER")
    
    print(f"Version finale : '{phrase_finale}'")

def exercise_2():
    """
    Extraction et formatage
    """
    # Définition des noms
    prenom = "Ada"
    nom = "Lovelace"
    
    # Création du format citation
    initiale = prenom[0]
    citation = f"{initiale}. {nom}"
    print(f"Citation : {citation}")
    
    # Création de l'email
    email = f"{prenom.lower()}.{nom.lower()}@example.com"
    print(f"Email : {email}")

def exercise_3():
    """
    Analyse de texte
    """
    texte = "Python est un langage de programmation orienté objet"
    
    # Analyse des caractères
    nb_a = texte.count('a')
    pos_langage = texte.find('langage')  # find retourne -1 si non trouvé
    print(f"Nombre de 'a' : {nb_a}")
    print(f"'langage' commence à la position : {pos_langage}")
    
    # Analyse des mots
    mots = texte.split()
    print(f"La phrase contient {len(mots)} mots :")
    for i, mot in enumerate(mots, 1):
        print(f"  Mot {i} : {mot}")

if __name__ == "__main__":
    print("\n=== Exercise 1 : Manipulation basique ===")
    exercise_1()
    
    print("\n=== Exercise 2 : Extraction et formatage ===")
    exercise_2()
    
    print("\n=== Exercise 3 : Analyse de texte ===")
    exercise_3()