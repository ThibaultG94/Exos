def exercise_1():
    """
    Manipulation basique de strings
    """
    # Votre code ici
    phrase = "Python est un langage génial"
    print(f"La longueur de la phrase '{phrase}' est de : {len(phrase)} caractères")
    nouvelle_phrase = phrase.replace("génial", "super").upper()
    print(f"La nouvelle phrase est : {nouvelle_phrase}")
    pass

def exercise_2():
    """
    Extraction et formatage
    """
    # Votre code ici
    prenom = "Ada"
    nom = "Lovelace"
    print(f"{prenom[0]}. {nom}")
    print(f"{prenom.lower()}.{nom.lower()}@example.com")
    pass

def exercise_3():
    """
    Analyse de texte
    """
    # Votre code ici
    text = "Python est un langage de programmation orienté objet"
    print(f"Le nombre de a dans la phrase '{text}' est de : {text.count('a')}")
    print(f"La disposition du mot 'langage' dans la phrase est : {text.index('langage')}")
    split_text = text.split(" ")
    print(f"Le nombre de mots dans la phrase est de : {len(split_text)}")
    pass

if __name__ == "__main__":
    # Exécution des exercices
    print("\n=== Exercise 1 : Manipulation basique ===")
    exercise_1()
    
    print("\n=== Exercise 2 : Extraction et formatage ===")
    exercise_2()
    
    print("\n=== Exercise 3 : Analyse de texte ===")
    exercise_3()