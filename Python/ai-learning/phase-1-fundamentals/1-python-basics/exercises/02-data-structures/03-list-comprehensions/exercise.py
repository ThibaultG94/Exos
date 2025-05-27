def exercise_1():
    """
    Transformations de base
    """
    # Liste de nombres
    nombres = range(10)
    # TODO : Créez les différentes transformations
    
    carres = [x**2 for x in nombres]
    print("Carrés :", carres)
    paires = [x for x in nombres if x % 2 == 0]
    print("Paires :", paires)
    divisibles_par_3 = [x for x in nombres if x % 3 == 0]
    print("Divisibles par 3 :", divisibles_par_3)
    
    
    # Liste de mots
    mots = ["Python", "est", "UN", "langage", "SUPER"]
    # TODO : Créez les différentes transformations
    mots_minuscules = [x.lower() for x in mots]
    print("Minuscules :", mots_minuscules)
    longueurs = [len(x) for x in mots]
    print("Longueurs :", longueurs)
    mots_longueur_sup_3 = [x for x in mots if len(x) > 3]
    print("Longueur > 3 :", mots_longueur_sup_3)

def exercise_2():
    """
    Filtrage et transformation
    """
    notes = [("math", 15), ("physique", 12), ("histoire", 18), ("sport", 8)]
    # TODO : Créez les différentes transformations
    liste_notes = [x[1] for x in notes]
    print("Notes :", liste_notes)

    notes_sup_15 = [x for x in notes if x[1] > 15]
    print("Notes > 15 :", notes_sup_15)

    mentions = [(x[0], "Très bien" if x[1] >= 16 else "Bien" if x[1] >= 14 else "Assez bien" if x[1] >= 12 else "Passable") for x in notes]
    print("Mentions :", mentions)

    matieres = [x[0].upper() for x in notes if x[1] % 2 == 0]
    print("Matières :", matieres)

def exercise_3():
    """
    Compréhensions imbriquées
    """
    # Coordonnées
    # TODO : Créez les coordonnées (x,y)

    coordonnees = [(x, y) for x in range(2) for y in range(2)]
    print("Coordonnées :", coordonnees)

    coordonnees_filtrees = [(x, y) for x, y in coordonnees if x > y]
    print("Coordonnées filtrées :", coordonnees_filtrees)
    
    # Matrice
    # TODO : Créez et transformez la matrice

    matrice = [[x*y for y in range(3)] for x in range(3)]
    print("Matrice :", matrice)

    valeurs = [x for ligne in matrice for x in ligne if x > 0]
    print("Valeurs > 0 :", valeurs)
    
    # Traitement de phrases
    phrases = [
        "Python est génial",
        "Les list comprehensions sont utiles",
        "Coder est amusant"
    ]
    # TODO : Traitez les phrases

    mots = [phrase.split() for phrase in phrases]
    print("Mots :", mots)

    mots_longueur_sup_4 = [mot for phrase in phrases for mot in phrase.split() if len(mot) > 4]
    print("Mots longueur > 4 :", mots_longueur_sup_4)

if __name__ == "__main__":
    print("\n=== Exercice 1 : Transformations de base ===")
    exercise_1()
    
    print("\n=== Exercice 2 : Filtrage et transformation ===")
    exercise_2()
    
    print("\n=== Exercice 3 : Compréhensions imbriquées ===")
    exercise_3()