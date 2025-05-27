def exercise_1():
    """
    Transformations de base avec des list comprehensions simples
    """
    # Partie 1 : Opérations sur les nombres
    nombres = range(10)
    
    # Création des différentes listes avec explications
    carres = [x**2 for x in nombres]
    print("1.1 Carrés des nombres de 0 à 9 :", carres)
    
    nombres_pairs = [x for x in nombres if x % 2 == 0]
    print("1.2 Nombres pairs :", nombres_pairs)
    
    multiples_trois = [x for x in nombres if x % 3 == 0]
    print("1.3 Multiples de 3 :", multiples_trois)
    
    # Partie 2 : Manipulation de texte
    mots = ["Python", "est", "UN", "langage", "SUPER"]
    
    # Transformations de texte avec explications
    minuscules = [mot.lower() for mot in mots]
    print("\n2.1 Mots en minuscules :", minuscules)
    
    longueurs = [(mot, len(mot)) for mot in mots]  # Ajout du mot pour plus de clarté
    print("2.2 Longueur des mots :", longueurs)
    
    longs_mots = [mot for mot in mots if len(mot) > 3]
    print("2.3 Mots de plus de 3 lettres :", longs_mots)

def exercise_2():
    """
    Filtrage et transformation avec conditions plus complexes
    """
    notes = [("math", 15), ("physique", 12), ("histoire", 18), ("sport", 8)]
    
    # Extraction et filtrage
    bonnes_notes = [(matiere, note) for matiere, note in notes if note >= 15]
    print("1. Matières avec note ≥ 15 :", bonnes_notes)
    
    # Function helper pour la lisibilité
    def obtenir_mention(note):
        if note >= 16: return "Très bien"
        if note >= 14: return "Bien"
        if note >= 12: return "Assez bien"
        return "Passable"
    
    # Création des mentions
    mentions = [(matiere, obtenir_mention(note)) for matiere, note in notes]
    print("2. Mentions par matière :", mentions)
    
    # Filtrage avec condition mathématique
    matieres_notes_paires = [matiere.upper() for matiere, note in notes if note % 2 == 0]
    print("3. Matières avec notes paires :", matieres_notes_paires)

def exercise_3():
    """
    Compréhensions imbriquées et manipulations avancées
    """
    # Partie 1 : Coordonnées
    coordonnees = [(x, y) for x in range(3) for y in range(3)]
    print("1.1 Toutes les coordonnées :", coordonnees)
    
    points_sup = [(x, y) for x, y in coordonnees if x > y]
    print("1.2 Points où x > y :", points_sup)
    
    # Partie 2 : Matrice
    matrice = [[x * y for y in range(3)] for x in range(3)]
    print("\n2.1 Matrice :")
    for ligne in matrice:
        print(f"     {ligne}")
    
    valeurs_positives = [val for ligne in matrice for val in ligne if val > 0]
    print("2.2 Valeurs positives :", valeurs_positives)
    
    # Partie 3 : Traitement de texte
    phrases = [
        "Python est génial",
        "Les list comprehensions sont utiles",
        "Coder est amusant"
    ]
    
    # Découpage en mots avec structure
    structure_mots = [phrase.split() for phrase in phrases]
    print("\n3.1 Structure des phrases :")
    for i, mots in enumerate(structure_mots, 1):
        print(f"     Phrase {i} : {mots}")
    
    # Extraction des mots longs
    mots_longs = [mot for phrase in phrases 
                  for mot in phrase.split() 
                  if len(mot) > 4]
    print("3.2 Mots de plus de 4 lettres :", mots_longs)

if __name__ == "__main__":
    print("\n=== Exercice 1 : Transformations de base ===")
    exercise_1()
    
    print("\n=== Exercice 2 : Filtrage et transformation ===")
    exercise_2()
    
    print("\n=== Exercice 3 : Compréhensions imbriquées ===")
    exercise_3()