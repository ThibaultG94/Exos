def exercise_1():
    """
    Calculatrice basique
    """
    def additionner(a, b):
        return a + b
    
    def soustraire(a, b):
        return a - b
    
    def multiplier(a, b):
        return a * b
    
    def diviser(a, b):
        if b == 0:
            return "Erreur: Division par zéro"
        return a / b
    
    # Tests avec affichage formaté
    print("Tests calculatrice :")
    print(f"5 + 3 = {additionner(5, 3)}")
    print(f"5 - 3 = {soustraire(5, 3)}")
    print(f"5 × 3 = {multiplier(5, 3)}")
    print(f"5 ÷ 3 = {diviser(5, 3):.2f}")
    print(f"5 ÷ 0 = {diviser(5, 0)}")

def exercise_2():
    """
    Gestionnaire de score
    """
    def nouveau_jeu():
        return 0
    
    def ajouter_points(score, points):
        return score + points
    
    def niveau_joueur(score):
        if score < 100:
            return "Débutant"
        elif score < 200:
            return "Intermédiaire"
        return "Expert"
    
    # Scénario de test
    score = nouveau_jeu()
    print(f"Nouveau jeu: {score} points - {niveau_joueur(score)}")
    
    score = ajouter_points(score, 50)
    print(f"Après 50 points: {score} points - {niveau_joueur(score)}")
    
    score = ajouter_points(score, 100)
    print(f"Après 100 points: {score} points - {niveau_joueur(score)}")

def exercise_3():
    """
    Analyseur de texte
    """
    def analyser_texte(texte):
        nb_caracteres = len(texte)
        mots = texte.split()
        nb_mots = len(mots)
        nb_phrases = sum(texte.count(sep) for sep in ".!?")
        mot_plus_long = max(mots, key=len)
        
        return (nb_caracteres, nb_mots, nb_phrases, mot_plus_long)
    
    texte = "Python est un langage génial! Il est facile à apprendre. Vous aimez programmer?"
    car, mots, phrases, long_mot = analyser_texte(texte)
    
    print(f"Caractères: {car}")
    print(f"Mots: {mots}")
    print(f"Phrases: {phrases}")
    print(f"Mot le plus long: {long_mot}")

if __name__ == "__main__":
    print("\n=== Exercise 1 : Calculatrice basique ===")
    exercise_1()
    
    print("\n=== Exercise 2 : Gestionnaire de score ===")
    exercise_2()
    
    print("\n=== Exercise 3 : Analyseur de texte ===")
    exercise_3()