def exercise_1():
    """
    Calculatrice basique
    """
    def addition(a, b):
        return a + b
    def soustraction(a, b):
        return a - b
    def multiplication(a, b):
        return a * b
    def division(a, b):
        if b == 0:
            return "Division par zéro impossible"
        return a / b
    print(addition(5, 3))
    print(soustraction(5, 3))
    print(multiplication(5, 3))
    print(division(5, 3))
    print(division(5, 0))

def exercise_2():
    """
    Gestionnaire de score
    """
    score = 0
    def add_points(points):
        nonlocal score
        score += points
        return score
    def new_game():
        return score == 0
    def level_up():
        nonlocal score
        score += 100
        return score
    print(add_points(10))
    print(add_points(20))
    print(new_game())
    print(level_up())
    print(new_game())

def exercise_3():
    """
    Analyseur de texte
    """
    def analyse_texte(texte):
        return {
            "nb_caracteres": len(texte),
            "nb_mots": len(texte.split()),
            "nb_phrases": texte.count(".") + texte.count("!") + texte.count("?"),
            "nb_paragraphes": texte.count("\n\n") + 1,
            "mot_le_plus_long": len(max(texte.split(), key=len))
        }
    print(analyse_texte("Python est un langage de programmation interprété, multi-paradigme et multiplateformes. Il favorise la programmation impérative structurée, fonctionnelle et orientée objet. Il est doté d'un typage dynamique fort, d'une gestion automatique de la mémoire par ramasse-miettes et d'un système de gestion d'exceptions ; il est ainsi similaire à Perl, Ruby, Scheme, Smalltalk et Tcl."))

if __name__ == "__main__":
    print("\n=== Exercise 1 : Calculatrice basique ===")
    exercise_1()
    
    print("\n=== Exercise 2 : Gestionnaire de score ===")
    exercise_2()
    
    print("\n=== Exercise 3 : Analyseur de texte ===")
    exercise_3()