def exercise_1():
    """
    Gestionnaire de contacts
    """
    # Création des contacts
    contacts = {
        "Alice": {"email": "alice@mail.com", "tel": "0123456789"},
        "Bob": {"email": "bob@mail.com", "tel": "0234567890"},
        "Charlie": {"email": "charlie@mail.com", "tel": "0345678901"}
    }
    contacts["David"] = {"email": "david@mail.com", "tel": "0456789012"}
    contacts["Bob"]["tel"] = "0234567891"
    del contacts["Charlie"]
    print(f"Mails: {', '.join([contact['email'] for contact in contacts.values()])}")

def exercise_2():
    """
    Stats de texte
    """
    texte = """
    Python est un langage de programmation puissant et Python est facile à apprendre.
    Python est utilisé dans le web, la data science et même les jeux.
    """
    occurence_mots = {}
    mots = texte.split()
    for mot in mots:
        mot = mot.lower().strip(".,")
        if mot in occurence_mots:
            occurence_mots[mot] += 1
        else:
            occurence_mots[mot] = 1
    mot_le_plus_frequent = max(occurence_mots, key=occurence_mots.get)
    ensemble_mots_uniques = set(mots)
    print(f"Nombre de mots: {len(mots)}")
    print(f"Nombre de mots uniques: {len(ensemble_mots_uniques)}")
    print(f"Mot le plus fréquent et son compte: {mot_le_plus_frequent} ({occurence_mots[mot_le_plus_frequent]})")

def exercise_3():
    """
    Comparaison d'ensembles
    """
    langages_web = {"HTML", "CSS", "JavaScript", "PHP"}
    langages_prog = {"Python", "JavaScript", "Java", "C++"}
    langages_communs_et_diff = langages_web.intersection(langages_prog)
    langages_uniques = langages_web.symmetric_difference(langages_prog)
    print(f"Langages communs: {', '.join(langages_communs_et_diff)}")
    isPythonInLangagesWeb = "Python" in langages_web
    print(f"Python dans langages_web: {isPythonInLangagesWeb}")

if __name__ == "__main__":
    print("\n=== Exercise 1 : Gestionnaire de contacts ===")
    exercise_1()
    
    print("\n=== Exercise 2 : Stats de texte ===")
    exercise_2()
    
    print("\n=== Exercise 3 : Comparaison d'ensembles ===")
    exercise_3()