def exercise_1():
    """
    Gestionnaire de contacts avec affichage détaillé des opérations
    """
    # Création des contacts initiaux
    contacts = {
        "Alice": {"email": "alice@mail.com", "tel": "0123456789"},
        "Bob": {"email": "bob@mail.com", "tel": "0234567890"},
        "Charlie": {"email": "charlie@mail.com", "tel": "0345678901"}
    }
    print("Contacts initiaux :", contacts)

    # Ajout d'un contact
    contacts["David"] = {"email": "david@mail.com", "tel": "0456789012"}
    print("Après ajout de David :", contacts["David"])

    # Modification d'un contact
    ancien_tel = contacts["Bob"]["tel"]
    contacts["Bob"]["tel"] = "0234567891"
    print(f"Modification du tel de Bob : {ancien_tel} → {contacts['Bob']['tel']}")

    # Suppression d'un contact
    del contacts["Charlie"]
    print("Contacts après suppression de Charlie :", list(contacts.keys()))

    # Affichage des emails avec formatage
    emails = [f"{nom}: {infos['email']}" for nom, infos in contacts.items()]
    print("Liste des emails :\n" + "\n".join(emails))

def exercise_2():
    """
    Analyseur de texte avec statistiques détaillées
    """
    texte = """
    Python est un langage de programmation puissant et Python est facile à apprendre.
    Python est utilisé dans le web, la data science et même les jeux.
    """
    # Nettoyage et comptage des mots
    mots = [mot.lower().strip(".,") for mot in texte.split()]
    occurences = {}
    
    for mot in mots:
        occurences[mot] = occurences.get(mot, 0) + 1
    
    # Analyse des statistiques
    mot_frequent = max(occurences.items(), key=lambda x: x[1])
    mots_uniques = set(mots)
    
    # Affichage formaté
    print("Statistiques du texte :")
    print(f"- Nombre total de mots : {len(mots)}")
    print(f"- Mots uniques : {len(mots_uniques)}")
    print(f"- Mot le plus fréquent : '{mot_frequent[0]}' ({mot_frequent[1]} fois)")
    print("\nTop 3 des mots :")
    for mot, compte in sorted(occurences.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"- {mot}: {compte} fois")

def exercise_3():
    """
    Analyse comparative des langages de programmation
    """
    langages_web = {"HTML", "CSS", "JavaScript", "PHP"}
    langages_prog = {"Python", "JavaScript", "Java", "C++"}

    # Analyses des ensembles
    communs = langages_web & langages_prog
    specifiques_web = langages_web - langages_prog
    specifiques_prog = langages_prog - langages_web
    tous_langages = langages_web | langages_prog

    # Affichage structuré
    print("Analyse des langages :")
    print(f"- Communs : {', '.join(communs)}")
    print(f"- Spécifiques au web : {', '.join(specifiques_web)}")
    print(f"- Spécifiques à la prog : {', '.join(specifiques_prog)}")
    print(f"- Total des langages : {len(tous_langages)}")

    # Vérifications d'appartenance
    langages_test = ["Python", "Ruby", "JavaScript"]
    for langage in langages_test:
        print(f"\nPrésence de {langage} :")
        print(f"- Dans langages web : {'✓' if langage in langages_web else '✗'}")
        print(f"- Dans langages prog : {'✓' if langage in langages_prog else '✗'}")

if __name__ == "__main__":
    print("\n=== Exercise 1 : Gestionnaire de contacts ===")
    exercise_1()
    
    print("\n=== Exercise 2 : Stats de texte ===")
    exercise_2()
    
    print("\n=== Exercise 3 : Comparaison d'ensembles ===")
    exercise_3()