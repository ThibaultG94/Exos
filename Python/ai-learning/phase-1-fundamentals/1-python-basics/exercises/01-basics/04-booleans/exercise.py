def exercise_1():
    """
    Comparaisons simples
    """
    x=5
    y=10
    print(f"x < y : {x<y}")
    print(f"x == y : {x==y}")
    print(f"x != y : {x!=y}")
    pass

def exercise_2():
    """
    Opérateurs logiques
    """
    age=20
    a_permis=True
    solde=1000
    peut_conduire=age>=18 and a_permis
    peut_louer=peut_conduire and solde>=500
    est_riche=solde>=5000
    offre_speciale=est_riche or age<=25
    print(f"Peut conduire : {peut_conduire}")
    print(f"Peut louer : {peut_louer}")
    print(f"Est riche : {est_riche}")
    print(f"Offre spéciale : {offre_speciale}")
    pass

def exercise_3():
    """
    Validation de données
    """
    email="user@example.com"
    mot_de_passe="12345"
    confirmation="12345"
    email_valide=email.count("@")==1 and email.count(".")==1
    password_valide=len(mot_de_passe)>=5
    confirmation_valide=mot_de_passe==confirmation
    tout_valide=email_valide and password_valide and confirmation_valide
    print(f"Email valide : {email_valide}")
    print(f"Mot de passe valide : {password_valide}")
    print(f"Confirmation valide : {confirmation_valide}")
    print(f"Tout valide : {tout_valide}")
    pass

if __name__ == "__main__":
    print("\n=== Exercise 1 : Comparaisons simples ===")
    exercise_1()
    
    print("\n=== Exercise 2 : Opérateurs logiques ===")
    exercise_2()
    
    print("\n=== Exercise 3 : Validation de données ===")
    exercise_3()