def exercise_1():
    """
    Classification de notes
    """
    note = 15
    
    if note >= 18:
        mention = "Excellent"
    elif note >= 16:
        mention = "Très bien"
    elif note >= 14:
        mention = "Bien"
    elif note >= 12:
        mention = "Assez bien"
    elif note >= 10:
        mention = "Passable"
    else:
        mention = "Insuffisant"
    
    print(f"Note : {note}/20 - Mention : {mention}")

def exercise_2():
    """
    Compteur intelligent
    """
    for count in range(1, 21):
        if count % 3 == 0 and count % 5 == 0:
            print(f"{count}: FizzBuzz")
        elif count % 3 == 0:
            print(f"{count}: Fizz")
        elif count % 5 == 0:
            print(f"{count}: Buzz")
        else:
            print(count)

def exercise_3():
    """
    Tables de multiplication
    """
    for n in range(1, 4):
        print(f"\nTable de {n} :")
        for m in range(1, 11):
            print(f"{n:2d} × {m:2d} = {n*m:2d}")

if __name__ == "__main__":
    print("\n=== Exercise 1 : Classification de notes ===")
    exercise_1()
    
    print("\n=== Exercise 2 : Compteur intelligent ===")
    exercise_2()
    
    print("\n=== Exercise 3 : Tables de multiplication ===")
    exercise_3()