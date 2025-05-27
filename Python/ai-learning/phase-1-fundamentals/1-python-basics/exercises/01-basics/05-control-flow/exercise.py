def exercise_1():
    """
    Classification de notes
    """
    note = 15
    if note >= 18:
        print("Excellent")
    elif note >= 16:
        print("Très bien")
    elif note >= 14:
        print("Bien")
    elif note >= 12:
        print("Assez bien")
    elif note >= 10:
        print("Passable")
    else:
        print("Insuffisant")
    pass

def exercise_2():
    """
    Compteur intelligent
    """
    count = 1
    while count <= 20:
        if count % 3 == 0 and count % 5 == 0:
            print("FizzBuzz")
        elif count % 3 == 0:
            print("Fizz")
        elif count % 5 == 0:
            print("Buzz")
        else:
            print(count)
        count += 1
    pass

def exercise_3():
    """
    Tables de multiplication
    """
    for n in range(1, 3):
        for m in range(1, 10):
            print(f"{n} x {m} = {n*m}")
        print()
    pass

if __name__ == "__main__":
    print("\n=== Exercise 1 : Classification de notes ===")
    exercise_1()
    
    print("\n=== Exercise 2 : Compteur intelligent ===")
    exercise_2()
    
    print("\n=== Exercise 3 : Tables de multiplication ===")
    exercise_3()