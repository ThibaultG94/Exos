"""
Solutions des exercices sur la manipulation de fichiers
Author: Claude
Date: 2024-11-28
"""
from datetime import datetime
import os

def ajouter_note(texte):
    """Ajoute une note avec la date et l'heure"""
    horodatage = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("notes.txt", "a", encoding="utf-8") as f:
        f.write(f"[{horodatage}] {texte}\n")

def lire_notes():
    """Lit et affiche toutes les notes"""
    if not os.path.exists("notes.txt"):
        print("Aucune note trouvée")
        return

    with open("notes.txt", "r", encoding="utf-8") as f:
        for i, ligne in enumerate(f, 1):
            print(f"Note {i}: {ligne.strip()}")

def analyser_texte(nom_fichier):
    """Analyse un fichier texte"""
    try:
        with open(nom_fichier, "r", encoding="utf-8") as f:
            texte = f.read().lower()
            
        # Nettoyage basique
        for p in ".,;!?":
            texte = texte.replace(p, "")
            
        mots = texte.split()
        lignes = texte.splitlines()
        
        # Statistiques
        nb_mots = len(mots)
        nb_lignes = len(lignes)
        longueur_moyenne = sum(len(mot) for mot in mots) / nb_mots
        
        # Mots fréquents
        frequence = {}
        for mot in mots:
            frequence[mot] = frequence.get(mot, 0) + 1
        
        top_mots = sorted(frequence.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Affichage
        print(f"Nombre de mots : {nb_mots}")
        print(f"Nombre de lignes : {nb_lignes}")
        print(f"Longueur moyenne des mots : {longueur_moyenne:.1f}")
        print("Mots les plus fréquents :")
        for mot, freq in top_mots:
            print(f"  {mot}: {freq} fois")
            
    except FileNotFoundError:
        print(f"Fichier {nom_fichier} non trouvé")

def ajouter_score(joueur, score):
    """Ajoute ou met à jour un score"""
    scores = {}
    
    # Lecture des scores existants
    if os.path.exists("scores.txt"):
        with open("scores.txt", "r", encoding="utf-8") as f:
            for ligne in f:
                nom, points = ligne.strip().split(",")
                scores[nom] = int(points)
    
    # Ajout/mise à jour du score
    scores[joueur] = score
    
    # Sauvegarde triée
    with open("scores.txt", "w", encoding="utf-8") as f:
        for nom, points in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{nom},{points}\n")

def voir_top_scores(n=5):
    """Affiche les n meilleurs scores"""
    if not os.path.exists("scores.txt"):
        print("Aucun score enregistré")
        return
        
    print(f"Top {n} scores :")
    with open("scores.txt", "r", encoding="utf-8") as f:
        for i, ligne in enumerate(f, 1):
            if i > n:
                break
            nom, score = ligne.strip().split(",")
            print(f"{i}. {nom}: {score} points")

if __name__ == "__main__":
    # Tests
    try:
        # Test des notes
        print("=== Test des notes ===")
        ajouter_note("Première note de test")
        ajouter_note("Deuxième note de test")
        lire_notes()

        # Test de l'analyseur
        print("\n=== Test de l'analyseur ===")
        # Création d'un fichier exemple
        with open("exemple.txt", "w", encoding="utf-8") as f:
            f.write("Voici un exemple de texte.\nIl contient plusieurs lignes.\nCertains mots sont répétés répétés.")
        analyser_texte("exemple.txt")

        # Test des scores
        print("\n=== Test des scores ===")
        ajouter_score("Alice", 100)
        ajouter_score("Bob", 85)
        voir_top_scores()

    except Exception as e:
        print(f"Erreur : {e}")