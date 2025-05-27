"""
Exercices sur la manipulation de fichiers
"""
from datetime import datetime
from collections import Counter
import os

def ajouter_note(texte):
    """Ajoute une note avec la date et l'heure"""
    with open('notes.txt', 'a') as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{timestamp} : {texte}\n")
    print(f"Note ajoutée : {texte}")

def lire_notes():
    """Lit et affiche toutes les notes"""
    if not os.path.exists('notes.txt'):
        print("Aucune note trouvée")
        return
    with open('notes.txt', 'r') as file:
        notes = file.readlines()
        for i, note in enumerate(notes, 1):
            print(f"{i}. {note.strip()}")

def analyser_texte(nom_fichier):
    """Analyse un fichier texte"""
    if not os.path.exists(nom_fichier):
        print(f"Le fichier {nom_fichier} n'existe pas")
        return
    with open(nom_fichier, 'r') as file:
        contenu = file.read()
    
    # Supprimer la ponctuation et mettre en minuscule
    mots = ''.join(c.lower() if c.isalnum() else ' ' for c in contenu).split()
    compteur = Counter(mots)

    nb_mots = len(mots)
    nb_lignes = contenu.count('\n') + 1
    mots_freq = compteur.most_common(5)
    longueur_moyenne = sum(len(mot) for mot in mots) / nb_mots if nb_mots > 0 else 0

    print(f"Nombre total de mots : {nb_mots}")
    print(f"Nombre de lignes : {nb_lignes}")
    print(f"5 mots les plus fréquents : {mots_freq}")
    print(f"Longueur moyenne des mots : {longueur_moyenne:.2f}")

def ajouter_score(joueur, score):
    """Ajoute ou met à jour un score"""
    scores = []
    if os.path.exists('scores.txt'):
        with open('scores.txt', 'r') as file:
            for line in file:
                nom, pts = line.strip().split(',')
                scores.append((nom, int(pts)))
    else:
        print("Aucun score trouvé")
    
    # Mettre à jour ou ajouter le score
    trouve = False
    for i, (nom, pts) in enumerate(scores):
        if nom == joueur:
            scores[i] = (joueur, max(score, pts)) # Garde le score le plus élevé
            trouve = True
            break
        if not trouve:
            scores.append((joueur, score))

        # Trier les scores
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # Écrire les scores mis à jour
        with open('scores.txt', 'w') as file:
            for nom, pts in scores:
                file.write(f"{nom},{pts}\n")
        print(f"Score ajouté/mis à jour pour {joueur} : {score}")

def voir_top_scores(n=5):
    """Affiche les n meilleurs scores"""
    if not os.path.exists('scores.txt'):
        print("Aucun score trouvé")
        return
    with open('scores.txt', 'r') as file:
        scores = [line.strip().split(',') for line in file]
        scores = [(nom, int(pts)) for nom, pts in scores]
    
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print(f"Top {n} scores :")
    for i, (nom, pts) in enumerate(scores[:n], 1):
        print(f"{i}. {nom} : {pts}")

if __name__ == "__main__":
    # Tests
    try:
        # Test des notes
        ajouter_note("Première note de test")
        ajouter_note("Deuxième note de test")
        lire_notes()

        # Test de l'analyseur
        analyser_texte("exemple.txt")
        analyser_texte("notes.txt")

        # Test des scores
        ajouter_score("Alice", 100)
        ajouter_score("Bob", 85)
        voir_top_scores()

    except Exception as e:
        print(f"Erreur : {e}")