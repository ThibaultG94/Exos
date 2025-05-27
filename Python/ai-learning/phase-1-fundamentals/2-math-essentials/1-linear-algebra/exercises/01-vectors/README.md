# Exercices : Vecteurs et Calculs Vectoriels 🎯

## Objectif

Mettre en pratique les concepts de base des vecteurs en créant un mini-système de jeu de rôle avec des personnages qui se déplacent, combattent et interagissent dans un espace 2D.

## Exercice 1 : Création d'un personnage

```python
"""
Objectif : Créer une classe Personnage avec gestion vectorielle des positions et mouvements

1. Attributs à implémenter :
   - position : vecteur numpy 2D (x, y)
   - direction : vecteur numpy 2D normalisé (la direction regardée)
   - vitesse : scalaire (unités par seconde)
   - points_de_vie : entier

2. Méthodes à implémenter :
   - se_deplacer(delta_temps) : met à jour la position selon la direction et la vitesse
   - tourner(angle_degres) : modifie la direction du personnage
   - distance_vers(autre_personnage) : retourne la distance avec un autre personnage
   - regarder_vers(position_cible) : oriente le personnage vers un point
"""
```

## Exercice 2 : Système de Combat

```python
"""
Objectif : Implémenter un système de combat basé sur les vecteurs

1. Créer une classe GestionnaireCombat avec :
   - portee_attaque : distance maximale d'attaque
   - angle_vision : angle maximal de vision (en degrés)

2. Méthodes à implémenter :
   - peut_attaquer(attaquant, cible) -> bool :
     * La cible doit être à portée
     * La cible doit être dans le cône de vision
   - calculer_degats(attaquant, cible) -> float :
     * Base de dégâts * (1 - distance/portee_max)
     * Bonus si attaque de dos (+50%)

3. Bonus :
   - Gestion des esquives basée sur l'angle d'attaque
   - Parade si la cible regarde l'attaquant
"""
```

## Exercice 3 : Analyse de Trajectoire

```python
"""
Objectif : Créer un système d'analyse de mouvement pour le ML

1. Implémenter une classe AnalyseurMouvement qui :
   - Enregistre l'historique des positions d'un personnage
   - Calcule des statistiques sur le mouvement :
     * Vitesse moyenne
     * Accélération
     * Distance totale parcourue
     * Zones les plus visitées

2. Méthodes principales :
   - enregistrer_position(position, temps)
   - calculer_statistiques()
   - predire_position_future(delta_temps)
   - identifier_zones_frequentes()

3. Utiliser ces données pour :
   - Détecter des patterns de mouvement
   - Prédire les déplacements futurs
   - Identifier les zones stratégiques
"""
```

## Critères d'évaluation

- Utilisation correcte de numpy pour les calculs vectoriels
- Gestion des cas limites (division par zéro, distances nulles...)
- Clarté et documentation du code
- Optimisation des calculs vectoriels
- Tests avec différents scénarios de jeu

## Points bonus

- Visualisation des trajectoires et zones de combat
- Gestion avancée des collisions
- Système de prédiction de mouvements
- Optimisation des performances pour beaucoup d'entités
