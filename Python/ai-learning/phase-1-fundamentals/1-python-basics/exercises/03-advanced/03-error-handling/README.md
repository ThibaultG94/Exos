# Exercices : La gestion des erreurs en Python 🛠️

## Exercice 1 : Validation d'entrées

```python
"""
Objectif : Créer des validateurs robustes pour les données du jeu

1. Créez une fonction creer_personnage(nom, niveau, points_de_vie) qui :
   - Vérifie que nom est une chaîne non vide
   - Vérifie que niveau est entre 1 et 100
   - Vérifie que points_de_vie est positif
   - Lève des exceptions appropriées si les conditions ne sont pas remplies
   - Retourne un objet Personnage si tout est valide

2. Créez une fonction ajouter_item(inventaire, nom, type_item) qui :
   - Vérifie que l'inventaire n'est pas plein (max 10 items)
   - Vérifie que le type_item est valide (arme/potion/armure)
   - Gère les cas d'erreur avec des messages explicites
"""
```

## Exercice 2 : Gestion de ressources

```python
"""
Objectif : Implémenter la sauvegarde/chargement sécurisé

1. Créez un gestionnaire de contexte (context manager) SauvegardePartie qui :
   - Ouvre un fichier de sauvegarde en mode écriture
   - Gère les erreurs d'écriture/lecture
   - Assure la fermeture du fichier même en cas d'erreur
   - Permet d'utiliser le with statement

2. Implémentez les fonctions :
   - sauvegarder_personnage(personnage)
   - charger_personnage(nom_fichier)
   Qui utilisent ce gestionnaire de contexte

3. Gérez tous les cas d'erreur possibles :
   - Fichier introuvable
   - Erreur de format JSON
   - Données corrompues
"""
```

## Exercice 3 : Exceptions personnalisées

```python
"""
Objectif : Créer une hiérarchie d'exceptions pour le jeu

1. Créez les classes d'exception suivantes :
   - GameError (classe de base)
   - InventaireError
   - PersonnageError
   - CombatError

2. Modifiez le système de combat pour utiliser ces exceptions :
   - AttaqueImpossibleError (personnage mort)
   - CibleInvalidError (cible hors de portée)
   - ManaInsuffisantError (pour les sorts)

3. Implémentez un système de logs qui :
   - Enregistre toutes les exceptions dans un fichier
   - Inclut la date/heure et la trace complète
   - Permet différents niveaux de détail (debug/info/error)
"""
```

## Points à évaluer

- Utilisation appropriée de try/except
- Exceptions spécifiques et bien nommées
- Messages d'erreur clairs et utiles
- Gestion propre des ressources
- Logging des erreurs
- Tests des cas limites

## Bonus

- Ajoutez des tests unitaires pour les cas d'erreur
- Implémentez un système de retry pour les opérations risquées
- Créez un décorateur pour le logging automatique des exceptions
- Ajoutez des assertions pour le débogage
