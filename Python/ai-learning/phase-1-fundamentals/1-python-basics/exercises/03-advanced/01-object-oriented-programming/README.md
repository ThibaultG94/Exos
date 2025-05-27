# Exercices : Programmation Orientée Objet en Python 🎮

## Niveau 1 : Les bases des classes et objets

### Exercice 1 : Création d'une classe Personnage

```python
"""
Objectif : Créer votre première classe représentant un personnage de jeu

1. Créez une classe Personnage avec :
   - Attributs : nom, points_de_vie, niveau (initialisé à 1)
   - Méthode : afficher_statut() qui affiche les infos du personnage
   - Méthode : est_vivant() qui renvoie True si points_de_vie > 0

2. Créez deux personnages et testez les méthodes
3. Faites combattre les personnages (perdre des points de vie)
4. Vérifiez leur statut après le combat
"""
```

### Exercice 2 : Gestion d'un inventaire

```python
"""
Objectif : Gérer l'inventaire d'un joueur

1. Créez une classe Item avec :
   - Attributs : nom, type (arme/potion/armure), valeur
   - Méthode : decrire() qui renvoie la description de l'item

2. Créez une classe Inventaire avec :
   - Attribut : items (liste)
   - Méthodes :
     * ajouter_item(item)
     * retirer_item(nom_item)
     * calculer_valeur_totale()
     * lister_items()

3. Testez avec plusieurs items
"""
```

### Exercice 3 : Système de combat

```python
"""
Objectif : Créer un système de combat basique

1. Créez une classe Combattant qui hérite de Personnage avec :
   - Attributs supplémentaires : force, defense
   - Méthodes :
     * attaquer(cible)
     * defendre()
     * subir_degats(montant)

2. Créez une classe Guerrier qui hérite de Combattant :
   - Capacité spéciale : coup_puissant(cible)

3. Créez une classe Mage qui hérite de Combattant :
   - Attribut supplémentaire : mana
   - Capacité spéciale : boule_de_feu(cible)

4. Faites combattre un Guerrier contre un Mage
"""
```

## Points clés à évaluer

- Utilisation correcte de l'héritage
- Encapsulation appropriée
- Méthodes bien définies
- Gestion des erreurs
- Code propre et commenté

## Bonus

- Ajoutez des stats aléatoires aux personnages
- Créez un système d'expérience et de niveau
- Ajoutez des effets spéciaux aux attaques
- Implémentez un système d'équipement
