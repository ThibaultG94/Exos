# Exercices : Les tests en Python 🧪

## Niveau 1 : Tests de base

### Exercice 1 : Premiers tests unitaires

```python
"""
Objectif : Tester une fonction de calcul de points d'expérience

1. Créez et testez une fonction calculer_xp(niveau, monstres_tues) qui :
   - Retourne niveau * 100 + monstres_tues * 50
   - Lève une ValueError si niveau < 1
   - Lève une ValueError si monstres_tues < 0

2. Écrivez les tests pour vérifier :
   - Le calcul normal fonctionne
   - Les cas limites (niveau 1, 0 monstres)
   - Les erreurs sont bien levées
   - Les types sont corrects
"""
```

### Exercice 2 : Fixtures et contexte

```python
"""
Objectif : Tester une classe Inventaire avec fixtures

1. Créez une classe Inventaire avec :
   - Limite de 10 items
   - Méthodes ajouter_item() et retirer_item()
   - Propriété est_plein

2. Écrivez des tests avec fixtures pour :
   - Un inventaire vide
   - Un inventaire avec quelques items
   - Un inventaire plein

3. Testez tous les cas :
   - Ajout normal
   - Ajout quand plein
   - Retrait existant
   - Retrait inexistant
"""
```

### Exercice 3 : Tests paramétrés

```python
"""
Objectif : Tester le système de combat avec paramètres

1. Créez une fonction calculer_degats(attaquant, defenseur) qui :
   - Utilise force, defense et bonus
   - Retourne les dégâts finaux
   - Gère les cas spéciaux (critique, esquive)

2. Écrivez des tests paramétrés pour :
   - Différentes combinaisons de stats
   - Cas limites (0 force, 0 defense)
   - Cas spéciaux (bonus, malus)
   - Valeurs invalides

3. Utilisez @pytest.mark.parametrize avec :
   - Au moins 5 cas de test
   - Des cas d'erreur
   - Des cas limites
"""
```

## Points à évaluer

- Organisation des tests
- Couverture des cas
- Utilisation des assertions
- Gestion des erreurs
- Documentation des tests

## Bonus

- Ajoutez des tests de performance
- Utilisez des mocks pour simuler le RNG
- Testez les entrées/sorties fichier
- Mesurez la couverture de code
