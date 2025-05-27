# Les tests en Python : Les bases 🧪

## Pourquoi tester son code ?

Imaginez que vous développez un jeu :

- Comment être sûr que les dégâts sont bien calculés ?
- Comment vérifier que le système de points fonctionne ?
- Comment s'assurer qu'un changement ne casse rien ?

Les tests sont là pour ça !

## 1. Les bases avec pytest

### Premier test simple

```python
# test_calcul.py
def calculer_degats(force, defense):
    return max(0, force - defense)

def test_calcul_degats():
    # Test basique
    assert calculer_degats(10, 5) == 5
    # Test avec défense supérieure
    assert calculer_degats(5, 10) == 0
```

### Lancer les tests

```bash
# Installation de pytest
pip install pytest

# Lancer les tests
pytest test_calcul.py
```

## 2. Fixtures : préparer le terrain

```python
import pytest

@pytest.fixture
def joueur_test():
    """Crée un joueur pour les tests"""
    return {
        "nom": "Test",
        "points_de_vie": 100,
        "force": 15
    }

def test_attaque(joueur_test):
    assert joueur_test["points_de_vie"] == 100
    assert joueur_test["force"] == 15
```

## 3. Tests paramétrés

```python
@pytest.mark.parametrize("force,defense,attendu", [
    (10, 5, 5),    # Cas normal
    (5, 10, 0),    # Défense supérieure
    (0, 0, 0),     # Valeurs nulles
    (100, 50, 50)  # Grandes valeurs
])
def test_calcul_degats_parametres(force, defense, attendu):
    assert calculer_degats(force, defense) == attendu
```

## 4. Organisation des tests

```
mon_projet/
    ├── src/
    │   └── jeu/
    │       ├── __init__.py
    │       ├── combat.py
    │       └── joueur.py
    └── tests/
        ├── __init__.py
        ├── test_combat.py
        └── test_joueur.py
```

## 5. Les assertions utiles

```python
def test_diverses_assertions():
    # Égalité
    assert resultat == 42

    # Type
    assert isinstance(joueur, Joueur)

    # Exceptions
    with pytest.raises(ValueError):
        creer_joueur("")  # Devrait lever une erreur

    # Approximation (pour les floats)
    assert 0.1 + 0.2 == pytest.approx(0.3)

    # Contenu
    assert "erreur" in message
```

## 6. Tests de classes

```python
class TestJoueur:
    def test_creation(self):
        joueur = Joueur("Test")
        assert joueur.nom == "Test"
        assert joueur.points_de_vie == 100

    def test_subir_degats(self):
        joueur = Joueur("Test")
        joueur.subir_degats(20)
        assert joueur.points_de_vie == 80
```

## 7. Simuler des dépendances (Mocking)

```python
from unittest.mock import Mock, patch

def test_sauvegarde_joueur(monkeypatch):
    # Simuler une fonction de sauvegarde
    mock_save = Mock()
    monkeypatch.setattr("jeu.joueur.sauvegarder", mock_save)

    # Test
    joueur = Joueur("Test")
    joueur.sauvegarder()

    # Vérification
    mock_save.assert_called_once_with(joueur)
```

## Points clés à retenir

1. Tests simples avec assert
2. Fixtures pour la réutilisation
3. Tests paramétrés pour les cas multiples
4. Organisation claire des tests
5. Mock pour isoler les tests

## 🎯 Bonnes pratiques

1. Un test = une fonctionnalité
2. Noms de tests explicites
3. Tests indépendants
4. Couverture de code raisonnable
5. Tests automatisés (CI/CD)

## ⚠️ Pièges courants

1. Tester trop de choses à la fois
2. Dépendances entre tests
3. Tests trop complexes
4. Oublier les cas limites

## 🔄 Workflow des tests

1. Écrire les tests d'abord (TDD)
2. Implémenter la fonctionnalité
3. Vérifier la couverture
4. Refactoriser si nécessaire
