import pytest

# Exercice 1 : Calcul des points d'expérience
def calculer_xp(niveau, monstres_tues):
    """
    Calcule les points d'expérience gagnés.
    """
    if niveau < 1:
        raise ValueError("Le niveau doit être supérieur ou égal à 1.")
    if monstres_tues < 0:
        raise ValueError("Le nombre de monstres tués doit être positif.")
    return niveau * 100 + monstres_tues * 50

# Exercice 2 : Gestion de l'inventaire
class Inventaire:
    """
    Gère l'inventaire d'un joueur.
    """
    def __init__(self):
        self.items = []
        self.limite = 10

    def ajouter_item(self, item):
        if self.est_plein:
            raise ValueError("L'inventaire est plein.")
        self.items.append(item)

    def retirer_item(self, item):
        if item not in self.items:
            raise ValueError("L'item n'est pas dans l'inventaire.")
        self.items.remove(item)

    @property
    def est_plein(self):
        return len(self.items) >= self.limite

# Exercice 3 : Calcul des dégâts
def calculer_degats(attaquant, defenseur):
    """
    Calcule les dégâts d'une attaque.
    """
    force = attaquant.get("force", 0)
    defense = defenseur.get("defense", 0)
    bonus = attaquant.get("bonus", 0)
    critique = attaquant.get("critique", False)
    esquive = defenseur.get("esquive", False)

    if force < 0 or defense < 0:
        raise ValueError("Les valeurs de force et de défense doivent être positives.")
    
    degats = force - defense + bonus
    if critique:
        degats *= 2
    if esquive:
        degats = 0
    return max(degats, 0)

# Tests pour les exercices
def test_calcul_xp_base():
    assert calculer_xp(1, 0) == 100
    assert calculer_xp(2, 10) == 700

def test_calcul_xp_erreurs():
    with pytest.raises(ValueError):
        calculer_xp(0, 10)
    with pytest.raises(ValueError):
        calculer_xp(1, -1)

@pytest.fixture
def inventaire_vide():
    return Inventaire()

@pytest.fixture
def inventaire_plein():
    inv = Inventaire()
    for i in range(inv.limite):
        inv.ajouter_item(f"Item {i}")
    return inv

def test_ajouter_item(inventaire_vide):
    inventaire_vide.ajouter_item("épée")
    assert "épée" in inventaire_vide.items

def test_inventaire_plein(inventaire_plein):
    assert inventaire_plein.est_plein
    with pytest.raises(ValueError):
        inventaire_plein.ajouter_item("bouclier")

@pytest.mark.parametrize("force,defense,expected", [
    ({"force": 10, "bonus": 2, "critique": False}, {"defense": 5, "esquive": False}, 7),
    ({"force": 10, "bonus": 2, "critique": True}, {"defense": 5, "esquive": False}, 14),
    ({"force": 10, "bonus": 0, "critique": False}, {"defense": 15, "esquive": False}, 0),
    ({"force": 10, "bonus": 0, "critique": False}, {"defense": 5, "esquive": True}, 0),
    ({"force": 0, "bonus": 0, "critique": False}, {"defense": 0, "esquive": False}, 0),
])
def test_calcul_degats_parametres(attaquant, defenseur, expected):
    assert calculer_degats(attaquant, defenseur) == expected