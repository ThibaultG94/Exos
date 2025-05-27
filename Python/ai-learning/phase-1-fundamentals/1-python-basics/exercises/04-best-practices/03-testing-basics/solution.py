"""
Solution des exercices sur les tests
Date: 2024-12-02
"""

import pytest
from typing import List, Any

def calculer_xp(niveau: int, monstres_tues: int) -> int:
    """
    Calcule les points d'expérience gagnés.
    
    Args:
        niveau: Niveau actuel du joueur
        monstres_tues: Nombre de monstres tués
        
    Returns:
        Points d'expérience calculés
        
    Raises:
        ValueError: Si niveau < 1 ou monstres_tues < 0
    """
    if not isinstance(niveau, int) or niveau < 1:
        raise ValueError("Le niveau doit être un entier positif")
    if not isinstance(monstres_tues, int) or monstres_tues < 0:
        raise ValueError("Le nombre de monstres doit être positif ou nul")
        
    return niveau * 100 + monstres_tues * 50

class Inventaire:
    """Gestion de l'inventaire d'un joueur"""
    
    def __init__(self, limite: int = 10):
        self.items: List[Any] = []
        self.limite = limite
    
    def ajouter_item(self, item: Any) -> None:
        """Ajoute un item si l'inventaire n'est pas plein"""
        if self.est_plein:
            raise ValueError("Inventaire plein")
        self.items.append(item)
    
    def retirer_item(self, item: Any) -> None:
        """Retire un item s'il existe"""
        try:
            self.items.remove(item)
        except ValueError:
            raise ValueError(f"Item {item} non trouvé dans l'inventaire")
    
    @property
    def est_plein(self) -> bool:
        """Vérifie si l'inventaire est plein"""
        return len(self.items) >= self.limite

def calculer_degats(attaquant: dict, defenseur: dict) -> int:
    """
    Calcule les dégâts d'une attaque.
    
    Args:
        attaquant: Dict avec force et bonus
        defenseur: Dict avec defense
    
    Returns:
        Dégâts calculés
    """
    degats_base = attaquant.get('force', 0) - defenseur.get('defense', 0)
    bonus = attaquant.get('bonus', 0)
    return max(0, degats_base + bonus)

# Tests
def test_calcul_xp_base():
    """Test du calcul d'XP normal"""
    assert calculer_xp(1, 0) == 100
    assert calculer_xp(2, 1) == 250
    assert calculer_xp(5, 10) == 1000

def test_calcul_xp_erreurs():
    """Test des cas d'erreur du calcul d'XP"""
    with pytest.raises(ValueError):
        calculer_xp(0, 1)
    with pytest.raises(ValueError):
        calculer_xp(1, -1)
    with pytest.raises(ValueError):
        calculer_xp("1", 1)

@pytest.fixture
def inventaire_vide():
    """Fixture : inventaire vide"""
    return Inventaire()

@pytest.fixture
def inventaire_plein():
    """Fixture : inventaire plein"""
    inv = Inventaire(2)
    inv.ajouter_item("épée")
    inv.ajouter_item("bouclier")
    return inv

def test_ajouter_item(inventaire_vide):
    """Test d'ajout d'items"""
    inv = inventaire_vide
    inv.ajouter_item("épée")
    assert len(inv.items) == 1
    assert "épée" in inv.items

def test_inventaire_plein(inventaire_plein):
    """Test avec inventaire plein"""
    with pytest.raises(ValueError):
        inventaire_plein.ajouter_item("potion")

@pytest.mark.parametrize("attaquant,defenseur,expected", [
    ({"force": 10}, {"defense": 5}, 5),
    ({"force": 5}, {"defense": 10}, 0),
    ({"force": 10, "bonus": 2}, {"defense": 5}, 7),
    ({"force": 0}, {"defense": 0}, 0),
])
def test_calcul_degats_parametres(attaquant, defenseur, expected):
    """Tests paramétrés du calcul de dégâts"""
    assert calculer_degats(attaquant, defenseur) == expected

if __name__ == "__main__":
    pytest.main(["-v"])