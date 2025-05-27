import os
import sys
import subprocess
from pathlib import Path

def verifier_environnement():
    """Vérifie si nous sommes dans un environnement virtuel"""
    return sys.prefix != sys.base_prefix

def creer_structure_projet(nom_projet: str):
    """
    Crée la structure de base d'un projet
    
    Args:
        nom_projet: Nom du dossier projet à créer
    """
    projet = Path(nom_projet)
    (projet / "src").mkdir(parents=True, exist_ok=True)
    (projet / "tests").mkdir(exist_ok=True)
    (projet / "src" / "main.py").touch()
    (projet / "tests" / "test_main.py").touch()
    
    # Création du .gitignore
    with open(projet / ".gitignore", "w") as f:
        f.write("venv/\n__pycache__/\n*.pyc\n")
    
    print(f"Structure du projet {nom_projet} créée !")

def generer_requirements():
    """Génère différents fichiers requirements"""
    requirements = {
        "requirements-dev.txt": [
            "black",
            "pylint",
            "ipython",
            "debug-tools",
            "-r requirements-test.txt"
        ],
        "requirements-test.txt": [
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "-r requirements-prod.txt"
        ],
        "requirements-prod.txt": [
            "requests",
            "python-dotenv"
        ]
    }
    
    for fichier, deps in requirements.items():
        with open(fichier, "w") as f:
            f.write("\n".join(deps))
    
    print("Fichiers requirements générés !")

def gerer_compatibilite_python():
    """Gère la compatibilité entre versions Python"""
    versions = ["3.7", "3.9"]
    for version in versions:
        venv_dir = Path(f"venv_{version.replace('.', '')}")
        subprocess.run(["virtualenv", str(venv_dir), f"--python=python{version}"])
        subprocess.run([str(venv_dir / "bin" / "pip"), "install", "numpy", "pandas"])
        
        with open("compatibility_test.py", "w") as f:
            f.write("""
# Test script for Python compatibility
import sys
try:
    case_test = {
        "example": lambda: match case:
            case "example":
                return "Compatible"
    }
except SyntaxError:
    print("Python version does not support 'match-case'")
else:
    print("All features are supported!")
""")
        
        print(f"Environnement avec Python {version} configuré.")

def main():
    """Fonction principale"""
    if not verifier_environnement():
        print("⚠️ ATTENTION: Vous n'êtes pas dans un environnement virtuel !")
        return
    
    print("1. Création structure projet...")
    creer_structure_projet("mon_projet")
    
    print("\n2. Génération requirements...")
    generer_requirements()
    
    print("\n3. Gestion compatibilité Python...")
    gerer_compatibilite_python()
    
    print("\nConfiguration terminée ! N'oubliez pas d'activer votre environnement virtuel.")

if __name__ == "__main__":
    main()
