"""
Solution complète des exercices sur les environnements virtuels
Author: Claude
Date: 2024-11-29
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

class GestionnaireEnvironnement:
    """Gestion des environnements virtuels Python"""
    
    def __init__(self, nom_projet: str):
        self.nom_projet = nom_projet
        self.chemin_projet = Path(nom_projet)
        self.est_windows = platform.system() == "Windows"
    
    def creer_structure(self):
        """Crée la structure complète du projet"""
        # Création des dossiers
        for dossier in ["src", "tests", "docs"]:
            (self.chemin_projet / dossier).mkdir(parents=True, exist_ok=True)
        
        # Création des fichiers de base
        fichiers = {
            "src/main.py": "def main():\n    print('Hello, World!')",
            "tests/test_main.py": "def test_main():\n    assert True",
            ".gitignore": "venv/\n__pycache__/\n*.pyc\n.pytest_cache/",
            "README.md": f"# {self.nom_projet}\n\nDescription du projet..."
        }
        
        for chemin, contenu in fichiers.items():
            with open(self.chemin_projet / chemin, "w") as f:
                f.write(contenu)
    
    def creer_environnement(self, version_python: str = ""):
        """Crée un environnement virtuel"""
        python = f"python{version_python}" if version_python else "python"
        commande = f"{python} -m venv venv"
        subprocess.run(commande, shell=True, check=True)
    
    def generer_requirements(self):
        """Génère les différents fichiers requirements"""
        requirements = {
            "requirements-dev.txt": [
                "black==22.3.0",
                "pylint==2.15.0",
                "ipython==8.4.0",
                "-r requirements-test.txt"
            ],
            "requirements-test.txt": [
                "pytest==7.1.2",
                "pytest-cov==3.0.0",
                "pytest-mock==3.8.2",
                "-r requirements-prod.txt"
            ],
            "requirements-prod.txt": [
                "requests==2.28.1",
                "python-dotenv==0.20.0"
            ]
        }
        
        for fichier, deps in requirements.items():
            with open(self.chemin_projet / fichier, "w") as f:
                f.write("\n".join(deps))
    
    def creer_scripts_activation(self):
        """Crée des scripts d'activation pour différents shells"""
        scripts = {
            "activate.sh": """
#!/bin/bash
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
""",
            "activate.bat": """
@echo off
call venv\\Scripts\\activate
set PYTHONPATH=%PYTHONPATH%;%CD%\\src
"""
        }
        
        for nom, contenu in scripts.items():
            with open(self.chemin_projet / nom, "w") as f:
                f.write(contenu.strip())
            
            # Rend le script shell exécutable sur Unix
            if not self.est_windows and nom.endswith(".sh"):
                path = self.chemin_projet / nom
                path.chmod(path.stat().st_mode | 0o111)

def main():
    """Démontre l'utilisation complète"""
    try:
        # Création et configuration du projet
        gestionnaire = GestionnaireEnvironnement("mon_projet")
        
        print("1. Création de la structure du projet...")
        gestionnaire.creer_structure()
        
        print("2. Création de l'environnement virtuel...")
        gestionnaire.creer_environnement()
        
        print("3. Génération des requirements...")
        gestionnaire.generer_requirements()
        
        print("4. Création des scripts d'activation...")
        gestionnaire.creer_scripts_activation()
        
        print("\nConfiguration terminée avec succès !")
        print("Pour commencer, exécutez :")
        if gestionnaire.est_windows:
            print("  .\\activate.bat")
        else:
            print("  source ./activate.sh")
            
    except Exception as e:
        print(f"Erreur lors de la configuration : {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()