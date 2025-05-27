# Guide d'Installation de StarCoder

Ce guide détaille le processus d'installation et de configuration de StarCoder pour le projet WebWise AI. Il couvre l'installation locale pour le développement et les tests.

## Prérequis Matériels

- GPU NVIDIA avec au moins 8GB de VRAM (testé sur GTX 1070Ti)
- Au moins 16GB de RAM système
- Environ 10GB d'espace disque pour le modèle

## Prérequis Logiciels

- Python 3.8 ou supérieur
- Git
- Pilotes NVIDIA et CUDA installés

## Étapes d'Installation

### 1. Préparation de l'Environnement

```bash
# Création du répertoire projet
mkdir starcoder-setup
cd starcoder-setup

# Création et activation de l'environnement virtuel
python -m venv venv
source venv/bin/activate
```

### 2. Installation des Dépendances

```bash
# Mise à jour de pip
pip install --upgrade pip

# Installation des dépendances principales
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate
pip install huggingface_hub
pip install -U bitsandbytes

# Pour la quantification et l'optimisation
pip install protobuf
```

### 3. Configuration de Hugging Face

1. Créer un compte sur https://huggingface.co/
2. Aller sur https://huggingface.co/bigcode/starcoder
3. Accepter les conditions d'utilisation du modèle
4. Se connecter via le terminal :

```bash
huggingface-cli login
# Entrer votre token d'authentification quand demandé
```

### 4. Test de l'Installation

Créer un fichier `test_starcoder.py` avec le contenu suivant :

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    print("Étape 1: Vérification du GPU...")
    print(f"GPU disponible : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Modèle GPU : {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU totale : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print("\nÉtape 2: Configuration de la quantification...")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    print("\nÉtape 3: Chargement du modèle...")
    model = AutoModelForCausalLM.from_pretrained(
        "bigcode/starcoder",
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder")

    print("\nÉtape 4: Test simple de génération...")
    prompt = "# Python function to calculate factorial"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        inputs,
        max_length=100,
        temperature=0.7,
        num_return_sequences=1
    )

    print("\nRésultat :")
    print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    main()
```

Exécuter le test :

```bash
python test_starcoder.py
```

Note : Le premier lancement téléchargera environ 10GB de données pour le modèle.

## Explications Techniques

### Quantification 8-bit

La quantification permet de réduire l'empreinte mémoire du modèle en convertissant les poids de 32-bit à 8-bit, tout en maintenant des performances acceptables. La configuration utilise :

- `load_in_8bit=True` : Active la quantification 8-bit
- `llm_int8_enable_fp32_cpu_offload` : Permet de décharger certaines opérations sur le CPU si nécessaire

### Gestion de la Mémoire

- `device_map="auto"` : Permet à Hugging Face de gérer automatiquement la répartition entre CPU et GPU
- La configuration est optimisée pour une carte graphique avec 8GB de VRAM

## Problèmes Courants

### Erreur 401 Unauthorized

Si vous obtenez une erreur 401, vérifiez que :

1. Vous avez bien accepté les conditions d'utilisation sur le site
2. Vous êtes correctement connecté via `huggingface-cli login`

### Erreurs de Mémoire CUDA

Si vous rencontrez des erreurs de mémoire :

1. Fermez les autres applications utilisant le GPU
2. Redémarrez Python/votre notebook
3. Vérifiez que la quantification 8-bit est bien activée

## Prochaines Étapes

Une fois l'installation validée, vous pourrez :

1. Créer un service FastAPI pour exposer le modèle
2. Configurer le déploiement Docker
3. Mettre en place l'intégration avec Django

Pour toute question ou problème, consulter la [documentation officielle de StarCoder](https://huggingface.co/bigcode/starcoder) ou ouvrir une issue sur le projet.
