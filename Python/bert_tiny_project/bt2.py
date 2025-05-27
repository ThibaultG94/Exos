import torch # Bibliothèque principale pour le deep learning
from transformers import (
    AutoTokenizer, # Pour convertir le texte en nombres (tokens)
    AutoModelForMaskedLM, # Le modèle BERT pour le masquage de mots
    DataCollatorForLanguageModeling, # Prépare les données pour l'entraînement
    Trainer, # Gère l'entraînement du modèle
    TrainingArguments #Configure les paramètres d'entraînement
)
from datasets import Dataset # Gère les jeux de données

# 1. Préparation des données
def prepare_data():
    """
    Prépare les données d'entrainement.
    Returns: Un objet Dataset contenant nos phrases d'exemple.
    """
    # Liste de phrases simples pour l'entraînement
    # Chaque phrase utilise un vocabulaire basique et des structures répétitives    
    texts = [
        # Phrases sur les animaux
        "Dogs are friendly pets that love to play.",
        "Cats like to sleep during the day.",
        "Birds can fly in the blue sky.",
        "Fish swim in the clear water.",
        
        # Phrases sur la nourriture
        "Pizza is a delicious food to eat.",
        "Ice cream is cold and sweet.",
        "Apples are healthy fruits to eat.",
        "Water is important to drink daily.",
        
        # Phrases sur la météo
        "The sun is bright and warm.",
        "Rain falls from dark clouds.",
        "Snow is white and cold.",
        "Wind blows through the trees.",
        
        # Phrases sur les activités quotidiennes
        "People walk in the park.",
        "Children play in the garden.",
        "Students study in the classroom.",
        "Workers sleep at night."
    ]

    # Création d'un Dataset à partir de notre liste de textes
    # Dataset est une classe qui facilite la gestion des données
    return Dataset.from_dict({"text": texts})

#2. Préparation du modèle et du tokenizer
def prepare_model():
    """
    Prépare le tokenizer et le modèle.
    Returns: tokenizer et model configurés
    """
    print("Chargement du modèle BERT-tiny...")
    # Le tokenizer convertit le texte en nombres que le modèle peut comprendre
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    # Le modèle lui-même, spécialisé dans le masquage de mots
    model = AutoModelForMaskedLM.from_pretrained("prajjwal1/bert-tiny")
    return tokenizer, model

#3. Tokenization des données
def tokenize_function(examples, tokenizer):
    """
    Convertit les textes en tokens (nombres).
    Args:
        examples: Dictionnaire contenant les textes
        tokenizer: L'outil de tokenization
    Returns: Textes convertis en tokens
    """

    return tokenizer(
        examples["text"],
        padding="max_length",  # Ajoute des padding pour avoir des longueurs égales
        truncation=True,  # Coupe les textes trop longs
        max_length=64,    # Longueur maximale (réduite car phrases simples)
        return_special_tokens_mask=True  # Nécessaire pour le masquage
    )

#4. Configuration de l'entraînement
def setup_training(model, tokenized_dataset, tokenizer):
    # Prépare les données pour l'entraînement masqué
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,  # Active le masquage de mots
        mlm_probability=0.15  # 15% des mots seront masqués
    )

    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir="./results",  # Dossier pour sauvegarder les résultats
        num_train_epochs=10,     # Nombre de passages sur les données
        per_device_train_batch_size=4,  # Nombre d'exemples traités à la fois
        learning_rate=1e-4,      # Vitesse d'apprentissage
        logging_steps=2,         # Fréquence des logs
        save_strategy="no",      # Ne sauvegarde pas les checkpoints
        use_cpu=True,           # Utilise le CPU
    )

    # Création du Trainer qui va gérer l'entraînement
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    return trainer

#5. Test du modèle entraîné
def test_model(text):
    """
    Teste le modèle avec une phrase donnée.
    Args:
        text: Phrase à tester avec un mot masqué [MASK]
    """
    # Charge le modèle entraîné
    tokenizer = AutoTokenizer.from_pretrained("./custom_model")
    model = AutoModelForMaskedLM.from_pretrained("./custom_model")
    
    # Convertit le texte en tokens
    inputs = tokenizer(text, return_tensors="pt")
    
    # Trouve la position du mot masqué
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    
    # Fait la prédiction
    output = model(**inputs)
    mask_token_logits = output.logits[0, mask_token_index, :]
    
    # Obtient les 5 meilleurs résultats
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    
    print(f"\nRésultats pour: {text}")
    print("-" * 50)
    for token in top_5_tokens:
        prediction = text.replace(tokenizer.mask_token, tokenizer.decode([token]).strip())
        print(f"• {prediction}")

#6. Fonction principale d'entraînement
def train_model():
    dataset = prepare_data()
    tokenizer, model = prepare_model()
    
    print("Tokenization des données...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    trainer = setup_training(model, tokenized_dataset, tokenizer)
    
    print("Début de l'entraînement...")
    trainer.train()
    
    print("Sauvegarde du modèle...")
    model.save_pretrained("./custom_model")
    tokenizer.save_pretrained("./custom_model")

if __name__ == "__main__":
    try:
        print("=== Début du processus d'entraînement ===")
        train_model()
        
        print("\n=== Test du modèle ===")
        # Phrases de test simples avec des réponses évidentes
        test_phrases = [
            "Dogs like to [MASK] with toys.",
            "Cats love to [MASK] during the day.",
            "Birds [MASK] in the sky.",
            "The sun is very [MASK] today.",
            "Children [MASK] in the park."
        ]
        
        for phrase in test_phrases:
            test_model(phrase)
            
    except Exception as e:
        print(f"\nErreur : {str(e)}")
        print(f"Type d'erreur : {e.__class__.__name__}")
    finally:
        print("\n=== Processus terminé ===")