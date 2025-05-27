import torch # Bibliothèque principale pour le deep learning
from transformers import (
    AutoTokenizer, # Pour convertir le texte en nombres (tokens)
    AutoModelForMaskedLM, # Le modèle BERT pour le masquage de mots
    DataCollatorForLanguageModeling, # Prépare les données pour l'entraînement
    Trainer, # Gère l'entraînement du modèle
    TrainingArguments #Configure les paramètres d'entraînement
)
from datasets import Dataset # Gère les jeux de données

def check_sequence_lengths(tokenizer, texts):
    """
    Vérifie la longueur des séquences après tokenization.
    Affiche des statistiques utiles pour comprendre l'utilisation des tokens.
    """
    lengths = []
    print("\n=== Analyse des longueurs de séquences ===")
    
    for text in texts:
        tokens = tokenizer.encode(text)
        lengths.append(len(tokens))
        print(f"Tokens: {len(tokens):2d} | Texte: {text}")
    
    print("\n=== Statistiques ===")
    print(f"Longueur minimum: {min(lengths)} tokens")
    print(f"Longueur maximum: {max(lengths)} tokens")
    print(f"Longueur moyenne: {sum(lengths)/len(lengths):.1f} tokens")
    print("=" * 50 + "\n")

# 1. Préparation des données
def prepare_data():
    """
    Prépare les données d'entrainement.
    Returns: Un objet Dataset contenant nos phrases d'exemple.
    """
    # Liste de phrases simples pour l'entraînement
    # Chaque phrase utilise un vocabulaire basique et des structures répétitives    
    texts = [
        # Définitions fondamentales - Répétition intentionnelle des termes clés
        """HTML (HyperText Markup Language) stands as the fundamental building block of all web development, serving as 
        the essential structural foundation that every website requires. As a building block technology, HTML defines 
        both the meaning and structure of web content, creating the basic framework that browsers interpret. While other 
        technologies like CSS handle presentation and JavaScript manages behavior, HTML's building blocks remain the 
        core foundation of web structure.""",

        """When we examine web technologies, HTML (HyperText Markup Language) emerges as the primary building block for 
        all content structure and organization. This fundamental building block determines how content is structured, 
        working as the essential framework that gives meaning to web elements. Modern websites rely on these HTML 
        building blocks for their structural foundation, complementing them with CSS for styling and JavaScript for 
        interactive features.""",

        # Structure et Sémantique - Focus sur la structure
        """The structural foundation of every webpage begins with proper HTML markup, which provides the essential 
        framework for organizing content. This structural approach ensures that web content maintains a logical 
        hierarchy, with each HTML element serving a specific structural purpose. Through careful structural planning, 
        developers create well-organized documents where each building block contributes to the overall content 
        structure and semantic meaning.""",

        """Understanding HTML structure remains crucial for effective web development, as the structural elements form 
        the building blocks of content organization. When developers properly implement HTML structure, they create 
        a solid foundation that improves both accessibility and search engine optimization. The structural hierarchy 
        created by HTML building blocks helps browsers and search engines interpret content relationships and 
        importance.""",

        # Relations avec CSS et JavaScript - Contexte élargi
        """Modern web development relies on the interplay between three core technologies, with HTML providing the 
        fundamental building blocks and structural foundation. While HTML's building blocks create the content 
        structure, CSS transforms these structural elements through visual styling. JavaScript then enhances these 
        HTML building blocks by adding interactivity and dynamic behavior, creating a complete web experience that 
        maintains its structural integrity.""",

        """Professional web developers recognize HTML as the essential building block that initiates the development 
        process, establishing the structural framework for their projects. This foundational building block works 
        seamlessly with CSS, which styles the structural elements, and JavaScript, which adds dynamic features. 
        Together, these technologies transform HTML's building blocks into rich, interactive websites while 
        maintaining proper structure.""",

        # Accessibilité et SEO - Impact de la structure
        """The semantic structure provided by HTML building blocks plays a crucial role in web accessibility, allowing 
        assistive technologies to interpret content effectively. Each structural element within the HTML framework 
        contributes to better accessibility, with proper building blocks ensuring that screen readers can navigate 
        content logically. This semantic approach to structure also enhances search engine optimization, as search 
        engines rely on HTML's building blocks to understand content relationships.""",

        """Web accessibility begins with proper HTML structure, using semantic building blocks to create meaningful 
        content organization. These structural elements, when properly implemented, form the building blocks of an 
        accessible website. Screen readers and other assistive technologies depend on this structural foundation to 
        interpret content, making HTML's semantic building blocks essential for ensuring web content remains 
        accessible to all users.""",

        # Performance et Maintenance - Importance de la structure
        """Well-structured HTML serves as the building block for high-performance websites, with proper structural 
        elements contributing to faster page rendering and better maintenance. When developers focus on creating 
        clean HTML structure, using appropriate building blocks for content organization, they establish a 
        foundation that's easier to maintain and update. This structural approach, based on proper HTML building 
        blocks, ensures websites remain efficient and manageable throughout their lifecycle.""",

        """Long-term website maintenance relies heavily on proper HTML structure, with well-organized building blocks 
        making future updates more manageable. These structural elements, when properly implemented as building 
        blocks, create a maintainable foundation that developers can easily modify and extend. The careful use of 
        HTML building blocks in creating this structural base pays dividends throughout a website's development 
        lifecycle, facilitating both maintenance and updates."""
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
        mlm_probability=0.20  # 20% des mots seront masqués
    )

    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir="./results",  # Dossier pour sauvegarder les résultats
        num_train_epochs=35,     # Nombre de passages sur les données
        per_device_train_batch_size=4,  # Nombre d'exemples traités à la fois
        learning_rate=3e-5,      # Vitesse d'apprentissage
        weight_decay=0.01,       # Réduit l'overfitting
        logging_steps=2,         # Fréquence des logs
        save_strategy="no",      # Ne sauvegarde pas les checkpoints
        use_cpu=True,           # Utilise le CPU
        lr_scheduler_type="linear", # Diminue progressivement le learning rate de sa valeur initiale à 0. Cela aide à affiner l'apprentissage vers la fin de l'entraînement
        warmup_steps=12, # Nombre d'étapes où le learning rate augmente progressivement de 0 à sa valeur maximale. Aide à stabiliser le début de l'entraînement
        max_grad_norm=1.0, # Limite la norme maximale des gradients à 1.0. Empêche les explosions de gradients et stabilise l'entraînement
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

    print("Vérification des longueurs de séquences...")
    check_sequence_lengths(tokenizer, dataset['text'])
    
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
            "HTML serves as the essential [MASK] block for all web development projects.",
            "The structural [MASK] created by HTML helps browsers understand content.",
            "Developers use HTML [MASK] blocks to create organized content hierarchies.",
            "Web accessibility depends on proper HTML [MASK] and semantic elements.",
            "HTML provides the [MASK] foundation that every website requires.",
            "Each HTML element serves a specific [MASK] purpose in web development.",
            "Professional developers understand how HTML [MASK] affects content organization.",
            "The semantic [MASK] of HTML improves both accessibility and SEO.",
            "Proper HTML [MASK] makes websites easier to maintain and update.",
            "Search engines rely on HTML's [MASK] blocks to understand content."
        ]
        
        for phrase in test_phrases:
            test_model(phrase)
            
    except Exception as e:
        print(f"\nErreur : {str(e)}")
        print(f"Type d'erreur : {e.__class__.__name__}")
    finally:
        print("\n=== Processus terminé ===")