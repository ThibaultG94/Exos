import os
from PIL import Image, ImageDraw, ImageFont

# Crée le dossier pour les images s'il n'existe pas déjà
images_dir = "alphabet_images"
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Définis la taille de l'image et la couleur de fond
size = (200, 200)
background_color = "#171f39"

# Chargement de la police DejaVu Sans, sans gras
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Assure-toi que ce chemin est correct
font_size = 100  # Taille de la police
font = ImageFont.truetype(font_path, font_size)

def create_image(letter):
    # Crée une image avec le fond spécifié
    image = Image.new("RGB", size, background_color)
    draw = ImageDraw.Draw(image)
    
    # Calcule la largeur et la hauteur du texte pour le centrer
    text_width, text_height = draw.textsize(letter, font=font)
    # Trouve le centre de l'image
    width_center = (size[0] - text_width) / 2
    
    # Pour centrer verticalement, prend en compte l'ascendant et le descendant de la police
    ascent, descent = font.getmetrics()
    text_height = ascent + descent
    height_center = (size[1] - text_height) / 2 + ascent

    # Ajuste le texte pour qu'il soit parfaitement centré
    draw.text((width_center, height_center - ascent), letter, font=font, fill="white")
    
    # Sauvegarde l'image dans le dossier spécifié
    image.save(os.path.join(images_dir, f"{letter}.png"))
    print(f"Image for {letter} saved in {images_dir}/!")

# Génère une image pour chaque lettre de l'alphabet
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    create_image(letter)
