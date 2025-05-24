import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Define the categories
categories = ['off', 'safe', 'unsafe']

# Function to get a random image from a category
def get_random_image(category):
    category_path = os.path.join('data', category)
    image_files = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
    random_image = random.choice(image_files)
    image_path = os.path.join(category_path, random_image)
    return image_path, random_image

# Create a figure with 3x3 subplots
fig, axs = plt.subplots(3, 3, figsize=(10, 10))

# Load and display an image from each category
for i, category in enumerate(categories):
    for j in range(3):
        image_path, image_name = get_random_image(category)
        img = Image.open(image_path)
        axs[i, j].imshow(np.array(img))
        axs[i, j].set_title(f"{category.capitalize()}: {image_name}")
        axs[i, j].axis('off')

# Adjust layout and add a title
plt.tight_layout()
plt.suptitle("Sample Images from Each Category", fontsize=16)
plt.subplots_adjust(top=0.92)

# Save the figure
plt.savefig('category_samples.png', dpi=300)

# Show the plot
plt.show()

print("Plot created and saved as 'category_samples.png'") 