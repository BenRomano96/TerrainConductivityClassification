#%% Import libraries
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
import os
import cv2
import numpy as np
import pandas as pd
#%%
# --------------------------------------------------------------------
# Function to generate terrain classification and weighted conductivity
# from the images folder created in PathSatelliteImages.py.
# --------------------------------------------------------------------
def conductivityTable(image_dir, folder_name, fig_size, images):
    class_labels = ['Desert', 'Forest', 'Mountain', 'Plains', 'Argicultural', 'Urban']
    
    terrain = []
    perc = []

    for image in images:
        print(image)
        # Load and preprocess the image
        img = cv2.imread(f'{image_dir}/{folder_name}/{image}')
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error processing image {image}: {e}")
            continue

        # Resize image to model input size
        resize = tf.image.resize(img_rgb, fig_size)

        # Predict class probabilities using the trained model
        yhat = model.predict(np.expand_dims(resize, 0))  # Add batch dimension

        # Get predicted class index
        yhat_class = np.argmax(yhat, axis=1)
        terrain.append(yhat_class)  # Save class index (as a list)

        # Print class name for tracking
        print(class_labels[yhat_class[0]])

        # Store predicted class probabilities (%)
        for c in yhat:
            perc.append(c * 100)

    # Remove '.DS_Store' if present
    if '.DS_Store' in images:
        images.remove('.DS_Store')

    # Extract numeric ID from filename (e.g., 'fig23.png' → '23')
    names = [img.replace('fig', '').replace('.png', '') for img in images]

    # Class-wise percentage lists
    desert, forest, mountain, plains, argicultural, urban = [], [], [], [], [], []

    for p in perc:
        desert.append(round(p[0], 2))
        forest.append(round(p[1], 2))
        mountain.append(round(p[2], 2))
        plains.append(round(p[3], 2))
        argicultural.append(round(p[4], 2))
        urban.append(round(p[5], 2))

    # Convert class index to class label
    terrain_class = [class_labels[i[0]] for i in terrain]

    # Weighted conductivity per image
    conductivity_lookup = [0.1, 0.1, 0.3, 1, 3, 6.5]
    conductivity = []

    for p in perc:
        weighted = sum([(p[i] * conductivity_lookup[i]) / 100 for i in range(6)])
        conductivity.append(round(weighted, 2))

    # Full image paths
    full_image_paths = [f'{image_dir}/{folder_name}/{img}' for img in images]

    # Build final data dictionary
    result_dict = {
        'Km': names,
        'image': full_image_paths,
        'Terrain': terrain_class,
        f'{class_labels[0]}%': desert,
        f'{class_labels[1]}%': forest,
        f'{class_labels[2]}%': mountain,
        f'{class_labels[3]}%': plains,
        f'{class_labels[4]}%': argicultural,
        f'{class_labels[5]}%': urban,
        'Weighted Conductivity level [mS/m]': conductivity
    }

    return result_dict

#%% Load trained model
model_dir = 'models'
model = load_model(f'{model_dir}/model_6.h5')

#%% Define folder and image list
image_dir = 'images'
folder_name = 'your_folder_name'  # Replace with the actual folder name
fig_size = (224, 224)
images = os.listdir(f'{image_dir}/{folder_name}')

# Remove system file if exists
if '.DS_Store' in images:
    images.remove('.DS_Store')

#%% Run classification and save table
data = conductivityTable(image_dir, folder_name, fig_size, images)
df = pd.DataFrame(data)
df = df.round(2)
df.to_csv('"your_tabel".csv', index=False)

# %%
