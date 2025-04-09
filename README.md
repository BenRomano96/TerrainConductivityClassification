# Terrain Conductivity Classification

This project combines satellite imagery, CNN-based terrain classification, geographic coordinate analysis, and conductivity modeling to build a complete system that maps and quantifies terrain conductivity between two geographic points.

---

## 📦 Project Overview

### 1. `TerrainClassification.ipynb`

Trains a CNN model (ResNet50-based) to classify terrain images into:  
**Desert, Forest, Mountain, Plains, Argicultural, Urban**

- **Dataset Source:**  
  [Durgesh Rao on Kaggle (durgeshrao9993)](https://www.kaggle.com/datasets/durgeshrao9993/different-terrain-types-classification/data)  
  **License:** Community Data License Agreement – Sharing – Version 1.0

- **Knowledge:**  
  The CNN model implementation was based on techniques and code referenced from a publicly available GitHub repository:  
  [nachi-hebbar/Transfer-Learning-ResNet-Keras](https://github.com/nachi-hebbar/Transfer-Learning-ResNet-Keras)
---

### 2. `TerrainPath.py`
- Calculates azimuth and distance between a transmitter and receiver.
- Generates a sequence of coordinates between them.
- Classifies each point as either **Land** or **Water**.
- Saves the output as a CSV table.

---

### 3. `PathSatelliteImages.py`
- Reads the coordinates CSV from `TerrainPath.py`.
- For each **Land** point, retrieves a high-resolution satellite image using the **Mapbox API**.
- Saves images to disk for classification.

> > 🔑 **To use Mapbox:**  
> Create a free access token here:  
> [mapbox](https://account.mapbox.com/auth/signin/?route-to=https%3A%2F%2Fconsole.mapbox.com%2Faccount%2Faccess-tokens%2F%3Fauth%3D1)
---

### 4. `TerrainConductivityTable.py`
- Loads the model from `TerrainClassification.ipynb`.
- Uses it to classify the terrain of each satellite image.
- Calculates **weighted conductivity** per image based on class probabilities.
- Conductivity values per class were taken from published research.

> 📄 **Source for conductivity values:**
> [ITU Handbook on Ground-Wave Propagation (2014)](https://extranet.itu.int/brdocsearch/R-HDB/R-HDB-59/R-HDB-59-2014/R-HDB-59-2014-PDF-E.pdf)


- Output is saved as a CSV file.

---

### 5. `PathConductivity.py`
- Merges the coordinate data from `TerrainPath.py` and the prediction results from `TerrainConductivityTable.py`.
- Generates a **final result table** including:
  - GPS coordinates
  - Terrain prediction
  - Class probabilities
  - Conductivity value
  - Thumbnail of satellite image
- Also saves it as an HTML file with preview.

---

## 📁 Repository Structure

