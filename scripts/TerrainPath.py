#%%
import numpy as np
from global_land_mask import globe
import pandas as pd
import os
import matplotlib.pyplot as plt

#%%
# --------------------------------------------------------------------
# Function to calculate azimuth (bearing) and great-circle distance
# between two latitude/longitude points using the haversine formula.
# --------------------------------------------------------------------
def aziDist(initial_latitude, final_latitude, initial_longitude, final_longitude):
    R = 6371  # Radius of Earth in kilometers

    phi_1 = np.radians(initial_latitude)
    phi_2 = np.radians(final_latitude)
    lamda_1 = np.radians(initial_longitude)
    lamda_2 = np.radians(final_longitude)

    del_phi = phi_2 - phi_1
    del_lam = lamda_2 - lamda_1

    # Haversine formula for distance
    a = np.sin(del_phi / 2)**2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(del_lam / 2)**2
    d = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Azimuth (bearing)
    theta = np.arctan2(
        np.sin(del_lam) * np.cos(phi_2),
        np.cos(phi_1) * np.sin(phi_2) - np.sin(phi_1) * np.cos(phi_2) * np.cos(del_lam)
    )
    theta = np.degrees(theta)

    return theta, d

# --------------------------------------------------------------------
# Function to compute a new point from an origin point given a distance
# and bearing (used to trace the path step-by-step).
# --------------------------------------------------------------------
def points(initial_latitude, initial_longitude, theta, step):
    R = 6371  # Radius of Earth in kilometers

    lat_2 = np.arcsin(
        np.sin(np.radians(initial_latitude)) * np.cos(step / R) +
        np.cos(np.radians(initial_latitude)) * np.sin(step / R) * np.cos(np.radians(theta))
    )

    lon_2 = np.radians(initial_longitude) + np.arctan2(
        np.sin(np.radians(theta)) * np.sin(step / R) * np.cos(np.radians(initial_latitude)),
        np.cos(step / R) - np.sin(np.radians(initial_latitude)) * np.sin(lat_2)
    )

    return np.degrees(lat_2), np.degrees(lon_2)

#%%
"""
This script calculates a step-by-step geographic path between a 
transmitter and a receiver using haversine + azimuth logic. 

Each coordinate is classified as either 'Land' or 'Water' using 
the global_land_mask library.

Results are saved to a CSV for use in ground wave modeling.
"""

# Replace with actual coordinates
final_lon, final_lat = "your_final_longitude", "your_final_latitude"
initial_lat, initial_lon = "your_initial_longitude", "your_initial_latitude"

# Calculate initial bearing and total distance
az, d = aziDist(
    initial_latitude=initial_lat,
    final_latitude=final_lat,
    initial_longitude=initial_lon,
    final_longitude=final_lon
)

step = 1  # Step size in km
path_cords = []

# Generate geographic points along the great-circle path
for point in range(round(d)):
    lat_2, lon_2 = points(
        initial_latitude=initial_lat,
        initial_longitude=initial_lon,
        theta=az,
        step=step
    )
    path_cords.append([lon_2, lat_2])
    initial_lat, initial_lon = lat_2, lon_2  # Move to next step

# Classify each coordinate as Land or Water
path_terrain = []
for lon, lat in path_cords:
    land = globe.is_land(lat, lon)
    path_terrain.append("Land" if land else "Water")

# Prepare data for saving
lat = [coord[1] for coord in path_cords]
lon = [coord[0] for coord in path_cords]

data = {
    "Lat": lat,
    "Lon": lon,
    "Terrain": path_terrain
}

# Save to CSV
output_dir = 'coordinates'
os.makedirs(output_dir, exist_ok=True)

filename = 'Coordinates_Example.csv'  # <-- Change as needed
df = pd.DataFrame(data)
df.to_csv(os.path.join(output_dir, filename), index=False)

# %%
