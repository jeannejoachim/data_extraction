#-----------------------------------------------------------------------------
# Script de restructuration de dossiers pour faciliter la correction
#
# Dernière MAJ : 12-09-2023
#-----------------------------------------------------------------------------

# Module imports
import os
import os.path
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

try:
    from functions import *
except:
    pass

class parameters():
    # Data folder path, the 'r' is important to convert the path in raw string!
    data_folder = r"C:\Users\Jeanne\Documents\Documents_communs\Laura_donnees\Donnees\BMC-006_19-10-23"
    nb_interp = 100     # Number of points for the data interpolation used to draw color maps
    begin_data = "<DATA>"      # String after which the data point begins
    end_data = "<END DATA>"    # String before which the data point ends
    # Dictionary that defines initial data names (key) vs. cleaned-up data name (value)
    data_cleanup = {"Time, s": "time",
                    "Position (z), mm": "pos_z",
                    "Position (x), mm": "pos_x",
                    "Position (y), mm": "pos_y",
                    "Fz, gf": "Fz"}

prm = parameters()

# Pour un fichier, tous points de mesure
f_thickness = [f for f in os.listdir(prm.data_folder) if (".txt" and "thickness") in f]
# file_path = os.path.join(dossier_donnees, f_thickness[0])
file_path = os.path.join(prm.data_folder, "C4_thickness_trial3_test1_0750gf_v1_a06.txt")

print("file : ", file_path)
data_points = extract_data(file_path, prm)

# Extraction of (skin) thickness for each data point
thickness = np.zeros(len(data_points))
pos_x = np.zeros(len(data_points))
pos_y = np.zeros(len(data_points))
for i in range(len(data_points)):
    plt.plot(data_points[i]["time"], data_points[i]["pos_z"], "-")
    thickness[i] = -np.max(data_points[i]["pos_z"])
    pos_x[i] = np.mean(data_points[i]["pos_x"])
    pos_y[i] = np.mean(data_points[i]["pos_y"])

# Affichage couleur
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(pos_x, pos_y, epaisseur, "o")
xi, yi, zi = interpolate_data(pos_x, pos_y, thickness, prm)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xi, yi, zi, cmap='viridis')
fig.colorbar(surf, ax=ax, label='Z Coordinate')

#print([pos_x,pos_y,epaisseur])
print(thickness)
print("min = ", np.min(thickness), "max = ", np.max(thickness))
plt.show()