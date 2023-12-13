#-----------------------------------------------------------------------------
# Post-treatment script
#
# Dernière MAJ : 12-09-2023
#-----------------------------------------------------------------------------

# Module imports
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from PIL import Image

try:
    from functions import *
except:
    pass

class parameters_thickness():
    # Data folder path, the 'r' is important to convert the path in raw string!
    data_folder = r"C:\Users\Jeanne\Documents\Documents_communs\Laura_donnees\Donnees"
    nb_interp = 100     # Number of points for the data interpolation used to draw color maps
    begin_data = "<DATA>"      # String after which the data point begins
    end_data = "<END DATA>"    # String before which the data point ends
    # Dictionary that defines initial data names (key) vs. cleaned-up data name (value)
    data_cleanup = {"Time, s": "time",
                    "Position (z), mm": "pos_z",
                    "Position (x), mm": "pos_x",
                    "Position (y), mm": "pos_y",
                    "Fz, gf": "Fz"}
    filename_separator = "_"       # string that separates sample ID to test name in the filename
    filename_ID_position = 1
    # 1 = sample ID is the first element given (so appears before the first file_name_separator)
    plot_2D = True
    plot_on_picture = True
    plot_3D = True
    show_plots = True
    alpha = 0.5  # 0 = transparent, 1 = opaque (used for if plot_thickness_on_image = True)

prm = parameters_thickness()

####################################################
## THICKNESS
####################################################
#TODO récupérer tous les indices des échantillons dans le dossier => généraliser pour plusieurs essais
sample_ID = "Q1-JAB9772"

# Extraction of (skin) thickness
f_thickness = [f for f in os.listdir(prm.data_folder) if (".txt" and "thickness") in f]
file = sample_ID + "_thickness.txt"  # TODO revoir pour robustesse

#TODO généraliser pour des points dans des fichiers différents
# Pour un fichier, tous points de mesure
file_path = os.path.join(prm.data_folder, sample_ID, file)

# Extraction for each data point
data_points_thickness = extract_data(file_path, prm)
pos_x, pos_y, thickness = extract_thickness(data_points_thickness)
xi, yi, zi = interpolate_data(pos_x, pos_y, thickness, prm)
# xi, yi, zi = interpolate_data(vect_sensor_sorted[0,:], vect_sensor_sorted[0,:], thickness, prm)

# 2D graph, sensor coordinates
if prm.plot_2D:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pcm = ax.pcolormesh(xi, yi, zi, shading='nearest', cmap="jet",
                        vmin=thickness.min(), vmax=thickness.max())
    fig.colorbar(pcm, label='thickness')
    plt.scatter(pos_x, pos_y, c=thickness, ec='k', cmap="jet",
                vmin=thickness.min(), vmax=thickness.max())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(sample_ID + " - sensor coordinates")
    fig.savefig(os.path.join(prm.data_folder, sample_ID, sample_ID + "_thickness_2D.png"), bbox_inches='tight')
    #plt.close(fig)

# 3D graph, sensor coordinates
if prm.plot_3D:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xi, yi, zi, cmap='jet',
                           vmin=thickness.min(), vmax=thickness.max())
    fig.colorbar(surf, ax=ax, label='thickness')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('thickness')
    plt.title(sample_ID + " - sensor coordinates")
    fig.savefig(os.path.join(prm.data_folder, sample_ID, sample_ID + "_thickness_3D.png"), bbox_inches='tight')
    pickle.dump(fig, open(os.path.join(prm.data_folder, sample_ID, sample_ID + "_thickness_3D.fig.pickle"), 'wb'))
    #TODO (voir avec Laura): plot 3D above image?

# Graph on pixel image
if prm.plot_on_picture:
    # Load image
    img = np.asarray(Image.open(os.path.join(prm.data_folder, sample_ID, sample_ID + ".jpg")))

    # Extract pixel map
    map_points = extract_map(os.path.join(prm.data_folder, sample_ID, sample_ID + "_map.map"), prm)

    # Sort map by point ID (security check)
    vect_px = np.array((map_points['PixelX'], map_points['PixelY'], map_points['PointID']))
    vect_px_sorted_by_point_id = vect_px[:, vect_px[2, :].argsort()]

    # Interpolate data on pixel map
    xpx, ypx, zpx = interpolate_data(vect_px_sorted_by_point_id[0, :], vect_px_sorted_by_point_id[1, :], thickness, prm)

    # 2D graph with pixel image coordinates
    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(img)
    pcm = ax.pcolormesh(xpx, ypx, zpx, shading='nearest', cmap="jet", alpha=prm.alpha,
                        vmin=thickness.min(), vmax=thickness.max())
    fig.colorbar(pcm, label='thickness')
    plt.scatter(vect_px_sorted_by_point_id[0, :], vect_px_sorted_by_point_id[1, :], c=thickness, s=2., cmap="jet",
                vmin=thickness.min(), vmax=thickness.max())
    plt.title(sample_ID)
    plt.axis('off')
    fig.savefig(os.path.join(prm.data_folder, sample_ID, sample_ID + "_thickness_on_image.png"), bbox_inches='tight')
    #plt.close(fig)

#print([pos_x,pos_y,epaisseur])
#print(thickness)
#print("min = ", np.min(thickness), "max = ", np.max(thickness))

if prm.show_plots:
    plt.show()


####################################################
# Extraction of elastic modulus
# Pour un fichier, tous points de mesure
f_indentation = [f for f in os.listdir(prm.data_folder) if (".txt" and "indentation") in f]
# file_path = os.path.join(dossier_donnees, f_indentation[0])
file_path = os.path.join(prm.data_folder, "Q1-JAB9772_indentation.txt")

# Extraction for each data point
prm.end_data = "<divider>"
data_points_indentation = extract_data(file_path, prm)

# Graph
plt.figure()
# for 1 point
plt.plot(data_points_indentation[0]["pos_z"], data_points_indentation[0]["Fz"], "-")


plt.show()