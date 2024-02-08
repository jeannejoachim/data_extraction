#-----------------------------------------------------------------------------
# Script to generate missing or broken .map, to be used in post-treatment
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

try:
    from functions import *
except:
    pass

data_folder = r"C:\Users\Jeanne\Documents\Documents_communs\Laura_donnees\Fichiers pour générer .map manquants\Exemple"
start = "MAPPING	MAPPING	MAPPING	MAPPING	MAPPING	MAPPING	MAPPING"
end = "Reference1"
nb_pt = 27       #max = 27

# Reference coordinates (ref number = lines, (x,y) = columns)
# To be gathered from test with same reference points
Pixel_ref = np.array([[506, 945],
                      [1097, 48]])
Scan_ref = np.array([[16.132000, -12.182000],
                     [-12.636000, 17.906000]])

# Extract map from reference
# Extract pixel map
map_points = extract_map(os.path.join(data_folder, "Z2-JAB9793_map.map"), "MAPPING", "Reference1")

#TEST PLOT
# Plot in pixel coordinates
fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.suptitle("points by pixel coordinates")
ax.set_aspect('equal', 'box')
#Plot in sensor coordinates
fig = plt.figure(2)
ax = fig.add_subplot(111)
plt.suptitle("points by sensor coordinates")
ax.set_aspect('equal', 'box')

# Plot points by pixel coordinates
a = True
if a:
    plt.figure(1)
    plt.scatter(map_points['PixelX'][:nb_pt], map_points['PixelY'][:nb_pt], c=map_points['PointID'][:nb_pt],
                cmap="jet", marker="x")
    plt.plot(*Pixel_ref[0, :], 'xr', *Pixel_ref[1, :], 'xb')

# Plot points by sensor coordinates
a = False
if a:
    plt.figure(2)
    plt.scatter(map_points['ScanX(mm)'][:nb_pt], map_points['ScanY(mm)'][:nb_pt], c=map_points['PointID'][:nb_pt], cmap="jet")
    plt.plot(*Scan_ref[0, :], 'xr', *Scan_ref[1, :], 'xb')

# Plot points by sensor coordinates - X-flip
a = False
map_points['ScanX(mm)'] = -map_points['ScanX(mm)']
Scan_ref[:, 0] = -Scan_ref[:, 0]
if a:
    plt.figure(2)
    plt.scatter(-map_points['ScanX(mm)'][:nb_pt], map_points['ScanY(mm)'][:nb_pt], c=map_points['PointID'][:nb_pt], cmap="jet")
    plt.plot(*Scan_ref[0, :], 'xr', *Scan_ref[1, :], 'xb')
    plt.title("with x-flip")

##################################################
# Find expand ratio
vec_px_ref = np.array(Pixel_ref[1, :]-Pixel_ref[0, :])
vec_scan_ref = np.array(Scan_ref[1, :]-Scan_ref[0, :])
ratio = np.linalg.norm(vec_px_ref)/np.linalg.norm(vec_scan_ref)

Scan_ref = Scan_ref * ratio
map_points['ScanX(mm)'] = map_points['ScanX(mm)']*ratio
map_points['ScanY(mm)'] = map_points['ScanY(mm)']*ratio

# Plot vectors
a = False
if a:
    plt.figure(1)
    plt.quiver(*Pixel_ref[0, :], *vec_px_ref, color=['k'], angles='xy', scale_units='xy', scale=1, width=5e-3,
               headwidth=10)
    plt.figure(2)
    plt.quiver(*Scan_ref[0, :], *vec_scan_ref, color=['k'], angles='xy', scale_units='xy', scale=1, width=3e-3,
               headwidth=10)

# Plot data expanded
a = False
if a:
    plt.figure(2)
    plt.scatter(map_points['ScanX(mm)'][:nb_pt], map_points['ScanY(mm)'][:nb_pt], c=map_points['PointID'][:nb_pt], cmap="jet")
    plt.plot(*Scan_ref[0, :], 'xr', *Scan_ref[1, :], 'xb')
    plt.title("with expansion")

# Find rotation angle
angle = -np.arccos(np.dot(vec_px_ref, vec_scan_ref)/(np.linalg.norm(vec_px_ref)*np.linalg.norm(vec_scan_ref)))
Rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

# Rotate all data points
Scan_ref[0, :] = np.matmul(Rot_mat, Scan_ref[0, :])
Scan_ref[1, :] = np.matmul(Rot_mat, Scan_ref[1, :])
PixelX = np.zeros((len(map_points['ScanX(mm)'])))
PixelY = np.zeros((len(map_points['ScanY(mm)'])))
for i in range(len(map_points['ScanX(mm)'])):
    PixelX[i], PixelY[i] = np.matmul(Rot_mat, np.array([map_points['ScanX(mm)'][i], map_points['ScanY(mm)'][i]]))

# Plot data rotated
a = False
if a:
    plt.figure(2)
    plt.scatter(PixelX[:nb_pt], PixelY[:nb_pt], c=map_points['PointID'][:nb_pt], cmap="jet", marker="o")
    plt.plot(*Scan_ref[0, :], 'xr', *Scan_ref[1, :], 'xb')
    plt.title("with expansion and rotation")

# Find translation
offset = Pixel_ref[0, :]-Scan_ref[0, :]
Scan_ref[0, :] += offset
Scan_ref[1, :] += offset
PixelX += offset[0]
PixelY += offset[1]

# Plot data translated
a = True
if a:
    plt.figure(1)
    plt.scatter(PixelX[:nb_pt], PixelY[:nb_pt], c=map_points['PointID'][:nb_pt], cmap="jet", marker="|")
    plt.plot(*Scan_ref[0, :], '|r', *Scan_ref[1, :], '|b')
    plt.title("with expansion, rotation and translation")

plt.show()
print()

# Loop on test files with missing .map

# Extract x and y positions

# Data to store
#map_points['PixelX'], map_points['PixelY'], map_points['PointID'], ScanX, ScanY
# ScanX, ScanY = unique (position x, position y) dans thickness ==> extract thickness nécessaire
#


