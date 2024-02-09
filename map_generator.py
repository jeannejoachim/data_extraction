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

data_folder = r"C:\Users\Jeanne\Documents\Documents_communs\Laura_donnees\Fichiers pour générer .map manquants"
# File path containing reference map with the same reference points as missing map tests
reference_file = r"C:\Users\Jeanne\Documents\Documents_communs\Laura_donnees\Fichiers pour générer .map manquants\Exemple\Z2-JAB9793_map.map"
start = "MAPPING	MAPPING	MAPPING	MAPPING	MAPPING	MAPPING	MAPPING"
end = "Reference1"
angle_multiplier = -1   #1 or -1

# Reference coordinates (1st column = x, 2nd column = y)
# To be gathered from test with same reference points
Pixel_ref = np.array([[506, 945],   # Reference1
                      [1097, 48]])  # Reference2
Scan_ref = np.array([[16.132000, -12.182000],   # Reference1
                     [-12.636000, 17.906000]])  # Reference2

#######################################################
# Find transformation parameters
#######################################################
# X-flip scan reference data
Scan_ref[:, 0] = -Scan_ref[:, 0]

# Create vector between reference points in both coordinates
vec_px_ref = np.array(Pixel_ref[1, :]-Pixel_ref[0, :])
vec_scan_ref = np.array(Scan_ref[1, :]-Scan_ref[0, :])

# Expansion ratio
ratio = np.linalg.norm(vec_px_ref)/np.linalg.norm(vec_scan_ref)
Scan_ref = Scan_ref * ratio

# Rotation angle and matrix
angle = angle_multiplier*np.arccos(np.dot(vec_px_ref, vec_scan_ref)/(np.linalg.norm(vec_px_ref)*np.linalg.norm(vec_scan_ref)))
Rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
Scan_ref[0, :] = np.matmul(Rot_mat, Scan_ref[0, :])
Scan_ref[1, :] = np.matmul(Rot_mat, Scan_ref[1, :])

# Translation
offset = Pixel_ref[0, :]-Scan_ref[0, :]
Scan_ref[0, :] += offset
Scan_ref[1, :] += offset

#######################################################
# Verify on reference map
#######################################################
# Extract map from reference
map_points = extract_map(reference_file, "MAPPING", "Reference1")

# X-flip scan reference data
map_points['ScanX(mm)'] = -map_points['ScanX(mm)']

# Apply ratio
map_points['ScanX(mm)'] = map_points['ScanX(mm)']*ratio
map_points['ScanY(mm)'] = map_points['ScanY(mm)']*ratio

# Rotate all data points
PixelX = np.zeros((len(map_points['ScanX(mm)'])))
PixelY = np.zeros((len(map_points['ScanY(mm)'])))
for i in range(len(map_points['ScanX(mm)'])):
    PixelX[i], PixelY[i] = np.matmul(Rot_mat, np.array([map_points['ScanX(mm)'][i], map_points['ScanY(mm)'][i]]))

# Translate all reference and data points
PixelX += offset[0]
PixelY += offset[1]

# Extract map from reference

#######################################################
#### OLD VERIFICATION PLOT
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
    plt.scatter(map_points['PixelX'], map_points['PixelY'], c=map_points['PointID'],
                cmap="jet", marker="x")
    plt.plot(*Pixel_ref[0, :], 'xr', *Pixel_ref[1, :], 'xb')

# Plot points by sensor coordinates
a = False
if a:
    plt.figure(2)
    plt.scatter(map_points['ScanX(mm)'], map_points['ScanY(mm)'], c=map_points['PointID'], cmap="jet")
    plt.plot(*Scan_ref[0, :], 'xr', *Scan_ref[1, :], 'xb')

# Plot points by sensor coordinates - X-flip
a = False
if a:
    plt.figure(2)
    plt.scatter(-map_points['ScanX(mm)'], map_points['ScanY(mm)'], c=map_points['PointID'], cmap="jet")
    plt.plot(*Scan_ref[0, :], 'xr', *Scan_ref[1, :], 'xb')
    plt.title("with x-flip")

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
    plt.scatter(map_points['ScanX(mm)'], map_points['ScanY(mm)'], c=map_points['PointID'], cmap="jet")
    plt.plot(*Scan_ref[0, :], 'xr', *Scan_ref[1, :], 'xb')
    plt.title("with expansion")

# Plot data rotated
a = False
if a:
    plt.figure(2)
    plt.scatter(PixelX, PixelY, c=map_points['PointID'], cmap="jet", marker="o")
    plt.plot(*Scan_ref[0, :], 'xr', *Scan_ref[1, :], 'xb')
    plt.title("with expansion and rotation")

# Plot data translated
a = True
if a:
    plt.figure(1)
    plt.scatter(PixelX, PixelY, c=map_points['PointID'], cmap="jet", marker="|")
    plt.plot(*Scan_ref[0, :], '|r', *Scan_ref[1, :], '|b')
    plt.title("with expansion, rotation and translation")

plt.show()
print()
#######################################################


# Loop on test files with missing .map

# Extract x and y positions

# Data to store
#map_points['PixelX'], map_points['PixelY'], map_points['PointID'], ScanX, ScanY
# ScanX, ScanY = unique (position x, position y) dans thickness ==> extract thickness nécessaire
#


