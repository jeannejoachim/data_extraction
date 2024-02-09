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
import warnings

try:
    from functions import *
except:
    pass

#######################################################
# Data parameters
#######################################################
data_folder = r"C:\Users\Jeanne\Documents\Documents_communs\Laura_donnees\Fichiers pour générer .map manquants"
# Folder path containing reference map with the same reference points as missing map tests
reference_map_subfolder = r"Reference"
filename_separator = "_"    # string that separates sample ID to test name in the filename
filename_ID_position = 1    # 1 = sample ID is the first element given (so appears before the first file_name_separator)

# Reference coordinates (1st column = x, 2nd column = y) from map with same reference points
Pixel_ref = np.array([[506, 945],   # Reference1
                      [1097, 48]])  # Reference2
Scan_ref = np.array([[16.132000, -12.182000],   # Reference1
                     [-12.636000, 17.906000]])  # Reference2
# Angle multiplier (1 or -1) because only the cosine of the angle is known (opposite angles have the same cosine)
angle_multiplier = -1   # change if verification with reference map fails

# Test from which to retrieve scan positions
data_file_keyword = "thickness"  # Keyword in filename for the test
data_file_format = ".txt"        # File format for the test
begin_data = "<DATA>"       # String after which the data point begins in test file
end_data = "<END DATA>"     # String before which the data point ends in test file

# Map file to be saved
map_file_extension = r".map"
begin_map = "MAPPING"

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
# Extract map from reference - for verification
reference_map_file = [f for f in os.listdir(os.path.join(data_folder, reference_map_subfolder))
                      if f.endswith(map_file_extension)]

if len(reference_map_file) > 1:
    warnings.warn("WARNING There is more than one map file in the reference folder. \n" +
                  "Taking data from: " + reference_map_file[0])
reference_map_file = reference_map_file[0]
map_points_true_value = extract_map(os.path.join(data_folder, reference_map_subfolder, reference_map_file),
                                    begin_map, "Reference1")

# Extract data from test file
reference_data_file = [f for f in os.listdir(os.path.join(data_folder, reference_map_subfolder))
                       if data_file_format in f if data_file_keyword in f]
if len(reference_data_file) > 1:
    warnings.warn("WARNING There is more than one data file in the reference folder. \n" +
                  "Taking data from: " + reference_data_file[0])
reference_data_file = reference_data_file[0]
data_points_list = extract_data(os.path.join(data_folder, reference_map_subfolder, reference_data_file),
                           begin_data, end_data)

# Generate map dictionary
map_points = generate_map(data_points_list, ratio, Rot_mat, offset)

# Test comparison
test_x = np.array_equal(map_points_true_value['PixelX'], map_points['PixelX'])
test_y = np.array_equal(map_points_true_value['PixelY'], map_points['PixelY'])

if test_x and test_y:
    print("... Test passed! Generating missing maps...")
else:
    raise ValueError("TEST FAILED! Try to using: angle_multiplier = " + str(-1*angle_multiplier))

# Generate map file
# A = np.array([map_points['PixelX'], map_points['PixelY'], map_points['PointID'],
#               map_points['ScanX(mm)'], map_points['ScanY(mm)']])
# np.savetxt(os.path.join(data_folder, reference_map_subfolder, "test.txt"), )

# End of map: copy Reference points from Reference map file

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
    plt.scatter(map_points_true_value['PixelX'], map_points_true_value['PixelY'], c=map_points_true_value['PointID'],
                cmap="jet", marker="x")
    plt.plot(*Pixel_ref[0, :], 'xr', *Pixel_ref[1, :], 'xb')

# Plot vectors
a = False
if a:
    plt.figure(1)
    plt.quiver(*Pixel_ref[0, :], *vec_px_ref, color=['k'], angles='xy', scale_units='xy', scale=1, width=5e-3,
               headwidth=10)
    plt.figure(2)
    plt.quiver(*Scan_ref[0, :], *vec_scan_ref, color=['k'], angles='xy', scale_units='xy', scale=1, width=3e-3,
               headwidth=10)

# Plot data rotated
a = False
if a:
    plt.figure(2)
    plt.scatter(map_points['PixelX'], map_points['PixelY'], c=map_points['PointID'], cmap="jet", marker="o")
    plt.plot(*Scan_ref[0, :], 'xr', *Scan_ref[1, :], 'xb')
    plt.title("with expansion and rotation")

# Plot data translated
a = True
if a:
    plt.figure(1)
    plt.scatter(map_points['PixelX'], map_points['PixelY'], c=map_points['PointID'], cmap="jet", marker="|")
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


