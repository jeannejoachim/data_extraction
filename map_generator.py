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
from datetime import datetime
import logging

try:
    from functions import *
except:
    pass

#######################################################
# Data parameters
#######################################################
data_folder = r"C:\Users\Jeanne\Documents\Documents_communs\Laura_donnees\Fichiers pour générer .map manquants"
# folder path containing data tests for sample with missing .map
reference_map_subfolder = r"Reference"  # sub-folder containing reference map with the same reference points
test_name_separator = "_"    # string that separates sample ID to test name in the filename
sample_ID_position = 0    # 0 = sample ID is the first element given (so appears before the first file_name_separator)

# Reference coordinates (1st column = x, 2nd column = y) from map with same reference points
Pixel_ref = np.array([[506, 945],   # Reference1
                      [1097, 48]])  # Reference2
Scan_ref = np.array([[16.132000, -12.182000],   # Reference1
                     [-12.636000, 17.906000]])  # Reference2

# Keyword used for reference point. Multiple references will be saved as reference_point_keyword + reference number
reference_point_keyword = "Reference"

# Angle multiplier (1 or -1) because only the cosine of the angle is known (opposite angles have the same cosine)
angle_multiplier = -1   # change if verification with reference map fails

# Test from which to retrieve scan positions
data_file_keyword = "thickness"  # Keyword in filename for the test data
data_file_extension = ".txt"        # File extension for the test data
begin_data = "<DATA>"       # String after which the data point begins in test file
end_data = "<END DATA>"     # String before which the data point ends in test file

# Map file to be saved
map_file_extension = r".map"
begin_map = "MAPPING"


class logfile():
    verbose = True  # boolean parameter, when True messages appear in terminal
    log_file_path = os.path.join(data_folder, "0_map-generator-log_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +
                                 ".log")


prm = logfile()

# Create logging file
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y/%m/%d %H:%M:%S', encoding='utf-8',
                    handlers=[logging.FileHandler(os.path.join(data_folder, '0_map-generator-log.log'), mode='w'),
                              logging.StreamHandler(sys.stdout)],
                    level=logging.INFO)
logging.info("================================")
logging.info("Welcome to map-generator.py")
logging.info("================================")

#######################################################
# Find transformation parameters
#######################################################
logging.info("Computing transformation parameters from reference coordinates")
# X-flip scan reference data
Scan_ref_initial_values = np.copy(Scan_ref)
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
logging.info("Verifying on reference map in folder: " + os.path.join(data_folder, reference_map_subfolder))
reference_map_file = [f for f in os.listdir(os.path.join(data_folder, reference_map_subfolder))
                      if f.endswith(map_file_extension)]

if len(reference_map_file) > 1:
    logging.warning("There is more than one map file in the reference folder. Taking data from: " +
                    reference_map_file[0])
reference_map_file = reference_map_file[0]
map_points_true_value = extract_map(os.path.join(data_folder, reference_map_subfolder, reference_map_file),
                                    begin_map, reference_point_keyword)

# Extract data from test file
data_file = [f for f in os.listdir(os.path.join(data_folder, reference_map_subfolder)) if data_file_extension in f
             if data_file_keyword in f]
if len(data_file) > 1:
    logging.warning("There is more than one data file in the reference folder. Taking data from: " + data_file[0])
data_file = data_file[0]
data_points_list = extract_data_from_file(os.path.join(data_folder, reference_map_subfolder, data_file), begin_data,
                                          end_data)

# Generate map dictionary
map_points = generate_map(data_points_list, ratio, Rot_mat, offset,
                          os.path.join(data_folder, reference_map_subfolder, data_file), prm)

# Test comparison
test_x = np.array_equal(map_points_true_value['PixelX'], map_points['PixelX'])
test_y = np.array_equal(map_points_true_value['PixelY'], map_points['PixelY'])

if test_x and test_y:
    logging.info("Test passed!")
else:
    logging.critical("TEST FAILED! Try using: angle_multiplier = " + str(-1*angle_multiplier))
    raise ValueError

#######################################################
# Generate missing map files
#######################################################
logging.info("Generating missing maps")
data_files = [f for f in os.listdir(data_folder) if data_file_extension in f if data_file_keyword in f]

# Loop on data test files
for df in data_files:
    sample_ID = df.split(test_name_separator)[sample_ID_position]
    logging.info("[" + str(data_files.index(df) + 1) + "/" + str(len(data_files)) + "] " + "Treating sample " +
                 sample_ID)

    # If map file already exists: pass this iteration (security to avoid overwriting initial map)
    map_file = [f for f in os.listdir(data_folder) if map_file_extension in f if sample_ID in f]
    if len(map_file) > 0:
        logging.warning("Map file for this sample already exists " + str(map_file) + "! Sample skipped.")
        continue

    # Extract data from test file
    data_points_list = extract_data_from_file(os.path.join(data_folder, df), begin_data, end_data)

    # Generate map dictionary
    map_points = generate_map(data_points_list, ratio, Rot_mat, offset, os.path.join(data_folder, df), log_file_path,
                              verbose)

    # Save map to file
    with open(os.path.join(data_folder, sample_ID + "_map" + map_file_extension), "w") as f:
        # Write header info
        f.write(datetime.now().strftime("%B %d, %Y, %I:%M %p")+"\n")
        f.write("Map generated by map_generator.py, using map with same reference points: " + reference_map_file + "\n"
                "and scan positions from: " + df + "\n")
        f.write((begin_map+" ")*len(map_points.keys())+"\n")

        # Write column labels
        f.write("\t".join(map_points.keys()))

        # Write array values line by line, with \t delimiter
        for i in range(len(map_points['PixelX'])):
            f.write("\n")
            f.write("\t".join([str(map_points[k][i]) for k in map_points.keys()]))

        # Copy reference points at the end
        for i in range(Pixel_ref.shape[0]):
            f.write("\n")
            f.write("\t".join([str(e) for e in Pixel_ref[i, :]] + [reference_point_keyword+str(i+1)] +
                              [str(e) for e in Scan_ref_initial_values[i, :]]))

logging.info("============================== END ================================")
