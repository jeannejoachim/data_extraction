#-----------------------------------------------------------------------------
# Post-treatment script
# Test failures: number after test name giving priority (data from thickness (contain all points) erased by those in
# thickness2 (contain previously failed points only))
# File name example :
# Give results organization in subfolders
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
# Function import
from functions import *


class ParametersFiles:
    # Data folder path, the 'r' prefix is important to convert the path in raw string!
    data_folder = r"C:\Users\Jeanne\Documents\Documents_communs\Laura_donnees\Donnees\BMC-007\test"
    verbose = True          # if True, messages appear in terminal
    show_plots = False
    treat_all_files_in_data_folder = True
    # if False, treatment is performed on given "sample_ID" list (below)
    # if True, "sample_ID" is not used and sample ID is retrieved from all files in data_folder
    sample_ID = []
    picture_keyword = ""     # Keyword in filename for test picture
    picture_extension = (".bmp")  # Tuple of strings for possible extension of the test pictures
    # Test result name formatting. Example/ Test result name: E1-JAB9735_indentation
    # with E1-JAB9735 = sample_ID ; E1 = sample name ; indentation = test name
    # => test_name_separator = "_"
    # => sample_ID_position = 0
    # => sample_name_separator = "-"
    # => sample_name_position = 0
    test_name_separator = "_"       # string that separates sample ID to test name in the filename
    sample_ID_position = 0      # 0 = sample ID is the 1st element given (appears before the first file_name_separator)
    sample_name_separator = "-"  # string that separates sample name in sample ID
    sample_name_position = 0
    

class ParametersThickness:
    plot_z_curve = True     # Plot position z over time curve for each point for verification purposes
    plot_2D = True          # Plot and save in .png
    plot_on_picture = True
    plot_contour = True  # Plot and save in .png
    cropping_frame = 150  # Picture cropping (in pixels) around measurement points limits. 0 = no cropping
    plot_3D = True          # Plot and save in .png and .pickle for interactivity
    alpha = 0.5  # 0 = transparent, 1 = opaque (used for if plot_thickness_on_image = True)
    file_keyword = "thickness"   # Keyword in filename for this test
    file_extension = ".txt"        # File extension for this test
    nb_interp = 100     # Number of points for the data interpolation used to draw color maps
    begin_data = "<DATA>"      # String after which the data point begins
    end_data = "<END DATA>"    # String before which the data point ends
    begin_map = "MAPPING"  # String contained in the line before column name line (entire line will be removed)
    end_map = "Reference"  # String contained in the line after points data (entire line will be removed)


class ParametersIndentation():
    fit_start_thickness_percent = 1     # Start value for the Fz fit = thickness percentage (at the same point)
    fit_range_thickness_percent = 15    # Range (on thickness percentage) on which Fz fit is performed = linear domain
    # If sensor position values is less than thickness range wanted for fitting, the point is skipped (test failure)
    file_keyword = "indentation"   # Keyword in filename for this test
    file_extension = ".txt"        # File format for this test
    begin_data = "<DATA>"  # String after which the data point begins
    end_data = "<divider>"  # String before which the data point ends
    nu = 0.5        # Poisson coefficient value
    radius = 1   # Sensor radius, in mm
    gravity = 9.80665   # Standard acceleration of gravity, in m/s², used to convert gf/mm² in kPa
    plot_fz_curve_fit = True


# Pass parameters
prm = ParametersFiles
prm_thickness = ParametersThickness
prm_indentation = ParametersIndentation

# Create logging file
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y/%m/%d %H:%M:%S', encoding='utf-8',
                    handlers=[logging.FileHandler(os.path.join(prm.data_folder, '0_post-treatment-log.log'), mode='w'),
                              logging.StreamHandler(sys.stdout)],
                    level=logging.INFO)
logging.info("================================")
logging.info("Welcome to post-treatment.py")
logging.info("================================")

try:
    ####################################################
    # GET SAMPLES
    ####################################################
    if prm.treat_all_files_in_data_folder:
        all_thickness_files = [f for f in os.listdir(prm.data_folder) if prm_thickness.file_keyword in f
                               if f.endswith(prm_thickness.file_extension)]
        sample_ID = list(set([f.split(prm.test_name_separator)[prm.sample_ID_position] for f in all_thickness_files]))
        # set() conversion removes duplicate
    else:
        sample_ID = prm.sample_ID

    for s in sample_ID:
        logging.info("[" + str(sample_ID.index(s)+1) + "/" + str(len(sample_ID)) + "] " + "Treating sample " + s)
        # Folder management
        sample_name = s.split(prm.sample_name_separator)[prm.sample_name_position]
        sample_folder = os.path.join(prm.data_folder, sample_name)
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        raw_extraction_folder = os.path.join(sample_folder, "Raw_extraction")
        if not os.path.exists(raw_extraction_folder):
            os.makedirs(raw_extraction_folder)

        ####################################################
        # THICKNESS
        ####################################################
        logging.info("--------")
        logging.info("1. Thickness data post-treatment:")

        # Extract data
        logging.info("Extracting thickness data...")
        data_points_list = extract_data(s, prm.data_folder, prm_thickness.file_keyword, prm_thickness.file_extension,
                                        prm_thickness.begin_data, prm_thickness.end_data, prm, tol=1e-3)

        # Compute thickness and interpolate results
        logging.info("Computing thickness and interpolations")
        result_thickness = calculate_thickness(data_points_list)
        result_thickness['x_interp'], result_thickness['y_interp'], result_thickness['thickness_interp'] = interpolate_data(
            result_thickness['pos_x'], result_thickness['pos_y'], result_thickness['thickness'], prm_thickness.nb_interp)

        # Plot graphs
        if prm_thickness.plot_z_curve:
            logging.info("Generating z position plots for " + str(len(data_points_list)) + " points")
            for i in range(len(data_points_list)):
                fig = plot_thickness_z_curve(s, i, data_points_list[i], result_thickness['thickness'][i])
                fig.savefig(os.path.join(raw_extraction_folder, s + "_thickness_z_curve_ID" + str(i+1) + ".png"), bbox_inches='tight')
                if not prm.show_plots:
                    plt.close(fig)
        if prm_thickness.plot_2D:
            logging.info("Generating thickness 2D plot")
            fig = plot_thickness_2d(s, result_thickness)
            fig.savefig(os.path.join(sample_folder, s + "_thickness_2D.png"), bbox_inches='tight')
            if not prm.show_plots:
                plt.close(fig)
        if prm_thickness.plot_3D:
            logging.info("Generating thickness 3D plot")
            fig = plot_thickness_3d(s, result_thickness)
            fig.savefig(os.path.join(sample_folder, s + "_thickness_3D.png"), bbox_inches='tight')
            pickle.dump(fig, open(os.path.join(sample_folder, s + "_thickness_3D.fig.pickle"), 'wb'))
            if not prm.show_plots:
                plt.close(fig)
        if prm_thickness.plot_on_picture:
            logging.info("Generating thickness plot on picture")
            fig = plot_thickness_on_picture(s, result_thickness['thickness'], prm, prm_thickness)
            fig.savefig(os.path.join(sample_folder, s + "_thickness_on_image.png"), bbox_inches='tight')
            if not prm.show_plots:
                plt.close(fig)
        if prm_thickness.plot_contour:
            logging.info("Generating thickness contour plot on picture")
            fig = plot_thickness_on_picture(s, result_thickness['thickness'], prm, prm_thickness, contour_plot=True)
            fig.savefig(os.path.join(sample_folder, s + "_thickness_on_image_contour.png"), bbox_inches='tight')
            if not prm.show_plots:
                plt.close(fig)

        ####################################################
        # INDENTATION
        ####################################################
        logging.info("--------")
        logging.info("2. Indentation data post-treatment:")

        # Extract data
        logging.info("Extracting indentation data...")
        data_points_list = extract_data(s, prm.data_folder, prm_indentation.file_keyword, prm_indentation.file_extension,
                                        prm_indentation.begin_data, prm_indentation.end_data, prm, tol=1e-3)

        # Compute Young modulus from indentation curve fitting
        logging.info("Computing Young modulus with force data fitting...")
        result_indentation = calculate_young_modulus(data_points_list, result_thickness['thickness'], prm_indentation,
                                                     prm)

        # Plot graphs
        if prm_indentation.plot_fz_curve_fit:
            logging.info("Generating Fz curve fit for " + str(len(data_points_list)) + " points...")
            for i in range(len(data_points_list)):
                fig = plot_fz_curve_fit(s, i, data_points_list[i], result_indentation,
                                        prm_indentation.fit_range_thickness_percent)
                fig.savefig(os.path.join(raw_extraction_folder, s + "_fz_curve_fit_ID" + str(i+1) + ".png"), bbox_inches='tight')
                if not prm.show_plots:
                    plt.close(fig)

        ####################################################
        # Excel output
        ####################################################
        # TODO
        logging.info("Sample treated")
        logging.info("---------------------------------")

    if prm.show_plots:
        plt.show()

except Exception as e:
    logging.critical(str(type(e)) + ": " + str(e))
    raise

logging.info("============== END ==============")
