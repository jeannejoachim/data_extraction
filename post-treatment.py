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

try:
    from functions import *
except:
    pass

class parameters_general():
    # Data folder path, the 'r' prefix is important to convert the path in raw string!
    data_folder = r"C:\Users\Jeanne\Documents\Documents_communs\Laura_donnees\Fichiers pour générer .map manquants\Exemple"
    filename_separator = "_"       # string that separates sample ID to test name in the filename
    filename_ID_position = 1
    # 1 = sample ID is the first element given (so appears before the first file_name_separator)
    show_plots = True
    sample_ID = ["Z2-JAB9793"] #C2-JAB9730", "E1-JAB9735", "B2-JAB9727", "I3-JAB9749"]
    treat_all_files_in_data_folder = False
    # if False, treatment is performed on given "sample_ID" list
    # if True, "sample_ID" is not used and sample ID is retrieved from all files in data_folder

class parameters_thickness():
    plot_2D = True          # Plot and save in .png
    plot_on_picture = True
    plot_contour = True  # Plot and save in .png
    cropping_frame = 100  # Picture cropping (in pixels) around measurement points limits. 0 = no cropping
    plot_3D = True          # Plot and save in .png and .pickle for interactivity
    alpha = 0.5  # 0 = transparent, 1 = opaque (used for if plot_thickness_on_image = True)
    file_keyword = "thickness"   # Keyword in filename for this test
    file_format = ".txt"        # File format for this test
    nb_interp = 100     # Number of points for the data interpolation used to draw color maps
    begin_data = "<DATA>"      # String after which the data point begins
    end_data = "<END DATA>"    # String before which the data point ends
    begin_map = "MAPPING"  # String contained in the line before column name line (entire line will be removed)
    end_map = "Reference1"  # String contained in the line after points data (entire line will be removed)

class parameters_indentation():
    fit_start_Fz_value = 0            # Start value for the Fz fit = Fz value after the first pike
    fit_stop_thickness_percent = 10     # Stop value for the Fz fit = thickness percentage (at the same point)
    file_keyword = "indentation"   # Keyword in filename for this test
    file_format = ".txt"        # File format for this test
    begin_data = "<DATA>"  # String after which the data point begins
    end_data = "<divider>"  # String before which the data point ends
    nu = 0.5        # Poisson coefficient value
    radius = 1   # Sensor radius, in mm
    gravity = 9.80665   # Standard acceleration of gravity, in m/s², used to convert gf/mm² in kPa
    plot_fz_curve_fit = True


prm = parameters_general()
prm_thickness = parameters_thickness()
prm_indentation = parameters_indentation()

#TODO récupérer tous les indices des échantillons dans le dossier => généraliser pour plusieurs essais
# if prm.treat_all_files_in_data_folder:
#     #TODO écraser sample_ID, utiliser ID_position#
# else:
#     sample_ID = prm.sample_ID
sample_ID = prm.sample_ID

for s in sample_ID:
    print("Treating sample " + s + "...")
    ####################################################
    # THICKNESS
    ####################################################
    # f_thickness = [f for f in os.listdir(prm.data_folder) if (prm_thickness.file_keyword and prm_thickness.file_format) in f]
    # TODO généraliser avec plusieurs fichiers pour 1 sample
    file = s + prm.filename_separator + prm_thickness.file_keyword + prm_thickness.file_format
    # TODO Vérifier avec nom fichiers réels

    # TODO généraliser pour des points dans des fichiers différents
    # Pour un fichier, tous points de mesure

    # Extract all data point
    data_points_list = extract_data(os.path.join(prm.data_folder, file), prm_thickness)
    result_thickness = extract_thickness(data_points_list)
    result_thickness['x_interp'], result_thickness['y_interp'], result_thickness['thickness_interp'] = interpolate_data(
        result_thickness['pos_x'], result_thickness['pos_y'], result_thickness['thickness'], prm_thickness)

    # Plot graphs
    if prm_thickness.plot_2D:
        fig = plot_thickness_2d(s, result_thickness)
        fig.savefig(os.path.join(prm.data_folder, s + "_thickness_2D.png"), bbox_inches='tight')
        print("... thickness 2D plot saved...")
        if not prm.show_plots:
            plt.close(fig)
    if prm_thickness.plot_3D:
        fig = plot_thickness_3d(s, result_thickness)
        fig.savefig(os.path.join(prm.data_folder, s + "_thickness_3D.png"), bbox_inches='tight')
        pickle.dump(fig, open(os.path.join(prm.data_folder, s + "_thickness_3D.fig.pickle"), 'wb'))
        print("... thickness 3D plot saved...")
        if not prm.show_plots:
            plt.close(fig)
    if prm_thickness.plot_on_picture:
        fig = plot_thickness_on_picture(s, result_thickness['thickness'], prm, prm_thickness)
        fig.savefig(os.path.join(prm.data_folder, s + "_thickness_on_image.png"), bbox_inches='tight')
        print("... thickness plot on picture saved...")
        if not prm.show_plots:
            plt.close(fig)
    if prm_thickness.plot_contour:
        fig = plot_thickness_on_picture(s, result_thickness['thickness'], prm, prm_thickness, contour_plot=True)
        fig.savefig(os.path.join(prm.data_folder, s + "_thickness_on_image_contour.png"), bbox_inches='tight')
        print("... thickness contour plot on picture saved...")
        if not prm.show_plots:
            plt.close(fig)

    ####################################################
    # INDENTATION
    ####################################################
    # Pour un fichier, tous points de mesure
    # f_indentation = [f for f in os.listdir(prm.data_folder) if (prm_indentation.file_keyword and prm_indentation.file_format) in f]
    # TODO généraliser avec plusieurs fichiers pour 1 sample
    file = s + prm.filename_separator + prm_indentation.file_keyword + prm_indentation.file_format

    # Extract all data point
    data_points_list = extract_data(os.path.join(prm.data_folder, file), prm_indentation)
    result_indentation = extract_young_modulus(data_points_list, result_thickness['thickness'], prm_indentation)

    # Graphs
    if prm_indentation.plot_fz_curve_fit:
        for i in range(len(data_points_list)):
            fig = plot_fz_curve_fit(s, i, data_points_list[i], result_indentation)
            fig.savefig(os.path.join(prm.data_folder, s + "_fz_curve_fit_ID" + str(i+1) + ".png"), bbox_inches='tight')
            if not prm.show_plots:
                plt.close(fig)
        print("... fz curve fit saved...")

    ####################################################
    # Excel output
    ####################################################

    print("... sample finished!")

if prm.show_plots:
    plt.show()
