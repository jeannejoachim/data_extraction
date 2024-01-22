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
from scipy.optimize import leastsq, curve_fit
from functools import partial
from sklearn.metrics import r2_score

try:
    from functions import *
except:
    pass

class parameters_general():
    # Data folder path, the 'r' prefix is important to convert the path in raw string!
    data_folder = r"C:\Users\Jeanne\Documents\Documents_communs\Laura_donnees\Donnees\BMC-006_19-10-23"
    filename_separator = "_"       # string that separates sample ID to test name in the filename
    filename_ID_position = 1
    # 1 = sample ID is the first element given (so appears before the first file_name_separator)
    show_plots = True
    sample_ID = ["C4_test1"] #C2-JAB9730", "E1-JAB9735", "B2-JAB9727", "I3-JAB9749"]
    treat_all_files_in_data_folder = False
    # if False, treatment is performed on given "sample_ID" list
    # if True, "sample_ID" is not used and sample ID is retrieved from all files in data_folder

class parameters_thickness():
    file_keyword = "thickness"   # Keyword in filename for this test
    file_format = ".txt"        # File format for this test
    nb_interp = 100     # Number of points for the data interpolation used to draw color maps
    begin_data = "<DATA>"      # String after which the data point begins
    end_data = "<END DATA>"    # String before which the data point ends
    plot_2D = False          # Plot and save in .png
    plot_on_picture = False
    plot_3D = False          # Plot and save in .png and .pickle for interactivity
    alpha = 0.5  # 0 = transparent, 1 = opaque (used for if plot_thickness_on_image = True)

class parameters_indentation():
    fit_start_Fz_value = 0.1            # Start of the fit for the given Fz value, first pike excluded
    fit_stop_thickness_percent = 10     # Stop of the fit for the given thickness percentage extracted on the same point
    file_keyword = "indentation"   # Keyword in filename for this test
    file_format = ".txt"        # File format for this test
    begin_data = "<DATA>"  # String after which the data point begins
    end_data = "<divider>"  # String before which the data point ends
    nu = 0.5        # Poisson coefficient value
    radius = 1   # indenter radius, in mm


prm = parameters_general()
prm_thickness = parameters_thickness()
prm_indentation = parameters_indentation()

#TODO récupérer tous les indices des échantillons dans le dossier => généraliser pour plusieurs essais
# if prm.treat_all_files_in_data_folder:
#     #TODO écraser sample_ID, utiliser ID_position
#     # Extraction of (skin) thickness
#     #f_thickness = [f for f in os.listdir(prm.data_folder) if (prm_thickness.file_keyword and prm_thickness.file_format) in f]
# else:
#     sample_ID = prm.sample_ID
sample_ID = prm.sample_ID

for s in sample_ID:
    ####################################################
    ## THICKNESS
    ####################################################
    file = s + prm.filename_separator + prm_thickness.file_keyword + prm_thickness.file_format
    # TODO Vérifier avec nom fichiers réels

    #TODO généraliser pour des points dans des fichiers différents
    # Pour un fichier, tous points de mesure

    # Extraction for each data point
    data_points_list = extract_data(os.path.join(prm.data_folder, file), prm_thickness)
    pos_x, pos_y, thickness = extract_thickness(data_points_list)
    xi, yi, zi = interpolate_data(pos_x, pos_y, thickness, prm_thickness)

    # 2D graph, sensor coordinates
    if prm_thickness.plot_2D:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pcm = ax.pcolormesh(xi, yi, zi, shading='nearest', cmap="jet",
                            vmin=thickness.min(), vmax=thickness.max())
        fig.colorbar(pcm, label='thickness')
        plt.scatter(pos_x, pos_y, c=thickness, ec='k', cmap="jet",
                    vmin=thickness.min(), vmax=thickness.max())
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(s + " - sensor coordinates")
        fig.savefig(os.path.join(prm.data_folder, s + "_thickness_2D.png"), bbox_inches='tight')
        #plt.close(fig)

    # 3D graph, sensor coordinates
    if prm_thickness.plot_3D:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xi, yi, zi, cmap='jet',
                               vmin=thickness.min(), vmax=thickness.max())
        fig.colorbar(surf, ax=ax, label='thickness')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('thickness')
        plt.title(s + " - sensor coordinates")
        fig.savefig(os.path.join(prm.data_folder, s + "_thickness_3D.png"), bbox_inches='tight')
        pickle.dump(fig, open(os.path.join(prm.data_folder, s + "_thickness_3D.fig.pickle"), 'wb'))
        #TODO (voir avec Laura): plot 3D above image?

    # Graph on pixel image
    if prm_thickness.plot_on_picture:
        # Load image
        ## TODO trouver le fichier image = extension jpg ou bmp ou png
        img = np.asarray(Image.open(os.path.join(prm.data_folder, s + ".bmp")))

        # Extract pixel map
        map_points = extract_map(os.path.join(prm.data_folder, s + "_map.map"), prm_thickness)

        # Sort map by point ID (security check)
        vect_px = np.array((map_points['PixelX'], map_points['PixelY'], map_points['PointID']))
        vect_px_sorted_by_point_id = vect_px[:, vect_px[2, :].argsort()]

        # Interpolate data on pixel map
        xpx, ypx, zpx = interpolate_data(vect_px_sorted_by_point_id[0, :], vect_px_sorted_by_point_id[1, :], thickness, prm_thickness)

        # 2D graph with pixel image coordinates
        fig = plt.figure()
        ax = fig.add_subplot(111)
        imgplot = plt.imshow(img)
        pcm = ax.pcolormesh(xpx, ypx, zpx, shading='nearest', cmap="jet", alpha=prm_thickness.alpha,
                            vmin=thickness.min(), vmax=thickness.max())
        fig.colorbar(pcm, label='thickness')
        plt.scatter(vect_px_sorted_by_point_id[0, :], vect_px_sorted_by_point_id[1, :], c=thickness, s=2., cmap="jet",
                    vmin=thickness.min(), vmax=thickness.max())
        plt.title(s)
        plt.axis('off')
        fig.savefig(os.path.join(prm.data_folder, s + "_thickness_on_image.png"), bbox_inches='tight')
        #plt.close(fig)

    #print([pos_x,pos_y,epaisseur])
    #print(thickness)
    #print("min = ", np.min(thickness), "max = ", np.max(thickness))

    ####################################################
    ## INDENTATION
    ####################################################
    # Pour un fichier, tous points de mesure
    f_indentation = [f for f in os.listdir(prm.data_folder) if (prm_indentation.file_keyword and prm_indentation.file_format) in f]
    file = s + prm.filename_separator + prm_indentation.file_keyword + prm_indentation.file_format

    # Extraction for each data point
    data_points_list = extract_data(os.path.join(prm.data_folder, file), prm_indentation)

    plt.figure()
    # for 1 point
    i = 0
    # Data
    time = data_points_list[i]["time"]
    pos_z = data_points_list[i]["pos_z"]
    Fz = data_points_list[i]["Fz"]

    # Fit range
    # Up until fit_stop_thickness_percent
    thickness_limit = thickness[i]*(1-prm_indentation.fit_stop_thickness_percent/100)
    ind_z_max = np.max(np.where(pos_z < -thickness_limit))
    pos_z_fit = pos_z[:ind_z_max]
    Fz_fit = Fz[:ind_z_max]
    # remove part where Fz negative (first pike unwanted)
    ind_Fz_positive = np.max(np.where(Fz_fit < 0)) + 1
    # consider Fz above given value
    ind_Fz_min = np.min(np.where(Fz_fit[ind_Fz_positive:] > prm_indentation.fit_start_Fz_value))
    Fz_fit = Fz_fit[ind_Fz_positive:][ind_Fz_min:]
    pos_z_fit = pos_z_fit[ind_Fz_positive:][ind_Fz_min:]

    # Least square fit
    #calculate_Fz
    #fitfunc = partial(calculate_Fz, R=prm_indentation.radius, nu=prm_indentation.nu)
    #coeffs, coeffs_cov = curve_fit(fitfunc, pos_z_fit, Fz_fit, bounds=(0, 1))

    popt, pcov = curve_fit(calculate_Fz, pos_z_fit-pos_z_fit[0], Fz_fit, bounds=(0, [2, 1]))
    res_fit = np.array([calculate_Fz(z-pos_z_fit[0], *popt) for z in pos_z_fit])
    E_fit = popt[0]*9.81
    E_log = 4.14
    print("erreur par rapport au logiciel : ", (E_fit-E_log)/E_log)
    print("correlation coefficient with Python: ", r2_score(Fz_fit, res_fit))

    # E = 4.14 kPa = 4.14/9.81 gf/mm2
    res_validation = np.array([calculate_Fz(z-pos_z_fit[0], 4.14/9.81, popt[1]) for z in pos_z_fit])
    print("correlation coefficient with Software: ", r2_score(Fz_fit, res_validation))
    # Graphs
    plt.plot(pos_z, Fz, "--k", linewidth=0.5)
    plt.plot(pos_z_fit, Fz_fit, "-k")
    plt.plot(pos_z_fit, res_validation, '--r')
    plt.plot(pos_z_fit, res_fit, ':b')
    plt.plot([pos_z_fit[0], pos_z_fit[0]], [np.min(Fz), np.max(Fz)],
             "-k", linewidth=0.5)
    plt.plot([pos_z_fit[-1], pos_z_fit[-1]], [np.min(Fz), np.max(Fz)], "-k", linewidth=0.5)
    plt.xlabel("pos_z [mm]")
    plt.ylabel("Fz [gF]")
    plt.title(s)
    plt.legend(["complete data", "data used for the fit",
                "fitted curve with software", "Python result"])

if prm.show_plots:
    plt.show()
