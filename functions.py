# Module imports
import os
import os.path
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from functools import partial
from sklearn.metrics import r2_score
import logging


def extract_data(s, data_folder, file_keyword, file_extension, begin_data, end_data, prm, tol=1e-3):
    """
    Function to extract data for a given sample, and apply correction if needed.

    :param s:
    :param data_folder: 
    :param file_keyword: 
    :param file_extension: 
    :param begin_data: 
    :param end_data:
    :param prm: file parameters
    :param tol: 
    :return: data_points_list
    """
    
    # Gather all filenames
    sample_initial_thickness_file = [f for f in os.listdir(data_folder) if s in f
                                     if f.endswith(file_keyword + file_extension)][0]

    # Extract all data point from first test file
    data_points_list = extract_data_from_file(os.path.join(data_folder, sample_initial_thickness_file), 
                                              begin_data, end_data)

    # Apply corrections
    sample_correction_thickness_files = [f for f in os.listdir(data_folder) if s in f
                                         if file_keyword in f if f.endswith(file_extension)
                                         if sample_initial_thickness_file not in f]

    if len(sample_correction_thickness_files) != 0:
        logging.info("... Multiple test files found for this sample: launching data correction procedure")
        data_points_list = correct_data_points(data_points_list, data_folder,
                                               sample_initial_thickness_file, sample_correction_thickness_files,
                                               begin_data, end_data, prm, tol=tol)
    
    return data_points_list


def extract_data_from_file(file_path, begin_data, end_data):
    """
    Function to extract experimental data in the text file given as en input.
    Multiple data point are stored in one file, each between <DATA> and <END DATA>.

    :param file_path: chemin du fichier à traiter
    :param begin_data: string after which data begin in data files
    :param end_data: string before which data end in data files
    :return: data_points_list: list of time, positions and Fz for each point
    """
    data_points_list = []  # List to store data sets as dictionaries

    # Dictionary that defines initial data names (key) vs. cleaned-up data name (value) for future use in the code
    data_cleanup = {"Time, s": "time",
                    "Position (z), mm": "pos_z",
                    "Position (x), mm": "pos_x",
                    "Position (y), mm": "pos_y",
                    "Fz, gf": "Fz"}

    with open(file_path, 'r') as file:
        file_content = file.read()

        # Use regex to find matches between begin_data and end_data
        data_matches = re.finditer(repr(begin_data + '(.*?)' + end_data)[1:-1], file_content, re.DOTALL)
        # repr()[1:-1] converts to raw string after string concatenation

        for match in data_matches:
            data_block = match.group(1).strip()

            # Split the data block into lines and extract column names and values
            lines = data_block.split('\n')
            column_names = lines[0].split('\t')

            # Clean up column names
            for c in range(len(column_names)):
                if column_names[c] in data_cleanup:
                    column_names[c] = data_cleanup[column_names[c]]

            data_values = [line.split('\t') for line in lines[1:]]

            # Create a dictionary for each data set
            data_set = {column: [] for column in column_names}

            # Populate the dictionary with values
            for values in data_values:
                for i, column in enumerate(column_names):
                    data_set[column].append(float(values[i]))

            # Convert to array
            for k in data_set:
                data_set[k] = np.array(data_set[k])

            # Add the data set to the list
            data_points_list.append(data_set)

    return data_points_list


def correct_data_points(data_points_list, data_folder, sample_initial_test_file, sample_correction_files, begin_data,
                        end_data, prm, tol=1e-3):
    """
    Function used to correct data_points_list of the initial test data with correction data saved in correction files.

    :param data_points_list: data list as extracted by extract_data_from_file
    :param data_folder: data folder, containing initial and correction files
    :param sample_initial_test_file: test file containing data from the original test on this sample
    :param sample_correction_files: test file containing data from the test correction on this sample
    :param begin_data: string after which data begin in data files
    :param end_data: string before which data end in data files
    :param prm: file parameters
    :param tol: (optional) tolerance for the x and y coordinates matching, default is 1e-3
    :return: data_points_list, data list with corrections
    """

    # Get x and y sensor position for initial data
    xy_sensor_data = extract_xy_sensor(data_points_list, os.path.join(data_folder, sample_initial_test_file),
                                       prm)
    for i in range(len(sample_correction_files)):
        logging.info("... Correction with data from: " + sample_correction_files[i])

        # Extract all data point from secondary test files
        data_file_path = os.path.join(data_folder, sample_correction_files[i])
        data_points_list_correction = extract_data_from_file(data_file_path, begin_data, end_data)

        # Get x and y sensor position for correction data
        xy_sensor_corr = extract_xy_sensor(data_points_list_correction, data_file_path, prm)

        for j in range(len(data_points_list_correction)):
            # Find line of corresponding point in initial data
            pt_index = np.where((np.abs(xy_sensor_data[:, 0] - xy_sensor_corr[j, 0]) < tol) &   # column 0 = x
                                (np.abs(xy_sensor_data[:, 1] - xy_sensor_corr[j, 1]) < tol))    # column 1 = y

            if len(pt_index[0]) == 1:
                pt_index = pt_index[0][0]  # extract value from tuple result
            elif len(pt_index[0]) == 0:
                logging.critical("There is no coordinates in the initial data file (" + sample_initial_test_file +
                                 ") matching the point (x, y = " + str(xy_sensor_corr[j]) +
                                 ") extracted from the correction data file (" + sample_correction_files[i] + ").")
                logging.critical("Verify data or try to increase correction tolerance (tol = " + str(tol) + ").")
                raise ValueError
            else:
                logging.critical("There are multiple coordinates in the initial data file (" + sample_initial_test_file
                                 + ") matching the point (x, y = " + str(xy_sensor_corr[j]) +
                                 ") extracted from the correction data file (" + sample_correction_files[i] + ").")
                logging.critical("Verify data or try to increase correction tolerance (tol = " + str(tol) + ").")
                raise ValueError

            # Replace initial data with corrected data
            data_points_list[pt_index] = data_points_list_correction[j]
            logging.info("... point " + str(pt_index) + " replaced")

    return data_points_list


def calculate_thickness(data_points_list):
    """
    Function to calculate thickness and positions from raw data_points_list and convert them to arrays.

    :param data_points_list: data list as extracted by extract_data_from_file
    :return: result_thickness: dictionary containing arrays of positions and thickness for all measurement points
    """
    # Initialize output dictionary
    result_thickness = {'thickness': np.zeros(len(data_points_list)),
                        'pos_x': np.zeros(len(data_points_list)),
                        'pos_y': np.zeros(len(data_points_list))}

    # Fill output dictionary
    for i in range(len(data_points_list)):
        result_thickness['thickness'][i] = -np.max(data_points_list[i]["pos_z"])
        result_thickness['pos_x'][i] = np.mean(data_points_list[i]["pos_x"])
        result_thickness['pos_y'][i] = np.mean(data_points_list[i]["pos_y"])

    return result_thickness


def interpolate_data(x, y, z, nb_interp):
    """
    Function to interpolate data for the thickness visualization.

    :param x: x position from data
    :param y: y position from data
    :param z: z position from data
    :param nb_interp: number of interpolation points
    :return: xi, yi, zi: interpolated data
    """
    # Create a regular grid
    xi, yi = np.meshgrid(np.linspace(min(x), max(x), nb_interp),
                         np.linspace(min(y), max(y), nb_interp))

    # Interpolate the values
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    return xi, yi, zi


def extract_map(file_path, begin_map, end_map):
    """
    Function to extract map data from the text file given in input.
    Multiple data point are stored in one file, each between <DATA> and <END DATA>.

    :param file_path: path of the map file
    :param begin_map: string contained in the line before column name line (entire line will be removed)
    :param end_map: string contained in the line after points data (entire line will be removed)
    :return: map_points: array containing map values for each point
    """
    with open(file_path, 'r') as file:
        file_content = file.read()

        # Use regex to find matches between prm.start and prm.end
        data_matches = re.finditer(repr(begin_map + '(.*?)' + end_map)[1:-1], file_content, re.DOTALL)
        # repr()[1:-1] converts to raw string after string concatenation

        for match in data_matches:
            data_block = match.group(1).strip()

            # Split the data block into lines and extract column names and values
            lines = data_block.split('\n')[1:-1]  # remove first and last lines (= containing start and end strings)
            column_names = lines[0].split('\t')

            data_values = [line.split('\t') for line in lines[1:]]

            # Create a dictionary for each data set
            map_points = {column: [] for column in column_names}

            # Populate the dictionary with values
            for values in data_values:
                for i, column in enumerate(column_names):
                    if values[i] == "":
                        values[i] = 0
                    map_points[column].append(float(values[i]))

            # Convert to array
            for k in map_points:
                map_points[k] = np.array(map_points[k])

    return map_points


def generate_map(data_points_list, ratio, Rot_mat, offset, data_file_path, prm):
    """
    Function to generate missing map file, used to translate sensor coordinates to pixel (picture) coordinates.

    :param data_points_list: data list as extracted by extract_data_from_file
    :param ratio: expansion ratio between sensor and pixel coordinates
    :param Rot_mat: rotation matrix between sensor and pixel coordinates
    :param offset: translation value between sensor and pixel coordinates
    :param data_file_path: path to data file used in warning message
    :param prm: file parameters
    :return: map_points: dictionary of x and y values in both sensor and pixel coordinates
    """

    # Create map points dictionary results
    map_points = {'PixelX': np.zeros((len(data_points_list)), dtype=int),
                  'PixelY': np.zeros((len(data_points_list)), dtype=int),
                  'PointID': np.arange(len(data_points_list))+1,
                  'ScanX(mm)': np.zeros((len(data_points_list))),
                  'ScanY(mm)': np.zeros((len(data_points_list)))}

    # Extract position in sensor coordinates
    xy_sensor = extract_xy_sensor(data_points_list, data_file_path, prm)

    # Store in map_points
    map_points['ScanX(mm)'] = xy_sensor[:, 0]
    map_points['ScanY(mm)'] = xy_sensor[:, 1]

    # X-flip scan reference data
    x_coordinate_adapted = -map_points['ScanX(mm)']

    # Apply ratio
    x_coordinate_adapted = x_coordinate_adapted * ratio
    y_coordinate_adapted = map_points['ScanY(mm)'] * ratio

    # Rotate data points
    for i in range(len(map_points['ScanX(mm)'])):
        map_points['PixelX'][i], map_points['PixelY'][i] = np.round(
            np.matmul(Rot_mat, np.array([x_coordinate_adapted[i], y_coordinate_adapted[i]])), 0)

    # Translate data points
    map_points['PixelX'] += int(offset[0])
    map_points['PixelY'] += int(offset[1])

    return map_points


def extract_xy_sensor(data_points_list, data_file_path, prm, data_keys=['pos_x', 'pos_y']):
    """
    Function to extract given position (x or y) in data_points_list, with security in case position is not equal for all
    time steps.

    :param data_points_list: data list as extracted by extract_data_from_file
    :param data_file_path: path to data file used in warning message
    :param prm: file parameters
    :param data_keys: (optional) list of data x and y position keys in data_points_list.
    Default is ['pos_x', 'pos_y'], as defined in extract_data_from_file.
    :return: xy_sensor: array with extracted sensor position, column = [x, y]
    """
    xy_sensor = np.zeros([len(data_points_list), 2])

    for i in range(len(data_points_list)):
        # Loop on data_keys entries
        for k in data_keys:
            # Get corresponding column number for xy_sensor
            c = data_keys.index(k)
            if np.all(data_points_list[i][k]):
                # Take first time value if position recorded for the test is equal at all test times
                xy_sensor[i, c] = data_points_list[i][k][0]
            else:
                # Take average if position recorded for the test changes during test
                logging.warning("Position " + position + " is not equal for all time for point: " + str(i) +
                                "in data:" + data_file_path + ". Taking average value rounded at 1e-6.")
                xy_sensor[i, c] = np.round(np.average(data_points_list[i][k]), 6)

    return xy_sensor


def calculate_fz(z, E, Fini, R, nu):
    """
    Function to calculate Fz by equation.

    :param z: penetration, in mm (given by data)
    :param E: Young modulus, determined by data fit
    :param Fini: force at the beginning of the fit range, determined by data fit
    :param R: indenter radius (given in parameters)
    :param nu: Poisson coefficient (given in parameters)
    :return: Fz (force) value, used to fit experimental data
    """
    return (4/3)*(np.sqrt(R)/(1-nu**2))*E*(np.abs(z)**(3/2)) + Fini


def skip_indentation_point(i, result_indentation, prm):
    """
    Skip point if indentation test failed

    :param i: point ID
    :param result_indentation: initial dictionary
    :param prm: file parameters
    :return: result_indentation: updated dictionary
    """
    result_indentation['E_fit'][i] = np.nan
    result_indentation['Fz_0_fit'][i] = np.nan
    result_indentation['corr_coeff'][i] = np.nan
    result_indentation['pos_z_fit'].append([])
    result_indentation['Fz_fit'].append([])
    result_indentation['res_fit'].append(np.nan)
    logging.warning("Indentation test failed for data point:" + str(i + 1) +
                    ". Skipping this point in Young modulus extraction")

    return result_indentation


def calculate_young_modulus(data_points_list, thickness, prm_indentation, prm):
    """
    Function to calculate the Young modulus from Fz fitting.

    :param data_points_list: data extracted from indentation experiments
    :param thickness: data extracted from thickness experiments
    :param prm_indentation: thickness parameter object, containing namely:
        * fit_start_thickness_percent : start value for the Fz fit
        * fit_range_thickness_percent : range value for the Fz fit
        * radius : sensor radius, in mm
        * nu : Poisson coefficient value
        * gravity : standard acceleration of gravity, in m/s², used to convert gf/mm² in kPa
    :param prm: file parameters
    :return: result_indentation: dictionary containing fit results for all measurement points
        * E_fit : array of Young modulus for each point
        * Fz_0_fit : array of Fz origin on the fitting range for each point
        * corr_coeff : array of correlation coefficients for each point
    """

    # Initialization (note: pos_z_fit, Fz_fit and res_fit are lists as the number of points to consider on the Fz curve
    # is not the same from one point to the other)
    result_indentation = {'E_fit': np.zeros(len(data_points_list)),
                        'Fz_0_fit': np.zeros(len(data_points_list)),
                        'corr_coeff': np.zeros(len(data_points_list)),
                        'pos_z_fit': [],
                        'Fz_fit': [],
                        'res_fit': []}

    for i in range(len(data_points_list)):
        # Data
        pos_z = data_points_list[i]["pos_z"]
        Fz = data_points_list[i]["Fz"]

        # Fit range from fit_start_thickness_percent
        thickness_min = thickness[i] * (1 - prm_indentation.fit_start_thickness_percent / 100)
        ind_z_min = np.min(np.where(pos_z > -thickness_min))
        pos_z_fit = pos_z[ind_z_min:]
        Fz_fit = Fz[ind_z_min:]

        # Security check: fit range excludes first contact pike (Fz overshot + Fz below 0 for a few points)
        ind_Fz_negative = np.where(Fz_fit < 0)[0]

        if ind_Fz_negative.size > 0:
            ind_Fz_min = np.max(ind_Fz_negative) + 1
            if ind_Fz_min == len(Fz_fit):
                # Skip this point: there is no positive value for Fz, so test failed
                result_indentation = skip_indentation_point(i, result_indentation, prm)
                continue
            # Remove negative values from fit
            Fz_fit = Fz_fit[ind_Fz_min:]
            pos_z_fit = pos_z_fit[ind_Fz_min:]

        # Stopping point with thickness range
        pos_z_max = pos_z_fit[0] + thickness[i] * prm_indentation.fit_range_thickness_percent / 100

        if pos_z_fit[-1] < pos_z_max:
            # Skip this point: test stopped before reaching pos_z_max, so test failed
            result_indentation = skip_indentation_point(i, result_indentation, prm)
            continue

        ind_z_max = np.max(np.where(pos_z_fit < pos_z_max))
        pos_z_fit = pos_z_fit[:ind_z_max]
        Fz_fit = Fz_fit[:ind_z_max]

        # Least square fit with some parameters (R, nu) given
        fitfunc = partial(calculate_fz, R=prm_indentation.radius, nu=prm_indentation.nu)
        [E_fit, Fz_0_fit], _ = curve_fit(fitfunc, pos_z_fit-pos_z_fit[0], Fz_fit, method='lm')

        res_fit = np.array([calculate_fz(z-pos_z_fit[0], E_fit, Fz_0_fit, prm_indentation.radius, prm_indentation.nu)
                            for z in pos_z_fit])
        E_fit = E_fit*prm_indentation.gravity
        corr_coeff = r2_score(Fz_fit, res_fit)

        # Store results
        result_indentation['E_fit'][i] = E_fit
        result_indentation['Fz_0_fit'][i] = Fz_0_fit
        result_indentation['corr_coeff'][i] = corr_coeff
        result_indentation['pos_z_fit'].append(pos_z_fit)
        result_indentation['Fz_fit'].append(Fz_fit)
        result_indentation['res_fit'].append(res_fit)

    return result_indentation


def plot_thickness_z_curve(s, i, data_point, thickness_value):
    """

    :param s:
    :param i:
    :param data_point:
    :param thickness_value:
    :return:
    """
    fig = plt.figure()
    plt.plot([data_point['time'][0], data_point['time'][-1]], [thickness_value, thickness_value], '--b')
    plt.plot(data_point['time'], -data_point['pos_z'], 'k-')
    plt.ylabel("-(pos_z) [mm]")
    plt.xlabel("time [s]")
    plt.legend(["thickness value [mm] = " + str(round(thickness_value, 3))])
    plt.title(s + " - point ID " + str(i))

    return fig


def plot_thickness_2d(s, result_thickness):
    """
    Function to plot thickness interpolation surface on a 2D graph.

    :param s: string of the sample ID
    :param result_thickness: dictionary containing results extracted from thickness test, namely:
        * pos_x: x position extracted from data
        * pos_y: y position extracted from data
        * thickness: thickness extracted from data
        * x_interp: x position interpolated
        * y_interp: y position interpolated
        * z_interp: z position (thickness) interpolated
    :return: fig object, 2D graph for thickness at sensor coordinates
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pcm = ax.pcolormesh(result_thickness['x_interp'], result_thickness['y_interp'],
                        result_thickness['thickness_interp'], shading='nearest', cmap="jet",
                        vmin=result_thickness['thickness'].min(), vmax=result_thickness['thickness'].max())
    fig.colorbar(pcm, label='thickness')
    plt.scatter(result_thickness['pos_x'], result_thickness['pos_y'], c=result_thickness['thickness'], ec='k',
                cmap="jet", vmin=result_thickness['thickness'].min(), vmax=result_thickness['thickness'].max())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(s + " - sensor coordinates")

    return fig


def plot_thickness_3d(s, result_thickness):
    """
    Function to plot thickness interpolation surface on a 3D graph.

    :param s: ID of the sample treated
    :param result_thickness: dictionary containing results extracted from thickness test, namely:
        * thickness: thickness extracted from data
        * x_interp: x position interpolated
        * y_interp: y position interpolated
        * z_interp: z position (thickness) interpolated
    :return: fig object, 3D graph for thickness at sensor coordinates
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(result_thickness['x_interp'], result_thickness['y_interp'],
                           result_thickness['thickness_interp'], cmap='jet',
                           vmin=result_thickness['thickness'].min(), vmax=result_thickness['thickness'].max())
    fig.colorbar(surf, ax=ax, label='thickness')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('thickness')
    plt.title(s + " - sensor coordinates")

    return fig


def plot_thickness_on_picture(s, thickness, prm, prm_thickness, contour_plot=False):
    """
    Function to plot thickness interpolation 2D surface on the photo taken during experiments.

    :param s: ID of the sample treated
    :param thickness: thickness extracted from data
    :param prm: general parameters
    :param prm_thickness: thickness extraction parameters
    :param contour_plot: (optional) boolean that is True for contour plot, False for regular surface plot
    :return: fig object, thickness graph on pixel picture
    """
    # Load image
    # Find picture file: contains sample name and picture keyword, and ends with appropriate extension
    image_file = [f for f in os.listdir(prm.data_folder) if s in f and prm.picture_keyword in f and
                  f.endswith(prm.picture_extension)]
    if len(image_file) > 1:
        logging.warning("There is more than one image file with the same ID (" + s + "), Using image: " + image_file[0])
    image_file = image_file[0]

    img = np.asarray(Image.open(os.path.join(prm.data_folder, image_file)))

    # Extract map
    map_points = extract_map(os.path.join(prm.data_folder, s + "_map.map"),
                             prm_thickness.begin_map, prm_thickness.end_map)

    # Sort map by point ID (security check)
    vect_px = np.array((map_points['PixelX'], map_points['PixelY'], map_points['PointID']))
    vect_px_sorted_by_point_id = vect_px[:, vect_px[2, :].argsort()]

    # Interpolate data on pixel map
    xpx, ypx, zpx = interpolate_data(vect_px_sorted_by_point_id[0, :], vect_px_sorted_by_point_id[1, :], thickness,
                                     prm_thickness.nb_interp)

    # 2D graph with pixel image coordinates
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(img)
    if not contour_plot:
        pcm = ax.pcolormesh(xpx, ypx, zpx, shading='nearest', cmap="jet", alpha=prm_thickness.alpha, vmin=thickness.min(),
                      vmax=thickness.max())
        fig.colorbar(pcm, label='thickness')
        # Scatter plot of measurement points
        plt.scatter(vect_px_sorted_by_point_id[0, :], vect_px_sorted_by_point_id[1, :], s=2., c="k")#, #c="k", #thickness, #cmap="jet",
                    #vmin=thickness.min(), vmax=thickness.max())
    else:
        # Surface plot with more transparence
        ax.pcolormesh(xpx, ypx, zpx, shading='nearest', cmap="jet", alpha=prm_thickness.alpha/2, vmin=thickness.min(),
                      vmax=thickness.max())
        # Contour plot with inline labels
        CS = ax.contour(xpx, ypx, zpx, cmap="jet", vmin=thickness.min(), vmax=thickness.max())
        ax.clabel(CS, inline=True, inline_spacing=-2, fontsize=5)

    if prm_thickness.cropping_frame != 0:
        # Perform cropping if wanted
        plt.xlim(np.min(xpx)-prm_thickness.cropping_frame, np.max(xpx)+prm_thickness.cropping_frame)
        plt.ylim(np.min(ypx)-prm_thickness.cropping_frame, np.max(ypx)+prm_thickness.cropping_frame)

    plt.title(s)
    plt.axis('off')

    return fig


def plot_fz_curve_fit(s, i, data_point, result_indentation, fit_range_thickness_percent):
    """
    Function to plot Fz evolution with z position of the sensor and the curve fit used to extract the Young modulus.

    :param s: string, sample name
    :param i: integer, point ID
    :param data_point: indentation data results for the given point
    :param result_indentation: dictionary containing fit results for all measurement points
    :param fit_range_thickness_percent: thickness range used for the fit, used in the legend
    :return: fig object, Fz graph for the given point
    """
    fig = plt.figure()

    plt.plot(data_point["pos_z"], data_point["Fz"], "--k", linewidth=0.5)

    if len(result_indentation['pos_z_fit'][i]) != 0:
        # Plot fitting range and curve, if test has not failed
        plt.plot(result_indentation['pos_z_fit'][i], result_indentation['Fz_fit'][i], "-k", linewidth=0.5)
        plt.plot(result_indentation['pos_z_fit'][i], result_indentation['res_fit'][i], '-b')
        plt.plot([result_indentation['pos_z_fit'][i][0], result_indentation['pos_z_fit'][i][0]],
                 [np.min(data_point["Fz"]), np.max(data_point["Fz"])],
                 "-k", linewidth=0.5)
        plt.plot([result_indentation['pos_z_fit'][i][-1], result_indentation['pos_z_fit'][i][-1]],
                 [np.min(data_point["Fz"]), np.max(data_point["Fz"])],
                 "-k", linewidth=0.5)

    plt.xlabel("pos_z [mm]")
    plt.ylabel("Fz [gF]")
    plt.title(s + " - point ID " + str(i+1))
    plt.legend(["original data", "data used for the fit search (" + str(fit_range_thickness_percent) +
                " % of point thickness)", "curve fit for E=" + str(round(result_indentation['E_fit'][i], 2)) +
                " kPa (R² = " + str(round(result_indentation['corr_coeff'][i], 2)) + ")"])

    return fig
