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


def extract_data(file_path, begin_data, end_data):
    """
    Function to extract experimental data in the text file given as en input.
    Multiple data point are stored in one file, each between <DATA> and <END DATA>.

    :param file_path: chemin du fichier à traiter
    :param begin_data: string after which data begin
    :param end_data: string before which data end
    :return: data_points_list: list of time, positions and Fz for each point
    """
    data_points_list = []  # List to store data sets as dictionaries

    # Dictionary that defines initial data names (key) vs. cleaned-up data name (value)
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


def extract_thickness(data_points_list):
    """
    Function to extract thickness and positions from raw data_points_list and convert them to arrays.

    :param data_points_list: data as extracted by extract_data
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

def generate_map(data_points_list, ratio, Rot_mat, offset):
    """
    Function to generate missing map file, used to translate sensor coordinates to pixel (picture) coordinates.

    :param data_points_list: data as extracted by extract_data
    :param ratio: expansion ratio between sensor and pixel coordinates
    :param Rot_mat: rotation matrix between sensor and pixel coordinates
    :param offset: translation value between sensor and pixel coordinates
    :return: map_points: dictionary of x and y values in both sensor and pixel coordinates
    """

    # Create map points dictionary results
    map_points = {'PixelX': np.zeros((len(data_points_list))),
                  'PixelY': np.zeros((len(data_points_list))),
                  'PointID': np.zeros((len(data_points_list))),
                  'ScanX(mm)': np.zeros((len(data_points_list))),
                  'ScanY(mm)': np.zeros((len(data_points_list)))}

    # Extract position in sensor coordinates
    for i in range(len(data_points_list)):
        # Define PointID
        map_points['PointID'][i] = i + 1

        # Extract x-position
        if np.all(data_points_list[i]['pos_x']):
            map_points['ScanX(mm)'][i] = data_points_list[i]['pos_x'][0]
        else:
            # If `pos_x` recorded for the test is not the same for all times
            warnings.warn("WARNING x position change for point: " + str(i) + "in data:" + reference_data_file + "\n" +
                          " Taking average value rounded at 1e-6.")
            map_points['ScanX(mm)'][i] = np.round(np.average(data_points_list[i]['pos_x']), 6)

        # Extract y-position
        if np.all(data_points_list[i]['pos_y']):
            map_points['ScanY(mm)'][i] = data_points_list[i]['pos_y'][0]
        else:
            # If `pos_x` recorded for the test is not the same for all times
            warnings.warn("WARNING y position change for point: " + str(i) + "in data:" + reference_data_file + "\n" +
                          " Taking average value rounded at 1e-6.")
            map_points['ScanY(mm)'][i] = np.round(np.average(data_points_list[i]['pos_y']), 6)

    # X-flip scan reference data
    x_coordinate_adapted = -map_points['ScanX(mm)']

    # Apply ratio
    x_coordinate_adapted = x_coordinate_adapted * ratio
    y_coordinate_adapted = map_points['ScanY(mm)'] * ratio

    # Rotate data points
    map_points['PixelX'] = np.zeros((len(map_points['ScanX(mm)'])))
    map_points['PixelY'] = np.zeros((len(map_points['ScanY(mm)'])))
    for i in range(len(map_points['ScanX(mm)'])):
        map_points['PixelX'][i], map_points['PixelY'][i] = np.round(
            np.matmul(Rot_mat, np.array([x_coordinate_adapted[i], y_coordinate_adapted[i]])), 0)

    # Translate data points
    map_points['PixelX'] += offset[0]
    map_points['PixelY'] += offset[1]

    # Round to integer
    map_points['PixelX'] = np.round(map_points['PixelX'], 0)
    map_points['PixelY'] = np.round(map_points['PixelY'], 0)

    return map_points

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


def extract_young_modulus(data_points_list, thickness, prm_indentation):
    """
    Function to extract the Young modulus from Fz fitting.

    :param data_points_list: data extracted from indentation experiments
    :param thickness: data extracted from thickness experiments
    :param prm_indentation: thickness parameter object, containing namely:
        * fit_start_Fz_value : start value for the Fz fit
        * fit_stop_thickness_percent : stop value for the Fz fit
        * radius : sensor radius, in mm
        * nu : Poisson coefficient value
        * gravity : standard acceleration of gravity, in m/s², used to convert gf/mm² in kPa
    :return: result_indentation: dictionary containing fit results for all measurement points
    """

    # Initialization (note: lists necessary for pos_z_fit, Fz_fit and res_fit as the number of points to consider on
    # the Fz curve is not the same from one point to the other)
    result_indentation = {'E_fit': np.zeros(len(data_points_list)),
                        'Fz_ini': np.zeros(len(data_points_list)),
                        'corr_coeff': np.zeros(len(data_points_list)),
                        'pos_z_fit': [],
                        'Fz_fit': [],
                        'res_fit': []}

    for i in range(len(data_points_list)):
        # Data
        pos_z = data_points_list[i]["pos_z"]
        Fz = data_points_list[i]["Fz"]

        # Fit range up until fit_stop_thickness_percent
        thickness_limit = thickness[i]*(1-prm_indentation.fit_stop_thickness_percent/100)
        ind_z_max = np.max(np.where(pos_z < -thickness_limit))
        pos_z_fit = pos_z[:ind_z_max]
        Fz_fit = Fz[:ind_z_max]

        # Fit range starting after the first pike + Fz above given value
        ind_Fz_positive = np.max(np.where(Fz_fit < 0)) + 1
        if ind_Fz_positive == len(Fz_fit):
            result_indentation['E_fit'][i] = np.nan
            result_indentation['Fz_ini'][i] = np.nan
            result_indentation['corr_coeff'][i] = np.nan
            result_indentation['pos_z_fit'].append(pos_z_fit)
            result_indentation['Fz_fit'].append(Fz_fit)
            result_indentation['res_fit'].append(np.nan)
            continue
        ind_Fz_min = np.min(np.where(Fz_fit[ind_Fz_positive:] > prm_indentation.fit_start_Fz_value))
        Fz_fit = Fz_fit[ind_Fz_positive:][ind_Fz_min:]
        pos_z_fit = pos_z_fit[ind_Fz_positive:][ind_Fz_min:]

        # Least square fit with some parameters (R, nu) given
        fitfunc = partial(calculate_fz, R=prm_indentation.radius, nu=prm_indentation.nu)
        [E_fit, Fz_ini], _ = curve_fit(fitfunc, pos_z_fit-pos_z_fit[0], Fz_fit, bounds=(0, [2, 1]))

        res_fit = np.array([calculate_fz(z-pos_z_fit[0], E_fit, Fz_ini, prm_indentation.radius, prm_indentation.nu)
                            for z in pos_z_fit])
        E_fit = E_fit*prm_indentation.gravity
        corr_coeff = r2_score(Fz_fit, res_fit)

        # Store results
        result_indentation['E_fit'][i] = E_fit
        result_indentation['Fz_ini'][i] = Fz_ini
        result_indentation['corr_coeff'][i] = corr_coeff
        result_indentation['pos_z_fit'].append(pos_z_fit)
        result_indentation['Fz_fit'].append(Fz_fit)
        result_indentation['res_fit'].append(res_fit)

    return result_indentation


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
                  f.endswith((".jpg", ".jpeg", ".bmp", ".png"))]

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
        plt.scatter(vect_px_sorted_by_point_id[0, :], vect_px_sorted_by_point_id[1, :], c=thickness, s=2., cmap="jet",
                    vmin=thickness.min(), vmax=thickness.max())
    else:
        # Surface plot with more transparence
        ax.pcolormesh(xpx, ypx, zpx, shading='nearest', cmap="jet", alpha=prm_thickness.alpha/2, vmin=thickness.min(),
                      vmax=thickness.max())
        # Contour plot with inline labels
        CS = ax.contour(xpx, ypx, zpx, cmap="jet", vmin=thickness.min(), vmax=thickness.max())
        ax.clabel(CS, inline=True, inline_spacing=-2, fontsize=8)

    if prm_thickness.cropping_frame != 0:
        # Perform cropping if wanted
        plt.xlim(np.min(xpx)-prm_thickness.cropping_frame, np.max(xpx)+prm_thickness.cropping_frame)
        plt.ylim(np.min(ypx)-prm_thickness.cropping_frame, np.max(ypx)+prm_thickness.cropping_frame)

    plt.title(s)
    plt.axis('off')

    return fig


def plot_fz_curve_fit(s, i, data_point, result_indentation):
    """
    Function to plot Fz evolution with z position of the sensor and the curve fit used to extract the Young modulus.

    :param s: string, sample name
    :param i: integer, point ID
    :param data_point: indentation data results for the given point
    :param result_indentation: dictionary containing fit results for all measurement points
    :return: fig object, Fz graph for the given point
    """
    fig = plt.figure()

    plt.plot(data_point["pos_z"], data_point["Fz"], "--k", linewidth=0.5)
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
    plt.legend(["original data", "data used for the fit search",
                "curve fit for E=" + str(round(result_indentation['E_fit'][i], 2)) + " kPa (R² = " +
                str(round(result_indentation['corr_coeff'][i], 2)) + ")"])

    return fig
