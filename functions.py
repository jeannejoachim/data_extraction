# Module imports
import os
import os.path
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def extract_data(file_path, prm):
    """
    Function to extract experimental data in the text file given as en input.
    Multiple data point are stored in one file, each between <DATA> and <END DATA>.

    :param file_path: chemin du fichier à traiter
    :param prm: parameters ........ <TO BE COMPLETED>
    :return: liste de données pour chaque point de mesure
    """
    data_list = []  # List to store data sets as dictionaries

    with open(file_path, 'r') as file:
        file_content = file.read()

        # Use regex to find matches between prm.begin_data and prm.end_data
        data_matches = re.finditer(repr(prm.begin_data + '(.*?)' + prm.end_data)[1:-1], file_content, re.DOTALL)
        # repr()[1:-1] converts to raw string after string concatenation

        for match in data_matches:
            data_block = match.group(1).strip()

            # Split the data block into lines and extract column names and values
            lines = data_block.split('\n')
            column_names = lines[0].split('\t')

            # Clean up column names
            for c in range(len(column_names)):
                if column_names[c] in prm.data_cleanup:
                    column_names[c] = prm.data_cleanup[column_names[c]]

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
            data_list.append(data_set)

    return data_list

def interpolate_data(x, y, z, prm):
    """
    Function to interpolate data for the thickness visualization.

    :param x: position de la sonde en x (coordonnée d'un point de la peau)
    :param y: position de la sonde en y (coordonnée d'un point de la peau)
    :param z: épaisseur calculée, extraite des coordonnées de la sonde
    :param nb: number of interpolation points (smoothness of final surface)
    :return: xi, yi, zi: interpolated data
    """
    # Create a regular grid
    xi, yi = np.meshgrid(np.linspace(min(x), max(x), prm.nb_interp), np.linspace(min(y), max(y), prm.nb_interp))

    # Interpolate the values
    zi = griddata((x, y), z, (xi, yi), method='cubic')

    return xi, yi, zi