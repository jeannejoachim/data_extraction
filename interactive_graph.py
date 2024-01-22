#-----------------------------------------------------------------------------
# Script to read interactive graphs
#
# Dernière MAJ : 12-09-2023
#-----------------------------------------------------------------------------

# Module imports
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Graph folder path, the 'r' is important to convert the path in raw string!
graph_folder = r"C:\Users\Jeanne\Documents\Documents_communs\Laura_donnees\Donnees\Q1-JAB9772"
fig_name = "Q1-JAB9772_thickness_3D.fig.pickle"
#figx = pickle.load(open(os.path.join(graph_folder, fig_name), 'rb'))
plt.figure().add_subplot(111, projection='3d'),
figx = pickle.load(open(os.path.join(graph_folder, fig_name), 'rb'))

plt.show()
#input()