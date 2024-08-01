Last update: 1st August 2024

######################################################

Welcome,

This folder contains multiple scripts used to compute the mechanical properties of tested skin samples. <ADD MORE INFO ABOUT TEST and/or publication>
* post-treatment.py: main script to treat the raw data given by the sensors. Parameters that users should change are located at the beginning of the script, after module imports (will be cha)
* functions.py: functions called by post-treatment.py. Users should NOT change anything in this file, but can consult the function headers. For programmers, comments can 
* map_generator.py: script to generate missing or broken .map, that are needed in post-treatment.py
* interactive_graph.py: small script used to open and manipulate interactive graphs

These scripts have been tested with PyCharm Community Edition (free version), on a Windows computer with a Python 3.9 interpreter.

######################################################

How to get started : <to be simplified>
1. open post-treatment.py in PyCharm or any other IDE you like
2. define your parameters under the classes (see comments in the script):
	2.1. ParametersFiles: folder path and general file parameters
	2.2. ParametersThickness: parameters for thickness results
	2.3 ParametersIndentation: parameters for indentation test results
3. run the script (in Pycharm: right click + Run)

Informations will be printed in the Python Console and in the log file.
######################################################

Outputs organization:
* Results will be organized in subfolders, by sample name, under the defined data_folder
* Log and excel Results files will be in the data_folder