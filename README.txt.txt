# Project Title: License Plate Character Labeling and Recognition

GITHUB Link: https://github.com/mjw8601/IPCVFinalProject

## Overview
This project facilitates the extraction and labeling of characters from license plates using OpenCV for image processing and Boost for statistical calculations. The system segments characters and enables manual labeling for further machine learning applications.

## Project Structure
statistics/
|-- knn_functions.cpp       # KNN algorithm implementation and helper functions
|-- knn_functions.h         # Header file for KNN functions
|-- labeling/
    |-- CMakeLists.txt      # CMake configuration for compiling the labeling tool
    |-- label_plates.cpp    # Main application for labeling license plates
    |-- license_plates/     # Folder for storing input license plate images
    |-- labeled_characters/ # Automatically created folder for output labeled characters


## Prerequisites

1. **File Organization:**
   - Place `knn_functions.cpp` and `knn_functions.h` in the `statistics/` folder.
   - Create a `labeling` subfolder inside the `statistics/` folder.
   - Place `CMakeLists.txt` and `label_plates.cpp` in the `statistics/labeling/` folder.
   - Create a `license_plates` folder inside `statistics/labeling/` and populate it with license plate images.

## Compilation Instructions

1. Navigate to the `statistics` directory.
2. Edit the existing `CMakeLists.txt` in `statistics/` and add the following line:

add_subdirectory(labeling)


## Running the Application

1. After compilation, execute the application:
bin/label_plates

2. Follow the instructions displayed during the labeling process:
- Press the `spacebar` to skip and delete noise characters.
- The characters should appear in the left-to-right order as they are on the license plate.
- If an incorrect character is labeled, delete the generated file manually and redo the labeling.

3. In case of a crash or unexpected exit:
- Use the last labeled file name to identify the last processed plate.
- Restart the process from the appropriate point by deleting files for incomplete plates.

## Notes
- All output files will be saved in the 'labeled_characters' folder.
- If 'labeled_characters' is not created, the program will create and populate it

## Credits
- Authors: Cooper White (cjw9009@rit.edu)
 	   Gian-Mateo Tifone (mt9485@rit.edu)
           Mason Wahlers (mjw8601@rit.edu)
	   Stavros Viron (sv6393@rit.edu)
	   Luke Callahan (ljc9507@rit.edu)
- Version: 1.4
- Date: December 5, 2024

