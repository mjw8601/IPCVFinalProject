- TO RUN THE FILE -

1) Create a folder "labeling" in the statistics directory
2) Put CMakeLists.txt and label_plates.cpp into the "labeling" folder
3) Create a folder "license_plates" inside of the "labeling" folder
4) Put all of your license plate images into the "license_plates" folder
5) Edit the "statistics/CMakeLists.txt", and add the line -> "add_subdirectory(labeling)"
6) Run: bin/label_plates
6.1) When labeling, if you encounter noise not in the original image
     press the 'spacebar' to delete it
6.2) The letters (should) appear left->right in the order they appear in the license plate
6.3) If you click the wrong character, there is no do-over script. Delete the file manually.


What your directory will look like:
imgs/statistics/labeling

Where "labeling" contains:
| labeling
- | license_plates
- - | *All the photos from Google Drive*
- | CMakeLists.txt
- | knn_functions.cpp
- | knn_functions.h
