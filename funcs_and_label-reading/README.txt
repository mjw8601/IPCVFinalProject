- TO RUN THE FILE -
1) Put knn_functions.cpp/.h inside of the "statistics" directory
1) Create a folder "labeling" in the statistics directory
2) Put CMakeLists.txt and label_plates.cpp into the "labeling" folder
4) Create a folder "license_plates" inside of the "labeling" folder
5) Put all of your license plate images into the "license_plates" folder
6) Edit the "statistics/CMakeLists.txt", and add the line -> "add_subdirectory(labeling)"
7) Run: bin/label_plates
7.1) When labeling, if you encounter noise not in the original image press the 'spacebar'
      to delete it
7.2) The letters (should) appear left->right mimicking the order they appear on the license plate
7.3) If you click the wrong character, there is no do-over script. Delete the output file manually.
     |_ However, if you do quit/crash, the first number after the character in the last 
        output image represents which plate was being analyzed before the crash/quit. You 
        can delete all the plates up to that point and continue where you left off.


Directory Visual (of imgs/statistics/):
| knn_functions.cpp
| knn_functions.h
| labeling (folder)
- | license_plates (folder)
- - | *All the photos from Google Drive*
- | CMakeLists.txt
- | label_plates.cpp

