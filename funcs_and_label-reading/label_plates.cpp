/**
 * \file label_plates.cpp
 * \author Gian-Mateo Tifone (mt9485@rit.edu)
 * \brief Function to label all the characters in the license plates
 * \version 1.4
 * \date 12-05-2024
 * 
 * @copyright Copyright (c) 2024
 */

#include "../knn_functions.h"

using namespace std;
namespace fs = std::filesystem;

int main(int argc, char* argv[]) {

    // ######################
    // %% Output Directory %%
    // ######################

    // Define the directory path 
    std::string directory_path = "../imgs/statistics/labeling/labeled_characters";

    // Check if the directory exists
    if (!fs::exists(directory_path)) {
        // Create the directory if it doesn't exist (along with parent directories)
        if (fs::create_directories(directory_path)) {
            std::cout << "Directory created: " << directory_path << std::endl;
        } else {
            std::cerr << "Failed to create directory: " << directory_path << std::endl;
            return -1;
        }
    } 

    // #######################################
    // %% Read in RGB Images from Directory %%
    // #######################################

    string plate_directory = "../imgs/statistics/labeling/license_plates";
    std::vector<cv::Mat> license_plates;
    std::vector<std::string> image_filenames;

    // Iterate through all files in the folder
    for (const auto& entry : fs::directory_iterator(plate_directory)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();

            // Attempt to load the image
            cv::Mat img = cv::imread(file_path, cv::IMREAD_GRAYSCALE);

            if (!img.empty()) {
                license_plates.push_back(img);
                image_filenames.push_back(file_path);
                std::cout << "Loaded: " << file_path << std::endl;
            } else {
                std::cerr << "Failed to load: " << file_path << std::endl;
            }
        }
    }


    // #######################################
    // %% Label characters and Remove Noise %%
    // #######################################

    int input_key;
    for (size_t i = 0; i < license_plates.size(); ++i) {
        cout << "Current plate #: " << i << endl;

        // Segment out characters to labeled
        cv::Mat plate = license_plates[i];
        vector<cv::Mat> segmented_characters = AutoExtractCharacters(plate);

        // Display the character image
        cv::namedWindow("License Plate", cv::WINDOW_NORMAL);
        cv::imshow("License Plate", plate);

        // Display all characters and let user annotate
        for (size_t j = 0; j < segmented_characters.size(); ++j){
            // Show character
            cv::imshow("character", segmented_characters[j]);

            // Read in user's keypress
            input_key = cv::waitKey(0);

            // Convert input key to ASCII character
            char character = static_cast<char>(input_key);

            // Add keypress to filename
             std::string dst_filename = "../imgs/statistics/labeling/labeled_characters/PlateLabel";
            dst_filename.append(1, character).append("_")
            .append(std::to_string(i)).append(std::to_string(j))
            .append(".png");
            
            // Write output image with the new filename
            if (character != ' ') cv::imwrite(dst_filename, segmented_characters[j]);
        }

    }

    // Close all OpenCV windows
    cv::destroyAllWindows();


    return EXIT_SUCCESS;
}



