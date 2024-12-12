/* Displays steps involving the processing of license plates for the knn
 * ..rit/imgs/statistics/labeling/knn_show.cpp
 * Assumes label_plates.cpp has been run and rit/imgs/statistics/labeling is populated 
 */

/**
 * \file knn_show.cpp
 * \authors Luke & Gian-Mateo & Cooper
 * \brief 
 * \version 1.0
 * \date 12-11-2024
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#include "knn_functions.h"
#include <filesystem>
#include <iostream>
#include <vector>
#include <numeric> // Required for std::iota

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

Point2f getCentroid(const Rect& rect);
double calculateTotalDistance(const vector<Rect>& rectangles);
vector<Rect> findBestCluster(const vector<Rect>& rectangles, uchar clusterSize);


int main() {
  string licensePlatesPath     = "../imgs/statistics/labeling/license_plates";
  string labeledCharactersPath = "../imgs/statistics/labeling/labeled_characters";

    // #######################################
    // %% Read in RGB Images from Directory %%
    // #######################################

    string plate_directory = "../imgs/statistics/labeling/carl_plate";
    std::vector<cv::Mat> license_plates;
    std::vector<cv::Mat> color_plates;
    std::vector<std::string> image_filenames;

    // Iterate through all files in the folder
    for (const auto& entry : fs::directory_iterator(plate_directory)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();

            // Attempt to load the image
            cv::Mat img = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
            cv::Mat img_color = cv::imread(file_path, cv::IMREAD_COLOR);

            if (!img.empty()) {
                license_plates.push_back(img);
                color_plates.push_back(img_color);
                image_filenames.push_back(file_path);
                std::cout << "Loaded: " << file_path << std::endl;
            } else {
                std::cerr << "Failed to load: " << file_path << std::endl;
            }
        }
    }


    // ########################
    // %% Show auto contours %%
    // ########################

    for (size_t i = 0; i < license_plates.size(); ++i) {
        cout << "Current plate #: " << i << endl;

        // Rectangles to draw bounding boxes
        vector<cv::Rect> rectangles;

        // Segment out characters to labeled
        cv::Mat plate = license_plates[i];
        cv::Mat plate_color = color_plates[i];
        vector<cv::Mat> segmented_characters = AutoExtractCharacters(plate, rectangles);

        // Display the character image
        cv::namedWindow("License Plate", cv::WINDOW_NORMAL);
        cv::imshow("License Plate", plate);

        // Find the closest cluster of 7 rectangles
        vector<Rect> rect_cluster = findBestCluster(rectangles, 7);

        for (auto& rect : rect_cluster) {
          cv::rectangle(plate_color, rect, cv::Scalar(0, 0, 255), 25);
        }

      // Close all OpenCV windows
      cv::destroyAllWindows();

      // Let the people marvel in the glory of the plate
      cv::namedWindow("The PLATE", cv::WINDOW_NORMAL);
      cv::resizeWindow("The PLATE", 512, 512);
      cv::imshow("The PLATE", plate_color);
      cv::waitKey(0);

    } // For every plate

  // Close all OpenCV windows
  cv::destroyAllWindows();

  // Succeeded in giving best presentation :)
  return EXIT_SUCCESS;
}






// Function to calculate the centroid of a rectangle
Point2f getCentroid(const Rect& rect) {
    return Point2f(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);
}

// Function to calculate the total distance between a set of rectangles
double calculateTotalDistance(const vector<Rect>& rectangles) {
    double totalDistance = 0.0;
    vector<Point2f> centroids;

    // Calculate centroids
    for (const auto& rect : rectangles) {
        centroids.push_back(getCentroid(rect));
    }

    // Calculate total distance between all pairs of centroids
    for (size_t i = 0; i < centroids.size(); ++i) {
        for (size_t j = i + 1; j < centroids.size(); ++j) {
            totalDistance += norm(centroids[i] - centroids[j]);
        }
    }

    return totalDistance;
}

// Function to find the best cluster of 7 rectangles
vector<Rect> findBestCluster(const vector<Rect>& rectangles, uchar clusterSize) {
    vector<Rect> bestCluster;
    double minDistance = numeric_limits<double>::max();

    // Check all combinations of rectangles
    size_t n = rectangles.size();

    // Generate combinations of rectangles
    vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., n-1

    // Use a combination algorithm to find the best cluster
    vector<int> combination(clusterSize);
    for (size_t i = 0; i < n; ++i) {
        combination[0] = i;
        for (size_t j = i + 1; j < n; ++j) {
            combination[1] = j;
            for (size_t k = j + 1; k < n; ++k) {
                combination[2] = k;
                for (size_t l = k + 1; l < n; ++l) {
                    combination[3] = l;
                    for (size_t m = l + 1; m < n; ++m) {
                        combination[4] = m;
                        for (size_t o = m + 1; o < n; ++o) {
                            combination[5] = o;
                            for (size_t p = o + 1; p < n; ++p) {
                                combination[6] = p;

                                // Create a vector of rectangles for this combination
                                vector<Rect> currentCluster;
                                for (int index : combination) {
                                    currentCluster.push_back(rectangles[index]);
                                }

                                // Calculate the total distance for this cluster
                                double currentDistance = calculateTotalDistance(currentCluster);
                                if (currentDistance < minDistance) {
                                    minDistance = currentDistance;
                                    bestCluster = currentCluster;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return bestCluster;
}