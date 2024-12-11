/** Implementation file for performing k-NN classification of a set of test
 *  images given a set of training images.
 *
 *  \file statistics/classifiers/Knn.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 22 Nov 2023
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>

#include <opencv2/core.hpp>

#include "imgs/statistics/classifiers/Knn.h"
#include "imgs/statistics/data_readers/Mnist.h"
#include "imgs/statistics/minkowski_distance/MinkowskiDistance.h"  // Include the MinkowskiDistance header


namespace statistics {

//flatten a cv::Mat image into a 1D vector of doubles.
std::vector<double> FlattenImage(const cv::Mat& image) {
  std::vector<double> flattened(image.rows * image.cols);
  for (int r = 0; r < image.rows; ++r) {
    for (int c = 0; c < image.cols; ++c) {
      flattened[r * image.cols + c] = static_cast<double>(image.at<uchar>(r, c));
    }
  }
  return flattened;
}

//k-NN Classifier implementation using Minkowski distance.
std::vector<unsigned char> Knn(
    const std::vector<cv::Mat>& test_images,
    const std::vector<cv::Mat>& training_images,
    const std::vector<unsigned char>& training_labels, const int k,
    const double p) {
  //vector to hold the predicted label for each test image
  std::vector<unsigned char> predicted_test_labels;


  //size matching
  if (training_images.size() != training_labels.size()) {
    std::cerr << "Training images and labels size mismatch!" << std::endl;
    return predicted_test_labels;
  }

  //test image processing
  for (const auto& test_image : test_images) {
    // Vector to store distances and their corresponding training labels
    std::vector<std::pair<double, unsigned char>> distances;

    //find distance from test to training
    for (size_t i = 0; i < training_images.size(); ++i) {
      // Use Cooper's MinkowskiDistance function
      double distance = MinkowskiDistance(test_image, training_images[i], static_cast<int>(p));
      distances.emplace_back(distance, training_labels[i]);
    }

    //find the k
    std::nth_element(distances.begin(), distances.begin() + k, distances.end(),
                     [](const auto& a, const auto& b) {
                       return a.first < b.first;
                     });

    //add the labels of the k
    std::map<unsigned char, int> label_counts;
    for (int i = 0; i < k; ++i) {
      ++label_counts[distances[i].second];
    }

    //find most common label
    unsigned char most_common_label = 0;
    int max_count = 0;
    for (const auto& [label, count] : label_counts) {
      if (count > max_count) {
        max_count = count;
        most_common_label = label;
      }
    }

    //append predicted label to results
    predicted_test_labels.push_back(most_common_label);
  }

  return predicted_test_labels;
}

} 

