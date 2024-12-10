#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "imgs/statistics/data_readers/DataReaders.h"
#include "imgs/statistics/classifiers/Classifiers.h"
#include "imgs/statistics/evaluators/Evaluators.h"

int main() {
  std::vector<unsigned char> training_labels = statistics::ReadMnistLabels(
      "../data/images/misc/final/train-labels-28-ubyte");
  std::cout << training_labels.size() << " our training labels read"
            << std::endl;

  std::vector<cv::Mat> training_images = statistics::ReadMnistImages(
      "../data/images/misc/final/train-images-28-ubyte");
  std::cout << training_images.size() << " our training images read"
            << std::endl;

  std::vector<unsigned char> test_labels = statistics::ReadMnistLabels(
      "../data/images/misc/final/test-labels-28-ubyte");
  std::cout << test_labels.size() << " our test labels read" << std::endl;

  std::vector<cv::Mat> test_images = statistics::ReadMnistImages(
      "../data/images/misc/final/test-images-28-ubyte");
  std::cout << test_images.size() << " our test images read" << std::endl;

  int k = 3;
  double p = 2;
  unsigned char ascii_offset_for_labels = 48;
  auto predicted_test_labels =
      statistics::Knn(test_images, training_images, training_labels, k, p);

  if (predicted_test_labels.size()) {
    std::cout << std::endl;
    std::cout << "For a k-NN classifier using " << k << " neighbors ";
    std::cout << "and a Minkowski distance of order " << p << std::endl;
    statistics::ConfusionMatrix(test_labels, predicted_test_labels,
                                ascii_offset_for_labels);
  }

  exit(EXIT_SUCCESS);
}