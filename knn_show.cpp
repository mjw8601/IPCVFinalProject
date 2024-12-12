/* Displays steps involving the processing of license plates for the knn
 * ..rit/imgs/statistics/labeling/knn_show.cpp
 * Assumes label_plates.cpp has been run and rit/imgs/statistics/labeling is
 * populated */
#include "../knn_functions.h"
#include <filesystem>
#include <iostream>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

void processContours(const cv::Mat& image, cv::Mat& contourImage) {
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  cv::Mat binary;
  cv::threshold(gray, binary, 100, 255, cv::THRESH_BINARY);

  /* Finding the contours of the image and returning a drawing of the contours
   * on a copy of the original image. */
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(binary, contours, hierarchy, cv::RETR_TREE,
                   cv::CHAIN_APPROX_SIMPLE);

  /* Draw the contours and bounding boxes around relevant characters */
  contourImage = image.clone();
  cv::drawContours(contourImage, contours, -1, cv::Scalar(0, 255, 0), 2);
}
void processBoundingBoxes(const cv::Mat& image, cv::Mat& boxedImage) {
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  cv::Mat binary;
  cv::threshold(gray, binary, 100, 255, cv::THRESH_BINARY);

  boxedImage = image.clone();
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(binary, contours, hierarchy, cv::RETR_TREE,
                   cv::CHAIN_APPROX_SIMPLE);
  for (const auto& contour : contours) {
    /* Calculate the bounding box for the contours */
    cv::Rect boundingBox = cv::boundingRect(contour);

    /* Draw the bounding box on the image */
    cv::rectangle(boxedImage, boundingBox, cv::Scalar(0, 255, 0), 2);
  }
}

void displayFromDirectory(const std::string& folderPath,
                          const std::string& windowName, bool& showContours,
                          bool& showBoundingBoxes) {
  /* Iterating through all of the files in the folder */

  for (const auto& entry : fs::directory_iterator(folderPath)) {
    std::string filePath = entry.path().string();

    /* Loading the image */
    cv::Mat image = cv::imread(filePath);
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 512, 512);
    /* Display the image */
    cv::imshow(windowName, image);
    if (showBoundingBoxes) {
      cv::Mat boxedImage;
      processBoundingBoxes(image, boxedImage);
      cv::namedWindow("Bounding Box Viewer", cv::WINDOW_NORMAL);
      cv::resizeWindow("Bounding Box Viewer", 512, 512);
      cv::imshow("Bounding Box Viewer", boxedImage);
    }
    if (showContours) {
      cv::Mat contourImage;
      processContours(image, contourImage);
      cv::namedWindow("Contours Viewer", cv::WINDOW_NORMAL);
      cv::resizeWindow("Contours Viewer", 512, 512);
      cv::imshow("Contours Viewer", contourImage);
    }
    cout << "Press Spacebar to reveal next step" << endl;
    int key = cv::waitKey(0);
    if (key != 32) {
      cv::destroyWindow(windowName);
    }
  }

  cv::destroyWindow(windowName);
  if (showContours) cv::destroyWindow("Contours Viewer");
  if (showBoundingBoxes) cv::destroyWindow("Bounding Box Viewer");
}

int main() {
  string licensePlatesPath = "../imgs/statistics/labeling/license_plates";
  string labeledCharactersPath =
      "../imgs/statistics/labeling/labeled_characters";

  /* Calling the function for license plate and function for labeled
   * characters.
   */
  bool showContours = true;
  bool showBoundingBoxes = true;
  displayFromDirectory(licensePlatesPath, "License Plate Viewer",
                       showContours, showBoundingBoxes);
  displayFromDirectory(labeledCharactersPath, "Labeled Character Viewer",
                       showContours = false, showBoundingBoxes = false);
  return 0;
}