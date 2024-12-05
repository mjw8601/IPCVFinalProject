/**
 * \file knn_functions.h
 * \author Gian-Mateo Tifone (mt9485@rit.edu)
 * \brief Header file for functions used in KNN project
 * \version 1.0
 * \date 12-05-2024
 * 
 * @copyright I wrote this during finals week, steal at your own risk
 */

#pragma once

#include <iostream>
#include <vector> // std::vector
#include <opencv2/core.hpp>    // opencv module
#include <opencv2/highgui.hpp> // imshow
#include <opencv2/imgproc.hpp> // img processing funcitons
#include <boost/accumulators/accumulators.hpp> // Effeceint adding
#include <boost/accumulators/statistics/weighted_mean.hpp> // Average
#include <boost/accumulators/statistics/weighted_variance.hpp> // Weighted average

/**
 * \brief Reduces the color depth in an image by a factor of {div}
 * 
 * \param[in]  src Source image of original color depth
 * \param[out] dst Reduced color image
 * \param[in]  div 256 / div = Amount of colors to reduce image into. Default int 64 
 */
void ColorReduce(const cv::Mat& src, cv::Mat& dst, const int& div=64);

/**
 * \brief Automatically finds and extracts characters from license plates
 * 
 * \param[in] license_plate Grayscale license plate photo from parking lot / Grayscale license plate from isolate function
 * \param[in] is_sorted If true, returns characters sorted left to right based on x-coordinate in license_plate
 * \return std::vector<cv::Mat> Vectorized list of every 32x32 [px] character unlabeled
 */
std::vector<cv::Mat> AutoExtractCharacters(cv::Mat& license_plate, const bool& is_stored = true);

/**
 * \brief Quantizes an image to the requestest bit depth. E.g. If given a uchar (uint8_t) and 
 *        bit depth is 4, the resulting image will only have 4 bits of color depth (uint4_t)
 *        which is: 256 -> 16
 *        Caveat: only works with uchar (uint8_t) src images. Does support in-place operations
 * 
 * \param[in] src Source image to quantize
 * \param[in] bit_depth Resulting bit depth. Does not change data type of src. Default is 4.
 * \return cv::Mat Quantized image to requested bit depth
 */
void Quantize(cv::Mat& src, cv::Mat& dst, const int bit_depth = 4);

/**
 * \brief Computes the distance between a test image and a training image for KNN comparisons
 * 
 * \param[in] test_image The unknown character image
 * \param[in] training_image The known training character image
 * \param[in] p Dimensionality of distance funciton. Default = 2 (Euclidian)
 */
double MinkowskiDistance(const cv::Mat& test_image, const cv::Mat& training_image, const int& p = 2);




