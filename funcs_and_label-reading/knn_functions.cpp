/**
 * \file knn_functions.cpp
 * \author Gian-Mateo Tifone (mt9485@rit.edu)
 * \brief Implementation file for functions used in KNN project
 * \version 1.2
 * \date 12-05-2024
 * 
 * @copyright Copyright (c) 2024
 */

#include "knn_functions.h"

using namespace std;
namespace acc = boost::accumulators;

/**
 * \brief Reduces the color depth in an image by a factor of {div}
 * 
 * \param[in]  src Source image of original color depth
 * \param[out] dst Reduced color image
 * \param[in]  div 256 / div = Amount of colors to reduce image into. Default int 64 
 */
void ColorReduce(const cv::Mat& src, cv::Mat& dst, const int& div)
{    
    if (dst.empty()) {
        dst.create(src.size(), src.type());
    }

    int nl = src.rows;                  // number of lines
    int nc = src.cols * src.channels(); // number of elements per line

    for (int j = 0; j < nl; j++)
    {
        // get the address of row j
        const uchar* src_ptr = src.ptr<uchar>(j);
        uchar* dst_ptr = dst.ptr<uchar>(j);

        for (int i = 0; i < nc; i++)
        {
            // process each pixel
            dst_ptr[i] = src_ptr[i] / div * div + div / 2;
        }
    }
}

/**
 * \brief Automatically finds and extracts characters from license plates
 * 
 * \param[in] license_plate Grayscale license plate photo from parking lot / Grayscale license plate from isolate function
 * \param[in] is_sorted If true, returns characters sorted left to right based on x-coordinate in license_plate
 * \return std::vector<cv::Mat> Vectorized list of every 32x32 [px] character unlabeled
 */
std::vector<cv::Mat> AutoExtractCharacters(cv::Mat& license_plate, const bool& is_sorted) {
    // ###################
    // %% Preprocessing %%
    // ###################

    license_plate -= 50;

    cv::Mat license_bilateral;
    cv::bilateralFilter(license_plate, license_bilateral, 9, 200, 200);
    license_plate = license_bilateral;
    cv::GaussianBlur(license_plate, license_plate, cv::Size(5, 5), 0); 

    // White text on a black background, findContours prefers it like this
    cv::threshold(license_plate, license_plate, 100, 255,  cv::THRESH_BINARY_INV);
    

    // ###################
    // %% Find Contours %%
    // ###################

    std::vector<std::vector<cv::Point>> contours;
    cv::Mat license_copy = license_plate.clone();

    // Find contours of the binary image
    // RETR_TREE, RETR_LIST, RETR_CCOMP
    cv::findContours(license_copy, contours, cv::RETR_TREE , cv::CHAIN_APPROX_NONE);


    // ########################
    // %% Extract Characters %%
    // ########################

    // Vectors to hold characters
    std::vector<cv::Mat> characters;
    std::vector<std::pair<cv::Mat, int>> characters_with_x_coords;

    // Accumulator to do statistics
    acc::accumulator_set<double, acc::features<acc::tag::weighted_mean, acc::tag::weighted_variance>, double> acc_set;

    for (const auto& contour : contours) {
        // Compute bounding box for each contour
        cv::Rect bounding_box = cv::boundingRect(contour);
        double ratio = (double) bounding_box.height/bounding_box.width;

        // A standard license plate is 1"x2.5625"
        // Find characters based on their ratios
        if (ratio > 2.0
         && ratio < 5.0
         && bounding_box.height > 25 
         && bounding_box.width > 25) {

            cv::Mat character = license_plate(bounding_box);
            characters.push_back(character); // For statistics

            // Resize character to 32x32 [px]
            cv::Mat resized_character;
            cv::resize(character, resized_character, cv::Size(28, 28));

            // Find whitepx (vs blackpx) to later sort out noisy contours
            double whitepx_value = cv::sum(character)[0];
            double weight = std::log(1 + whitepx_value); // Log weight

            // Add value and weight to the accumulator
            acc_set(whitepx_value, acc::weight = weight);

            // Add to the result vector
            // Pushback won't create new memory if it's already been allocated
            characters_with_x_coords.push_back({resized_character, bounding_box.x});
        }
    }


    // ################
    // %% Statistics %%
    // ################

    double weighted_mean = acc::weighted_mean(acc_set);
    double weighted_variance = acc::weighted_variance(acc_set);
    double weighted_std_dev = std::sqrt(weighted_variance);


    // #############################################
    // %% Filter out noise contours w/ Statistics %%
    // #############################################

    double threshold = weighted_mean - weighted_std_dev;

    auto it_characters = characters.begin();
    auto it_coords = characters_with_x_coords.begin();

    while (it_characters != characters.end() && it_coords != characters_with_x_coords.end()) {
        double whitepx_value = cv::sum(*it_characters)[0]; // Get the white pixel value for the current character

        // Check if the current character meets the threshold condition
        if (whitepx_value <= threshold) {
            // Remove from both vectors
            it_characters = characters.erase(it_characters);
            it_coords = characters_with_x_coords.erase(it_coords);
        } else {
            // Move to the next element in both vectors
            ++it_characters;
            ++it_coords;
        }
    }


    // #####################
    // %% Sort Characters %%
    // #####################

    if (is_sorted) {
        std::sort(characters_with_x_coords.begin(), characters_with_x_coords.end(),
                [](const std::pair<cv::Mat, int>& a, const std::pair<cv::Mat, int>& b) {
                    return a.second < b.second; // Sort by x-coordinate
                });
}

    // Extract only the sorted characters
    vector<cv::Mat> sorted_characters;
    for (const auto& pair : characters_with_x_coords) {
        sorted_characters.push_back(pair.first);
    }

    // Return list of characters
    return sorted_characters; // vector<Mat>
}

/**
 * \brief Quantizes an image to the requestest bit depth. E.g. If given a uchar (uint8_t) and 
 *        bit depth is 4, the resulting image will only have 4 bits of color depth (uint4_t)
 *        which is: 256 -> 16
 *        Caveat: only works with uchar (uint8_t) src images. Does support in-place operations
 * 
 * \param[in] src Source image to quantize
 * \param[in] bit_depth Resulting bit depth. Does not change data type of src.
 * \return cv::Mat Quantized image to requested bit depth
 */
void Quantize(cv::Mat& src, cv::Mat& dst, const int bit_depth) {

    // Create output image if dst is not given
    if (dst.empty()) {
        dst.create(src.size(), src.type());
    }

    // Maximum value (DC) new image can hold
    int max_value = (1 << bit_depth) - 1; // 2^bitDepth - 1
    int factor = 255 / max_value;         // Reduction factor

    // Pointer access to data
    uint8_t* src_data = src.data;
    uint8_t* dst_data = dst.data;
    size_t total_pixels = src.total();
    for (size_t i = 0; i < total_pixels; ++i) {
        dst_data[i] = (src_data[i] / factor) * factor;
    }
}

/**
 * \brief Computes the distance between a test image and a training image for KNN comparisons
 * 
 * \param[in] test_image The unknown character image
 * \param[in] training_image The known training character image
 * \param[in] p Dimensionality of distance funciton. Default = 2 (Euclidian)
 */
double MinkowskiDistance(const cv::Mat& test_image, const cv::Mat& training_image, const int& p) {
    double distance = 0;

    if (p == 1) // Manhattan distance
    { 
        return cv::norm(test_image, training_image, cv::NORM_L1);
    } else if (p == 2) // Euclidian distance
    { 
        return cv::norm(test_image, training_image, cv::NORM_L2);
    } else // Higher order
    { 
        // Pointers to begining cv::Mat location in memory
        // Methods works only because of grayscale's pointer stride
        const uchar* test_ptr = test_image.ptr<uchar>();
        const uchar* training_ptr = training_image.ptr<uchar>();
        int total_pixels = test_image.total();
            for (int i = 0; i < total_pixels; ++i) {
                distance += std::pow(std::abs(test_ptr[i] - training_ptr[i]), p);
            }
        return std::pow(distance, 1.0 / p);
    }
    

    }




