#include <iostream>
#include <vector> // std::vector
#include <opencv2/core.hpp>    // opencv module
#include <opencv2/highgui.hpp> // imshow
#include <opencv2/imgproc.hpp> // img processing funcitons
#include <boost/accumulators/accumulators.hpp> // Effeceint adding
#include <boost/accumulators/statistics/weighted_mean.hpp> // Average
#include <boost/accumulators/statistics/weighted_variance.hpp> // Weighted average

using namespace std;
namespace acc = boost::accumulators;

// Functions - written below main
void ColorReduce(const cv::Mat& src, cv::Mat& dst, const int& div=64);
std::vector<cv::Mat> auto_extract_characters(cv::Mat& license_plate, const bool& is_stored = true);

int main(int argc, char* argv[]) {
    // ######################################
    // %% Automatic Character Segmentation %%
    // ######################################

    // Read in image as grayscale, reduces data and processing
    cv::Mat src_image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    // Segment characters automatically
    std::vector<cv::Mat> characters_in_plate = auto_extract_characters(src_image);

    // Display and save segmented characters
    for (size_t i = 0; i < characters_in_plate.size(); ++i) {
        try {
        cv::imshow("Character " + std::to_string(i), characters_in_plate[i]);
        } 
        catch (...) { // Catch any errors in displaying empty matrices
            cerr << i << " is where we crashed" << endl;
        }

    }

    cv::waitKey(0);


}

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
std::vector<cv::Mat> auto_extract_characters(cv::Mat& license_plate, const bool& is_sorted) {
    // ###################
    // %% Preprocessing %%
    // ###################

    cv::Mat license_bilateral; // Bilateral does not support in-place operations
    cv::bilateralFilter(license_plate, license_bilateral, 9, 200, 200);
    license_plate = license_bilateral;
    cv::GaussianBlur(license_plate, license_plate, cv::Size(5, 5), 0); 
    // White text on a black background, findContours prefers it like this
    cv::threshold(license_plate, license_plate, 80, 255,  cv::THRESH_BINARY_INV);
    

    // ###################
    // %% Find Contours %%
    // ###################

    std::vector<std::vector<cv::Point>> contours;
    cv::Mat license_copy = license_plate.clone();


    cv::namedWindow("pre-processed", cv::WINDOW_NORMAL);
    cv::moveWindow("pre-processed", 600, 400);
    cv::imshow("pre-processed", license_copy);
    cv::waitKey(2);

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

        // A standard license plate is 1"x2.5625", so the height is its ratio
        // Find characters based on their ratios
        if (ratio > 2.0
         && ratio < 5.0
         && bounding_box.height > 25 
         && bounding_box.width > 25) {

            cv::Mat character = license_plate(bounding_box);
            characters.push_back(character); // For statistics

            // Resize character to 32x32 [px]
            cv::Mat resized_character;
            cv::resize(character, resized_character, cv::Size(32, 32));

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
    std:vector<cv::Mat> sorted_characters;
    for (const auto& pair : characters_with_x_coords) {
        sorted_characters.push_back(pair.first);
    }

    // Return list of characters
    return sorted_characters; // vector<Mat>
}

/**
 * \brief Displays original image followed by 
 * 
 * \param[in] character_list 
 * \return 
 */
std::vector<uchar> label_characters (std::vector<cv::Mat>& character_list) {
    // Vector to hold onto characters I've labeled
    for (auto& character : character_list) {

    }
}










