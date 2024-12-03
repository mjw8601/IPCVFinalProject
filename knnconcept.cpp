#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

// Functions - written below main
void ColorReduce(const cv::Mat& src, cv::Mat& dst, const int& div=64);
std::vector<cv::Mat> auto_extract_characters(cv::Mat& license_plate);

int main(int argc, char* argv[]) {
    // ##########################
    // ###### OTSU TESTING ######
    // ##########################

    // // Read in image - less data the better
    // cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

    // // Case 1: Just otsu's threshold
    // cv::Mat otsu, otsu_reduced, otsu_blur;
    // cv::threshold(src, otsu, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);

    // // Case 2: Otsu after color reduction
    // ColorReduce(src, otsu_reduced, 128);
    // cv::threshold(otsu_reduced, otsu_reduced, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);

    // // Case 3: Applied Gaussian blur 
    // cv::GaussianBlur(src, otsu_blur, cv::Size(5, 5), 0);
    // cv::threshold(otsu_blur, otsu_blur, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // cv::imshow("Source", src);
    // cv::imshow("Otsu + color reduced", otsu_reduced);
    // cv::imshow("otsu", otsu);
    // cv::imshow("otsu + blur", otsu_blur);
    // cv::waitKey(0);


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
 * \return std::vector<cv::Mat> Vectorized list of every 32x32 [px] character unlabeled
 */
std::vector<cv::Mat> auto_extract_characters(cv::Mat& license_plate) {
    // ###################
    // %% Preprocessing %%
    // ###################

    // Threshold image to increase contrast
    cv::GaussianBlur(license_plate, license_plate, cv::Size(5, 5), 0); // (Optional) Improves threshold results
    cv::threshold(license_plate, license_plate, 80, 255,  cv::THRESH_BINARY | cv::THRESH_OTSU);
    

    // ###################
    // %% Find Contours %%
    // ###################

    // White text on a black background, findContours prefers it like this
    cv::bitwise_not(license_plate, license_plate);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat license_copy = license_plate.clone();

    // cv::imshow("plate", license_copy);
    // cv::waitKey(0);

    // Find contours of the binary image
    // RETR_TREE, RETR_LIST, RETR_CCOMP
    cv::findContours(license_copy, contours, cv::RETR_TREE , cv::CHAIN_APPROX_NONE);


    // ########################
    // %% Extract Characters %%
    // ########################
    std::vector<cv::Mat> characters;

    for (const auto& contour : contours) {
        // Compute bounding box for each contour
        cv::Rect bounding_box = cv::boundingRect(contour);

        // Find characters based on size of their contours
        if (bounding_box.width > 100 && bounding_box.width < 500 
        && bounding_box.height > 50 && bounding_box.height < 700) { // (Adjustable) threshold values
        cout << "Bounding size: " << bounding_box.size() << endl;

            
            cv::Mat character = license_plate(bounding_box);

            // Resize character to 32x32 [px]
            cv::Mat resized_character;
            cv::resize(character, resized_character, cv::Size(32, 32));

            // Add to the result vector
            // Pushback won't create new memory if it's already been allocated
            characters.push_back(resized_character);
        }
    }

    // Return list of characters
    return characters; // vector<Mat>

}












