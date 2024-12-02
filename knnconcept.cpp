#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

// Functions - written below main
void ColorReduce(const cv::Mat& src, cv::Mat& dst, const int& div=64);
void MouseHandler(int event, int x, int y, int flags, void* userdata);
std::vector<cv::Mat> auto_extract_characters(cv::Mat& license_plate);

// Points for manual segmentation of characters
std::vector<cv::Point2f> src_points (4);
std::vector<cv::Point2f> dst_points (4);

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


    // ##############################
    // ###### Draw Box Testing ######
    // ##############################

    // // Read in an image
    // cv::Mat src_image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    // cv::Mat dst_image;

    // // Fit image to the screen and maintain image ratio
    // cv::namedWindow("Select Points", cv::WINDOW_KEEPRATIO);
    // cv::setWindowProperty("Select Points", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    // // Show image to crop
    // cv::imshow("Select Points", src_image);
    // cv::setMouseCallback("Select Points", MouseHandler);

    // // Wait to select 4 points
    // while (src_points.size() != 4) {
    //     cv::waitKey(1);
    // }

    // // Define letter box size (28x28)
    // // However, I could make this rectilinear, 
    // // I have no arguement for one or the other
    // dst_points = {
    //     cv::Point2f(0, 0),    // Top-left corner
    //     cv::Point2f(28, 0),   // Top-right corner
    //     cv::Point2f(28, 28),  // Bottom-right corner
    //     cv::Point2f(0, 28)    // Bottom-left corner
    // };

    //     // Calculate the perspective transformation matrix
    // cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_points, dst_points);

    // // Perform the perspective warp
    // cv::Mat perspective_image;
    // cv::warpPerspective(src_image, perspective_image, perspective_matrix, src_image.size());

    // // Display the output image
    // cv::namedWindow("Warped Image", cv::WINDOW_NORMAL);
    // cv::imshow("Warped Image", perspective_image);
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
        cv::imwrite("character_" + std::to_string(i) + ".jpg", characters_in_plate[i]);
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
 * \brief Handles mouse inputs for clicking 4 points to create quadralateral
 * 
 * \param[in] event Listens to mouse events (i.e. left click, right click, etc.)
 * \param[in] x Horizontal coordinate of the click
 * \param[in] y Vertical coordinate of the click
 * \param[in] flags (Optional) User flags (N/A)
 * \param[in] userdata (Optional) Reference to user data (N/A)
 */
void MouseHandler(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Record the point
        src_points.push_back(cv::Point2d(x, y));
        std::cout << "Point selected: (" << x << ", " << y << ")\n";

        // If 4 points are selected, close the window
        if (src_points.size() == 4) {
            std::cout << "All 4 points selected!\n";
        }
    }
}

/**
 * \brief Crops (isolates) letter from license plate image 
 * 
 * \param[in] license_plate Image to perform q2q
 * \param[out] src_points 
 * \param[out] dst_points 
 * \return cv::Mat Isolated character from license plate
 */
std::vector<cv::Mat> extract_characters(
        const cv::Mat& license_plate, 
        std::vector<cv::Point2f>& src_points, 
        std::vector<cv::Point2f>& dst_points) {
    // Clear srcPoints for new image
    src_points.clear();

    // Show the image and set up the callback
    cv::imshow("Select Points", license_plate);
    cv::setMouseCallback("Select Points", MouseHandler);

    // Wait for the user to select 4 points
    while (src_points.size() < 4) {
        cv::waitKey(1);
    }

    // Define the destination quadrilateral
    std::vector<cv::Point2f> dstPoints = {
        cv::Point2f(100, 100),
        cv::Point2f(500, 100),
        cv::Point2f(500, 400),
        cv::Point2f(100, 400)
    };

    // Calculate the perspective transformation matrix
    cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(src_points, dstPoints);

    // Perform the perspective warp
    cv::Mat outputImage;
    cv::warpPerspective(license_plate, outputImage, perspectiveMatrix, license_plate.size());

    // Display the output
    cv::imshow("Warped Image", outputImage);
    cv::waitKey(0);

    // Clean up
    cv::destroyAllWindows();
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
    cv::threshold(license_plate, license_plate, 80, 255,  cv::THRESH_BINARY);
    
    /* This, in my tests, produces worse results. Reason being is that the letters are always black, and the background always isn't. 
       So, choosing a color near black works for letters in the plate itself, and the bounding box takes care of whatever else is out there */
    // cv::threshold(license_plate, license_plate, 100, 255,  cv::THRESH_OTSU | cv::THRESH_BINARY); 


    // ###################
    // %% Find Contours %%
    // ###################

    // White text on a black background, findContours prefers it like this
    cv::bitwise_not(license_plate, license_plate);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat license_copy = license_plate.clone();

    // Find contours of the binary image
    // Most simple, and fastest implementation
    cv::findContours(license_copy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);


    // ########################
    // %% Extract Characters %%
    // ########################
    std::vector<cv::Mat> characters;

    for (const auto& contour : contours) {
        // Compute bounding box for each contour
        cv::Rect bounding_box = cv::boundingRect(contour);

        // Find characters based on size of their contours
        if (bounding_box.width > 100 && bounding_box.width < 300 
        && bounding_box.height > 200 && bounding_box.height < 500) { // (Adjustable) threshold values
        cout << bounding_box.size() << endl;
            
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

/**
 * \brief Isolates license plate from image assuming common ratios
 * 
 * \param[in] image Grayscale parking-lot image
 * \return cv::Mat 
 */
cv::Mat isolateLicensePlate(const cv::Mat& image) {
    cv::Mat gray, edges, license_plate;
    
    // Resize image for consistent plate size detection
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(1000, 750));
    
    // Convert to grayscale
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    
    // Apply GaussianBlur to reduce noise
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    
    // Edge detection
    cv::Canny(gray, edges, 100, 200);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Filter contours to find license plate
    for (const auto& contour : contours) {
        cv::Rect bounding_box = cv::boundingRect(contour);
        
        // Filter based on aspect ratio and size
        double aspect_ratio = (double)bounding_box.width / bounding_box.height;
        if (aspect_ratio > 2.0 && aspect_ratio < 6.0 && // Typical license plate aspect ratio
            bounding_box.width > 200 && bounding_box.width < 1000 && // Width range
            bounding_box.height > 50 && bounding_box.height < 300) { // Height range
            
            // Extract the license plate region
            license_plate = resized(bounding_box);
            break;
        }
    }

    return license_plate;
}














