/** Interface file for calculating the minkowski distance between two image
 * vectors for kNN
 *
 *  \file statistics/evaluators/MinkowskiDistance.h
 *  \author Cooper White (cjw9009@g.rit.edu)
 *  \date 11 Nov 2024
 */

#pragma once

#include <opencv2/core.hpp>

namespace statistics {

/** Perform minkowski distance calculations
 *
 *  \param[in] vector_one            source image to be vectorized
 *  \param[in] vector_two            vectorized images of the k-NN to compare
 * against
 * 
 *  \param[in] p_val                 P_val for the minkowski distance
 *  formula [Default is 1]
 *
 *  \return distance between source image vector and vector of k-NN vector
 */

double MinkowskiDistance(const cv::Mat& source_image, const cv::Mat& vector_two,
                       const int p_val = 1);
}



