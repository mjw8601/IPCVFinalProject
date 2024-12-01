/** Interface file for calculating the minkowski distance between two image
 * vectors for kNN
 *
 *  \file statistics/evaluators/MinkowskiDistance.h
 *  \author Cooper White (cjw9009@g.rit.edu)
 *  \date 11 Nov 2024
 */

#include <iostream>

#include "imgs/statistics/minkowski_distance/MinkowskiDistance.h"

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
                         const int p_val) {
  double sum = 0;
  cv::Mat vec_og;
  vec_og.create(source_image.size(), CV_64F);
  source_image.convertTo(vec_og, CV_64F, 1 / 255.0);
  cv::Mat vec_two;
  vec_two.create(vector_two.size(), CV_64F);
  vector_two.convertTo(vec_two, CV_64F, 1 / 255.0);

  for (int cols = 0; cols < vec_og.cols; cols++) {
    for (int rows = 0; rows < vec_og.rows; rows++) {
      sum +=
          pow(abs(vec_og.at<double>(cols, rows) - vec_two.at<double>(cols, rows)),
              p_val);
    }
  }
  sum = pow(sum, 1.0 / p_val);
  return sum;
}
}
