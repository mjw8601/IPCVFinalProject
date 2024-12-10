
License Plate Recognition using kNN Algorithm
=============================================

This project implements a license plate recognition system using the k-Nearest Neighbor (kNN) algorithm. The workflow includes preprocessing, character segmentation, and classification via statistical filtering. While leveraging datasets like MNIST and RITCIS for testing, we created a custom dataset of license plates to evaluate the system's real-world applicability.

Overview
--------

The goal of this project is to develop a robust machine learning model for symbolic classification and detection, specifically for license plate recognition. Using statistical methods and kNN, this project explores the challenges and constraints of such a system when applied to real-world datasets.

Key Features:
- **kNN Implementation**: A basic yet effective algorithm for character classification.
- **Custom Dataset**: A collection of over 2,500 license plate images with alphanumeric characters.
- **Image Preprocessing**: Noise reduction, contrast enhancement, and character segmentation using OpenCV.
- **Statistical Filtering**: Reduces noise by filtering out contours that do not match expected character properties.

Repository Structure
--------------------

- `App File/`: Contains the test file that reads in u-byte data and tests the kNN algorithm
- `Cropping/`: Python Cropping script, unused but made for inital testing of license plate data
- `classifiers/`: Contains the kNN implementation.
- `data_readers/`: Carl Salvaggios MNIST Data reading functions, used in our kNN implementation.
- `evaluators/`: Implementation of confusion matrix
- `funcs_and_label-reading/`: Statistical Filtering functions and License plate labeling scripts
- `minkowski_distance/`: Minkowski distance used for finding the nearest neighbors in kNN
- `CMakeLists.txt`: CMakeLists to accompany minkowski distance.
- `README.md`: Documentation for the repository.

Methodology
-----------

1. Preprocessing
    - Convert images to grayscale.
    - Apply bilateral filtering and Gaussian blur to reduce noise.
    - Threshold the images to binary (white characters on a black background).

2. Character Segmentation
    - Use OpenCV's `findContours` to extract characters.
    - Apply bounding box ratio filtering and statistical measures to remove noise.

3. Classification
    - Flatten images into vectors and calculate distances using the Minkowski formula.
    - Use the kNN algorithm with optimal parameters (`k=2`, `p=3`) to classify characters.

Results
-------

| Dataset       | Training Images | Testing Images | Accuracy |
|---------------|-----------------|----------------|----------|
| MNIST         | 60,000          | 10,000         | 97.05%   |
| RITCIS        | 36,831          | 7,977          | 89.61%   |
| Custom Plates | 5,500           | 1,100          | 97.22%   |

Testing cross-dataset compatibility revealed the limitations of generalized training models, with accuracies dropping significantly when training on MNIST or RITCIS and testing on custom license plates.

Limitations and Future Work
---------------------------

While the system achieves high accuracy with its custom dataset, it struggles with cross-dataset generalization due to differences in image properties (e.g., padding, character size). Future improvements could include:
- Implementing adaptive thresholding for better contrast in preprocessing.
- Experimenting with neural networks for more robust classification.

Usage
-----

1. Clone this repository:
   ```bash
   git clone https://github.com/mjw8601/IPCVFinalProject.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```
4. Train and test the kNN model:
   ```bash
   python knn_train_test.py
   ```

Contributors
------------
- Cooper White
- Stavros Viron
- Luke Callahan
- Gian-Mateo Tifone
- Mason Wahlers
