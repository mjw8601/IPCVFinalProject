Image Cropping and Warping Tool

This Python tool allows users to recursively process all images in a folder, manually select regions of interest by clicking on 4 points (e.g., corners of a license plate), and then warp and resize the selected region to a uniform resolution (1280x720 by default). The processed images are saved with a `_cropped` suffix in an output folder while preserving the folder structure.

Features
--------
- Interactive Cropping: Users can click on the four corners of the region of interest for each image.
- Recursive Processing: The tool processes images in the input folder and all its subdirectories.
- Output Management: Cropped images are saved in the specified output folder with the original folder structure intact.
- Support for Common Image Formats: Works with `.png`, `.jpg`, `.jpeg`, `.bmp`, and `.tiff` files.

Requirements
------------
To run this project, make sure you have Python 3.6+ installed along with the following libraries:
- opencv-python
- numpy
- matplotlib

Install the required libraries using:
pip install opencv-python-headless numpy matplotlib

How to Use
----------
1. Clone this repository or download the script to your local machine:
   git clone https://github.com/mjw8601/IPCVFinalProject
   cd IPCVFinalProject/Cropping

2. Prepare your input images:
   - Place all uncropped images in a folder (e.g., `Cropping/uncropped`).

3. Run the script:
   python script_name.py
   Replace `script_name.py` with the actual filename.

4. Follow the instructions:
   - For each image, click on the four corners of the region you want to crop. Close the plot window after selecting the points.

5. Find the processed images in the output folder (e.g., `Cropping/cropped`).

Folder Structure
-----------------
project-folder/
├── script_name.py       # The main script
├── Cropping/
│   ├── uncropped/       # Input images go here
│   ├── cropped/         # Processed images will be saved here
├── README.md            # Project documentation

Example Workflow
----------------
1. Place your uncropped images in `Cropping/uncropped`:
   Cropping/uncropped/
   ├── image1.jpg
   ├── folder1/
   │   └── image2.png
   └── folder2/
       └── image3.jpeg

2. Run the script and interactively crop each image:
   python script_name.py

3. After processing, you’ll see the results in `Cropping/cropped`:
   Cropping/cropped/
   ├── image1_cropped.jpg
   ├── folder1/
   │   └── image2_cropped.png
   └── folder2/
       └── image3_cropped.jpeg

Notes for Fellow Gooners
------------------------
- The tool is designed to be intuitive, but it requires user input for cropping.
- You can customize the resolution of the output images by modifying the `warp_image` function in the script (default is 1280x720).


Support
-------
Feel free to reach out if you have any questions!
