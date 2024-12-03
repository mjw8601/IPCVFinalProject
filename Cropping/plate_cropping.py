import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to let the user click on 4 points
def get_four_points(img):
    points = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:  # Ensure clicks are within the image area
            if len(points) < 4:
                points.append((event.xdata, event.ydata))
                plt.scatter(event.xdata, event.ydata, color='red')
                plt.draw()
            if len(points) == 4:  # Disconnect once 4 points are selected
                plt.close()

    # Display the image and let the user select points
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    fig.suptitle("Click on the four corners of the region")
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()  # Wait for the user to select points
    return np.array(points, dtype=np.float32)

# Function to warp the selected region to a 720p resolution
def warp_image(img, points, width=1280, height=720):
    dst_points = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(points, dst_points)
    return cv2.warpPerspective(img, matrix, (width, height))

# Function to process all images in a folder recursively
def process_folder(folder_path, output_folder):
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                print(f"Processing: {image_path}")

                # Load the image
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error: Could not load image at {image_path}")
                    continue

                # Get four points from the user
                print("Click on the four corners of the region. Close the window after selecting four points.")
                points = get_four_points(img)

                # Warp the image
                warped = warp_image(img, points)
                print("Warped image to 720p resolution.")

                # Save the warped image with '_cropped' suffix
                relative_path = os.path.relpath(root, folder_path)  # Get relative subfolder path
                save_folder = os.path.join(output_folder, relative_path)
                os.makedirs(save_folder, exist_ok=True)  # Ensure output folder exists

                base_name, ext = os.path.splitext(file)
                save_path = os.path.join(save_folder, f"{base_name}_cropped{ext}")
                cv2.imwrite(save_path, warped)
                print(f"Warped image saved as '{save_path}'.")

# Main function
def main(input_folder, output_folder):
    # Ensure the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found!")
        return

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process all images in the folder
    process_folder(input_folder, output_folder)
    print("Processing complete.")

if __name__ == "__main__":
    # Input and output folders (update these paths as needed)
    input_folder = "Cropping/uncropped"  # Replace with your input folder path
    output_folder = "Cropping/cropped"  # Replace with your desired output folder path

    main(input_folder, output_folder)
