import numpy as np
import cv2
import os
from skimage.morphology import skeletonize

def dip_morphological_processing(sourcePath, outputPath0, outputPath1, outputPath2):
    # Ensure output directories exist
    os.makedirs(outputPath0, exist_ok=True)
    os.makedirs(outputPath1, exist_ok=True)
    os.makedirs(outputPath2, exist_ok=True)

    for imagePath in os.listdir(sourcePath):
        # imagePath contains name of the image (filename only)
        inputPath = os.path.join(sourcePath, imagePath)

        # Read image; if it fails skip
        img = cv2.imread(inputPath)
        if img is None:
            # skip non-image files
            continue

        # Resize the image using INTER_CUBIC interpolation
        resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        filename0 = os.path.splitext(imagePath)[0] + '_resized.tif'
        cv2.imwrite(os.path.join(outputPath0, filename0), resize)

        # Reduce brightness and blur
        resize = cv2.convertScaleAbs(resize, alpha=1.0, beta=-50)
        resize = cv2.GaussianBlur(resize, (7, 7), 0)

        # Convert to grayscale (from BGR)
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        filename1 = os.path.splitext(imagePath)[0] + '_gray.tif'
        cv2.imwrite(os.path.join(outputPath1, filename1), gray)

        # Binary thresholding using Otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)  # Invert the binary image
        filename2 = os.path.splitext(imagePath)[0] + '_binary.tif'
        cv2.imwrite(os.path.join(outputPath2, filename2), binary)

# Generate ground truth masks 
def generate_ground_truth_masks(sourcePath, outputPath):
    # Ensure output directory exists
    os.makedirs(outputPath, exist_ok=True)

    for imagePath in os.listdir(sourcePath):
        inputPath = os.path.join(sourcePath, imagePath)

        # Read image; if it fails skip
        img = cv2.imread(inputPath)
        if img is None:
            # skip non-image files
            continue

        # Reduce brightness and blur
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=-50)
        img = cv2.GaussianBlur(img, (7, 7), 0)

        # Convert to grayscale (from BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binary thresholding using Otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)  # Invert the binary image
        filename = os.path.splitext(imagePath)[0] + '_binary.tif'
        cv2.imwrite(os.path.join(outputPath, filename), binary)

# Randome images and save to the folder for training and testing
def random_save_images(sourcePath_img, outputPath_train, outputPath_test, train_frac=0.5):
    # Ensure output directories exist
    os.makedirs(outputPath_train, exist_ok=True)
    os.makedirs(outputPath_test, exist_ok=True)

    # list only files (skip directories)
    imageFiles = [f for f in os.listdir(sourcePath_img) if os.path.isfile(os.path.join(sourcePath_img, f))]
    num_images = len(imageFiles)
    print(f"Total number of images available: {num_images}")
    if num_images == 0:
        print("No images found. Nothing to do.")
        return

    # compute the number of training images (fraction of total)
    n_images = max(1, int(num_images * train_frac))
    n_images = min(n_images, num_images)

    # randomly select n_images from the sourcePath_img
    selectedFiles = list(np.random.choice(imageFiles, n_images, replace=False))
    for file in selectedFiles:
        inputPath = os.path.join(sourcePath_img, file)
        img = cv2.imread(inputPath)
        if img is None:
            continue
        outputFilePath = os.path.join(outputPath_train, file)
        cv2.imwrite(outputFilePath, img)

    # save the remaining images to another folder
    remainingFiles = set(imageFiles) - set(selectedFiles)
    for file in remainingFiles:
        inputPath = os.path.join(sourcePath_img, file)
        img = cv2.imread(inputPath)
        if img is None:
            continue
        outputFilePath = os.path.join(outputPath_test, file)
        cv2.imwrite(outputFilePath, img)

if __name__ == '__main__':
    # path of the folder containing the images
    sourcePath = 'd://z/master/comvis/leaf_vein/images/roi/daun/'
    # path of the folder that will contain the modified image
    outputPath0 = 'd://z/master/comvis/leaf_vein/images/resize/daun/'
    outputPath1 = 'd://z/master/comvis/leaf_vein/images/gray/daun/'
    outputPath2 = 'd://z/master/comvis/leaf_vein/images/morph/daun2/'
    # run
    #dip_morphological_processing(sourcePath, outputPath0, outputPath1, outputPath2)
    
    # random save images for training and testing
    outputPath_train = 'd://z/master/comvis/leaf_vein/dataset/train/images/'
    outputPath_test = 'd://z/master/comvis/leaf_vein/dataset/test/images/'
    #random_save_images(outputPath0, outputPath_train, outputPath_test)

    # process the images in the morph folder for training and testing
    mask_outputPath_train = 'd://z/master/comvis/leaf_vein/dataset/train/masks/'
    mask_outputPath_test = 'd://z/master/comvis/leaf_vein/dataset/test/masks/'
    generate_ground_truth_masks(outputPath_train, mask_outputPath_train)
    generate_ground_truth_masks(outputPath_test, mask_outputPath_test)