import numpy as np
import cv2
import os
from skimage.morphology import skeletonize

def dip_morphological_processing(sourcePath, outputResize, outputGray, outputMorph):
    for imagePath in os.listdir(sourcePath):

        # imagePath contains name of the image
        inputPath = os.path.join(sourcePath, imagePath)

        # Read images in containing folder
        img = cv2.imread(inputPath)

        # Define the new size (width, height)
        new_width, new_height = img.shape[1] * 2, img.shape[0] * 2  # Doubling the size

        # Resize the image using INTER_CUBIC interpolation
        resize = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        
        filename1 = os.path.splitext(inputPath)[0] + '_resize.tif'
        basename1 = os.path.basename(filename1)
        cv2.imwrite(os.path.join(outputResize, basename1), resize)
        # Convert images from bgr to gray
        rgb_gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

        # Convert images from hsv to bgr
        hsv_bgr = cv2.cvtColor(resize, cv2.COLOR_HSV2BGR)

        # Convert images from bgr to gray
        bgr_gray = cv2.cvtColor(hsv_bgr, cv2.COLOR_BGR2GRAY)

        # Find the minimum pixel value
        min_gray_value = np.min(bgr_gray)

        # Find the maximum pixel value
        max_gray_value = np.max(bgr_gray)

        filename2 = os.path.splitext(inputPath)[0] + '_gray.tif'
        basename2 = os.path.basename(filename2)
        cv2.imwrite(os.path.join(outputGray, basename2), bgr_gray)

        # Get structuring elements
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31,31))
        
        # Top-hat transformation
        tophat = cv2.morphologyEx(bgr_gray, cv2.MORPH_TOPHAT, kernel)

        # Bottom-hat transformation
        bothat =  cv2.morphologyEx(bgr_gray, cv2.MORPH_BLACKHAT, kernel)

        # Resulting image
        res = bothat - tophat

        # Image erosion 
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        erode = cv2.erode(res, kernel2)

        # Remove small unconnected pixels using morphological operations
        cleaned_image = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel2)

        # Connect promising pixels in a line using morphological operations
        connected_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel2)
        
        # Tresholding, Object Segmentation
        thresh, binary = cv2.threshold(connected_image, 65, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Image Invertion
        inverted_image = 255 - binary

        # After thresholding, morphological closing before skeletonization
        #kernel_link = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Try (3,3) or (5,5)
        #closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_link)

        # perform skeletonization
        #skeleton = skeletonize(closed//255)

        # (Optional) Dilate the skeleton to further connect lines
        #kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        #skeleton_connected = cv2.dilate(skeleton.astype(np.uint8), kernel_dilate, iterations=1)
        
        # Ensure skeleton_connected is binary (0 or 1) and res is uint8
        #skeleton_mask = (skeleton_connected > 0).astype(np.uint8)
        
        # (Optional) Dilate the skeleton to further connect lines
        #kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        #dilate = cv2.dilate(skeleton_mask, kernel_dilate)
        
        # If res is not uint8, convert it
        #res_uint8 = res.astype(np.uint8)
        
        # Overlay: where skeleton is white, set res to 255 (white)
        #combined = res_uint8.copy()
        #combined[skeleton_mask == 1] = 255

        # Tresholding, Object Segmentation
        #_, binary2 = cv2.threshold(closed, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Image Invertion
        #inverted_image2 = 255 - binary2

        filename3 = os.path.splitext(inputPath)[0] + '_processed.tif'
        basename3 = os.path.basename(filename3)
        cv2.imwrite(os.path.join(outputMorph, basename3), inverted_image)

if __name__ == '__main__':
    sample_name = 'daun'
    # path of the folder containing the images
    sourcePath = 'd://z/master/comvis/leaf_vein/images/roi/'+sample_name
    # path of the folder that will contain the modified image
    outputPath1 = 'd://z/master/comvis/leaf_vein/images/resize/'+sample_name
    outputPath2 = 'd://z/master/comvis/leaf_vein/images/gray/'+sample_name
    outputPath3 = 'd://z/master/comvis/leaf_vein/images/morph/'+sample_name

    # run
    dip_morphological_processing(sourcePath, outputPath1, outputPath2, outputPath3)
