# Code: https://github.com/bnsreenu/python_for_microscopists
# Dataset: https://www.epfl.ch/labs/cvlab/data/data-em/
# Tutorial: https://www.youtube.com/watch?v=L5iV5BHkMzM 

# ==============================================================================================
""" IMPORT REQUIRED LIBRARIES """
# ==============================================================================================
from datetime import datetime
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from models import unet_2d

# ==============================================================================================
""" LOAD AND VISUALIZE DATA """
# ==============================================================================================
# Train directory
dir = 'd://z/master/comvis/leaf_vein/images'
image_path = dir + '/gray/daun/'
mask_path = dir + '/morph/daun/'
output = 'd://z/master/comvis/leaf_vein/outputs'

IMG_SIZE = 256
image_dataset = []
mask_dataset = []

images = os.listdir(image_path)
for i, image_name in enumerate(images):
    if (image_name.split('.')[1] == 'tif'):
        # Print image directory + image name
        image = cv2.imread(image_path+image_name, 1)
        image = Image.fromarray(image)
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_dataset.append(np.array(image))

masks = os.listdir(mask_path)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'tif'):
        image = cv2.imread(mask_path + image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((IMG_SIZE, IMG_SIZE))
        mask_dataset.append(np.array(image))

# Normalize images
image_dataset = np.array(image_dataset)/255.
# Do not normalize masks, just rescale to 0 to 1
mask_dataset = np.expand_dims((np.array(mask_dataset)), axis=3)/255.
#mask_dataset = np.array(mask_dataset)[..., np.newaxis] / 255.

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(image_dataset, mask_dataset, test_size=0.2)

image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (IMG_SIZE, IMG_SIZE, 3)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (IMG_SIZE, IMG_SIZE)), cmap='gray')
plt.show()

# ==============================================================================================
""" RUN MODELS """
# ==============================================================================================
model_name = 'UNet'
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
print(IMG_CHANNELS)
num_labels = 1 # Binary
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
batch_size = 8
num_epochs = 10

""" UNet """
unet_model = unet_2d(input_shape)

# Gets the current date and time to start compiling model
start1 = datetime.now()

print('Fitting model ....')
unet_history = unet_model.fit(
    X_train, y_train,
    batch_size = batch_size,
    validation_data = (X_val, y_val),
    shuffle = False, 
    epochs = num_epochs,
    verbose=1
)
# Gets the current date and time while stopping model compilation.
stop1 = datetime.now()

# Execution time of the model
execution_time_UNet = stop1 - start1
print("UNet execution time is: ", execution_time_UNet)

unet_model.save(f'{output}/leafvein_{model_name}_50epochs.keras')

# ==============================================================================================
""" MODEL EVALUATION """
# ==============================================================================================

# Convert the history dict to a pandas dataframe and save as csv for future plotting
import pandas as pd
unet_history_df = pd.DataFrame(unet_history.history)

with open(f'{output}/leafvein_{model_name}_history_df.csv', mode='w') as f:
    unet_history_df.to_csv(f)

# Check history plot
history = unet_history

# Plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{output}/leafvein_{model_name}_accuracy.png')
plt.show()

#acc = history.history['accuracy']
acc = history.history['jacard_coef']
#val_acc = history.history['val_accuracy']
val_acc = history.history['val_jacard_coef']

plt.plot(epochs, acc, 'y', label = 'Training Jacard')
plt.plot(epochs, val_acc, 'r', label = 'Validation Jacard')
plt.title('Training and validation Jacard')
plt.xlabel('Epochs')
plt.ylabel('Jacard')
plt.savefig(f'{output}/leafvein_{model_name}_Jacard_coef.png')
plt.legend()
plt.show()

import random
# Selecting a random image
test_img_number = random.randint(0, X_val.shape[0]-1)
# Extracting the image and its corresponding image
test_img = X_val[test_img_number]
mask_img = y_val[test_img_number]
# Expanding dimensions for model input
test_img_number = np.expand_dims(test_img, 0)
# Making predictions
pred_img = (unet_model.predict(test_img_number)[0,:,:,0]>0.5).astype(np.uint8)
# Note: Pixels above a threshold (0.5) are classified as foreground (1), others as background (0).

plt.figure(figsize=(16,8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img, cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(mask_img, cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(pred_img, cmap='gray')
plt.show()

# IoU for a single image
from keras.metrics import MeanIoU
n_classes = 2
IoU_keras = MeanIoU(num_classes = n_classes)
IoU_keras.update_state(mask_img[:,:,0], pred_img)
print("Mean IoU =", IoU_keras.result().numpy())

# Calculate IoU for all test images and average
IoU_values = []
for img in range(0, X_val.shape[0]):
    temp_img = X_val[img]
    mask_img = y_val[img]
    temp_img_input = np.expand_dims(temp_img, 0)
    pred_img = (unet_model.predict(temp_img_input)[0,:,:,0]>0.5).astype(np.uint8)

    IoU = MeanIoU(num_classes = n_classes)
    IoU.update_state(mask_img[:,:,0], pred_img)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)

    print(IoU)
    
IoU_df = pd.DataFrame(IoU_values, columns=['IoU'])
IoU_df = IoU_df[IoU_df != 1.0]
mean_IoU = IoU_df.mean().values
print("Mean IoU : ", mean_IoU)

# ==============================================================================================
"""MAKE PREDICTION USING TEST DATA"""
# ==============================================================================================

testdir = 'd://z/master/comvis/leaf_vein/dataset/test'
image_testpath = testdir + '/images/'
mask_testpath = testdir + '/masks/'
predict_testpath = testdir + '/predict/'

image_testset = []
mask_testset = []

images = os.listdir(image_testpath)
for i, image_name in enumerate(images):
    if (image_name.split('.')[1] == 'tif'):
        # Print image directory + image name
        image = cv2.imread(image_testpath+image_name, 1)
        image = Image.fromarray(image)
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_testset.append(np.array(image))

masks = os.listdir(mask_testpath)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'tif'):
        image = cv2.imread(mask_testpath + image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((IMG_SIZE, IMG_SIZE))
        mask_testset.append(np.array(image))

# Normalize images
image_testset = np.array(image_testset)/255.
# Do not normalize masks, just rescale to 0 to 1
mask_testset = np.expand_dims((np.array(mask_testset)), 3)/255.
# Mask prediction      
pred_masks = (unet_model.predict(image_testset, batch_size=batch_size, verbose=1)[0,:,:,0]>0.5).astype(np.uint8)
# Save results (predicted masks)
for idx, pred in enumerate(pred_masks):
    # If pred is (H, W, 1)
    if pred.ndim == 3:
        pred_mask = (pred[:, :, 0] > 0.5).astype(np.uint8) * 255
    # If pred is (H, W)
    elif pred.ndim == 2:
        pred_mask = (pred > 0.5).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unexpected pred shape: {pred.shape}")
    image_name = images[idx]
    cv2.imwrite(f"{predict_testpath}/{image_name}", pred_mask)