# Code: https://github.com/bnsreenu/python_for_microscopists
# Dataset: https://www.epfl.ch/labs/cvlab/data/data-em/
# Tutorial: https://www.youtube.com/watch?v=L5iV5BHkMzM 

# ======================================================================================================================
""" IMPORT REQUIRED LIBRARIES """
# ======================================================================================================================
from datetime import datetime
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

from algorithms.unet import UNet, ResUNet, Attention_UNet, Attention_ResUNet, unet_2d
from model._resunet import resunet_a_2d
from algorithms.segnet import segnet, SegNet, ResSegNet, Attention_SegNet, Attention_ResSegNet
from model._fcn import fcn_8
#from model._att_unet import Attention_UNet
from algorithms.att_unet import attention_unet
from algorithms.wnet import wnet_2d
#from model._att_resunet import Attention_ResUNet
from model._pspnet import pspnet_2d
from model.metrics import iou, jacard_coef, dice_coef

# ======================================================================================================================
""" DEFINE THE MODEL """
# ======================================================================================================================
# Set model name
model_name = 'att_segnet'  
model = UNet if model_name == 'unet' else \
        ResUNet if model_name == 'resunet' else \
        Attention_UNet if model_name == 'att_unet' else \
        Attention_ResUNet if model_name == 'att_resunet' else \
        SegNet if model_name == 'segnet' else \
        ResSegNet if model_name == 'ressegnet' else \
        Attention_SegNet if model_name == 'att_segnet' else \
        Attention_ResSegNet if model_name == 'att_ressegnet' else \
        None
if model is None:
    raise ValueError("Invalid model name. Choose from 'unet', 'resunet', 'att_unet', att_resunet', 'segnet', 'ressegnet', "
    "'att_segnet', 'att_ressegnet")
batch_size = 8
num_epochs = 50

# ======================================================================================================================
""" LOAD AND VISUALIZE DATA """
# ======================================================================================================================
# Train directory
dir = 'd://z/master/comvis/leaf_vein/dataset/train'
image_path = dir + '/images/'
mask_path = dir + '/masks/'
outputs = f'd://z/master/comvis/leaf_vein/outputs/'+ model_name

IMG_SIZE = 256
image_dataset = []
mask_dataset = []

images = os.listdir(image_path)
for i, image_name in enumerate(images):
    if (image_name.split('.')[1] == 'tif'):
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

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(image_dataset, mask_dataset, test_size=0.2)

image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (IMG_SIZE, IMG_SIZE, 3)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (IMG_SIZE, IMG_SIZE)), cmap='gray')
plt.show()

# ======================================================================================================================
""" RUN MODELS """
# ======================================================================================================================
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
num_labels = 1 # Binary
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

""" Model """
model = model(input_shape)
# Gets the current date and time to start compiling model
start1 = datetime.now()

print('Fitting model ....')
model_history = model.fit(
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
training_time = stop1 - start1
print("UNet execution time is: ", training_time)

model.save(f'{outputs}/leafvein_{model_name}_50epochs.keras')

# ======================================================================================================================
""" MODEL EVALUATION """
# ======================================================================================================================

# Convert the history dict to a pandas dataframe and save as csv for future plotting
import pandas as pd
model_history_df = pd.DataFrame(model_history.history)
with open(f'{outputs}/leafvein_{model_name}_history_df.csv', mode='w') as f:
    model_history_df.to_csv(f)

# Plot the training and validation loss at each epoch
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{outputs}/leafvein_{model_name}_accuracy.png')
plt.show()

# Plot the training and validation jacard at each epoch
acc = model_history.history['jacard_coef']
val_acc = model_history.history['val_jacard_coef']
plt.plot(epochs, acc, 'y', label = 'Training Jacard')
plt.plot(epochs, val_acc, 'r', label = 'Validation Jacard')
plt.title('Training and validation Jacard')
plt.xlabel('Epochs')
plt.ylabel('Jacard')
plt.savefig(f'{outputs}/leafvein_{model_name}_Jacard_coef.png')
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
pred_img = (model.predict(test_img_number)[0,:,:,0]>0.5).astype(np.uint8)
# Note: Pixels above a threshold (0.5) are classified as foreground (1), others as background (0).

# Plotting the image, ground truth and prediction
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
plt.savefig(f'{outputs}/leafvein_{model_name}_predict.png')
plt.show()

# IoU for a single image
from keras.metrics import MeanIoU
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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
    pred_img = (model.predict(temp_img_input)[0,:,:,0]>0.5).astype(np.uint8)

    IoU = MeanIoU(num_classes = n_classes)
    IoU.update_state(mask_img[:,:,0], pred_img)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)

    print(IoU)

IoU_df = pd.DataFrame(IoU_values, columns=['IoU'])
IoU_df = IoU_df[IoU_df != 1.0]
mean_IoU = IoU_df.mean().values
print("Mean IoU : ", mean_IoU)

y_pred = model.predict(X_val)

# Convert continuous predictions to binary
y_pred_bin = (y_pred >= 0.5).astype(np.uint8).flatten()
y_true_bin = y_val.flatten()
y_true_bin = (y_true_bin > 0.5).astype(np.uint8)

# Ensure both arrays contain only 0 and 1
assert np.all(np.isin(y_true_bin, [0, 1])), "y_true_bin contains non-binary values!"
assert np.all(np.isin(y_pred_bin, [0, 1])), "y_pred_bin contains non-binary values!"

accuracy = accuracy_score(y_true_bin, y_pred_bin)
precision = precision_score(y_true_bin, y_pred_bin)
recall = recall_score(y_true_bin, y_pred_bin)
f1 = f1_score(y_true_bin, y_pred_bin)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Dice coefficient (same as F1 for binary masks)
dice = f1_score(y_true_bin, y_pred_bin)

# Jaccard coefficient (IoU)
from sklearn.metrics import jaccard_score
jaccard = jaccard_score(y_true_bin, y_pred_bin)

print(f"Dice coefficient: {dice:.4f}")
print(f"Jaccard coefficient (IoU): {jaccard:.4f}")

# ======================================================================================================================
"""MAKE PREDICTION USING TEST DATA"""
# ======================================================================================================================

testdir = 'd://z/master/comvis/leaf_vein/dataset/test'
image_testpath = testdir + '/images/'
mask_testpath = testdir + '/masks/'
predict_testpath = outputs + '/mask_predict/'

image_testset = []
mask_testset = []

start2 = datetime.now()
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
pred_masks = (model.predict(image_testset, batch_size=batch_size, verbose=1)[0,:,:,0]>0.5).astype(np.uint8)

# Predict all test images
pred_masks = model.predict(image_testset, batch_size=batch_size, verbose=1)

# Save images at each prediction
for idx, pred in enumerate(pred_masks):
    pred_mask = (pred[:, :, 0] > 0.5).astype(np.uint8) * 255
    image_name = images[idx]
    save_path = os.path.join(predict_testpath, image_name)
    cv2.imwrite(save_path, pred_mask)

# Save testing duration on test images
stop2 = datetime.now()
testing_time = stop2 - start2

# Save model performance
metrics = [['accuracy', accuracy],
           ['precision', precision],
            ['recall', recall], 
            ['f1-score', f1],
            ['dice', dice],
            ['jaccard', jaccard], 
            ['IoU', mean_IoU],
            ['training_time', training_time], 
            ['testing_time', testing_time]]
metrics_df = pd.DataFrame(metrics, columns=['params.', 'values'])
metrics_df.to_csv(f'{outputs}/leafvein_{model_name}_metrics.csv', index=False)

with open(f'{outputs}/leafvein_{model_name}_history_df.csv', mode='w') as f:
    model_history_df.to_csv(f)