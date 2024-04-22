import os
import itertools
from PIL import Image

# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import warnings
warnings.filterwarnings("ignore")

print('modules loaded')

# MAKING DATAFRAME FOR TRAINING, TESTING AND VALIDATION

# TRAINING DATA
# Generate data paths with labels
train_data_dir = './chest_xray_balanced/train'
filepaths = []
labels = []

folds = os.listdir(train_data_dir)

for fold in folds:
    foldpath = os.path.join(train_data_dir, fold)
    if os.path.isdir(foldpath):  # Check if the path is a directory
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            
            filepaths.append(fpath)
            labels.append(fold)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')

train_df = pd.concat([Fseries, Lseries], axis= 1)
train_df

# TESTING DATA
# Generate data paths with labels
test_data_dir = './chest_xray_balanced/test'
filepaths = []
labels = []

folds = os.listdir(test_data_dir)

for fold in folds:
    foldpath = os.path.join(test_data_dir, fold)
    if os.path.isdir(foldpath):  # Check if the path is a directory
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            
            filepaths.append(fpath)
            labels.append(fold)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
test_df = pd.concat([Fseries, Lseries], axis= 1)

# VALIDATION DATA
# Generate data paths with labels
val_data_dir = './chest_xray_balanced/val'
filepaths = []
labels = []

folds = os.listdir(val_data_dir)

for fold in folds:
    foldpath = os.path.join(val_data_dir, fold)
    if os.path.isdir(foldpath):  # Check if the path is a directory
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            
            filepaths.append(fpath)
            labels.append(fold)

# Concatenate data paths with labels into one dataframe
Fseries = pd.Series(filepaths, name= 'filepaths')
Lseries = pd.Series(labels, name='labels')
valid_df = pd.concat([Fseries, Lseries], axis= 1)

# PREPROCESSING

# crop image size
batch_size = 16
img_size = (224, 224)

tr_gen = ImageDataGenerator(
    rotation_range=20, 
    rescale=1./255,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.10,
    zoom_range=0.05,
    fill_mode='nearest'
)
ts_gen = ImageDataGenerator(rescale=1./255)
val_gen= ImageDataGenerator(rescale=1./255)

train_gen = tr_gen.flow_from_dataframe( 
    train_df, x_col= 'filepaths', y_col= 'labels', 
    target_size= img_size, class_mode= 'categorical',
    color_mode= 'rgb', shuffle= True, batch_size= batch_size
)

valid_gen = val_gen.flow_from_dataframe( 
    valid_df, x_col= 'filepaths', y_col= 'labels', 
    target_size= img_size, class_mode= 'categorical',
    color_mode= 'rgb', shuffle= True, batch_size= batch_size
)

test_gen = ts_gen.flow_from_dataframe( 
    test_df, x_col= 'filepaths', y_col= 'labels', 
    target_size= img_size, class_mode= 'categorical',
    color_mode= 'rgb', shuffle= False, batch_size= batch_size
)

# SAMPLES THE DATA

# Defines dictionary {'class': index}
g_dict = train_gen.class_indices
# Defines list of dictionary's kays (classes), classes names : string   
classes = list(g_dict.keys())
# Getting a batch size samples from the generator
images, labels = next(train_gen)
# Difference between next iterator and for iterator

plt.figure(figsize= (20, 20))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    # scales data to range (0 - 255) and plots it
    image = images[i] / 255
    plt.imshow(image)
    # get image index
    index = np.argmax(labels[i])
    # get class of image
    class_name = classes[index]
    plt.title(class_name, color= 'blue', fontsize= 12)
    plt.axis('off')
plt.tight_layout()

# MODEL STRUCTURE

# Load pre-trained VGG16 model
base_model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))
# Freeze base model layers
base_model.trainable = False

# Count the number of unique labels in the training data
class_count = len(train_df['labels'].unique())

# Add custom classification layers on top of VGG16
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(class_count, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze the top layers of the model
for layer in base_model.layers[-6:]:
    layer.trainable = True

model.compile(Adamax(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# TRAINING THE DATA

# Train the model
epochs = 12   # number of all epochs in training
history = model.fit(train_gen, epochs= epochs, verbose= 1, validation_data= valid_gen, shuffle= False)

# Print model summary
model.summary()

# Get training and validation loss and accuracy
tr_loss = history.history['loss']
val_loss = history.history['val_loss']
tr_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Find the epoch with the lowest validation loss
index_loss = val_loss.index(min(val_loss))
val_lowest = val_loss[index_loss]
loss_label = f'Lowest Validation Loss: {val_lowest:.4f} at Epoch {index_loss+1}'

# Find the epoch with the highest validation accuracy
index_acc = val_acc.index(max(val_acc))
acc_highest = val_acc[index_acc]
acc_label = f'Highest Validation Accuracy: {acc_highest:.4f} at Epoch {index_acc+1}'

# Create a range of epochs for plotting
epochs_range = range(1, epochs+1)

# Save the training history plot as an image file
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, tr_loss, 'r', label='Training Loss')
plt.plot(epochs_range, val_loss, 'g', label='Validation Loss')
plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('output_results/balanced/vgg16_train_val_loss(B).png')

plt.figure(figsize=(10, 6))
plt.plot(epochs_range, tr_acc, 'r', label='Training Accuracy')
plt.plot(epochs_range, val_acc, 'g', label='Validation Accuracy')
plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('output_results/balanced/vgg16_train_val_accuracy(B).png')

# EVALUATING THE MODEL

train_score = model.evaluate(train_gen, verbose= 1)
valid_score = model.evaluate(valid_gen, verbose= 1)
test_score = model.evaluate(test_gen, verbose= 1)

print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Validation Loss: ", valid_score[0])
print("Validation Accuracy: ", valid_score[1])
print('-' * 20)
print("Test Loss: ", test_score[0])
print("Test Accuracy: ", test_score[1])

test_gen.reset()
preds = model.predict(test_gen, steps=len(test_gen), verbose=1)
y_pred = np.argmax(preds, axis=1)

g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion matrix
cm = confusion_matrix(test_gen.classes, y_pred)
cm

plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Save the confusion matrix plot as an image file
plt.savefig('output_results/balanced/vgg16_confusion_matrix(B).png')

class_report = classification_report(test_gen.classes, y_pred, target_names=classes)
print(class_report)

# Save classification report as a figure
plt.figure(figsize=(10, 6))
plt.text(0.1, 0.5, class_report, fontsize=12, ha='left', va='center')
plt.axis('off')
plt.tight_layout()
plt.savefig('output_results/balanced/vgg16_classification_report(B).png')

# ROC curve
from sklearn.metrics import roc_curve, auc

# Assuming binary classification (two classes)
y_true = test_gen.classes
y_score = preds[:, 1]  # Probability of the positive class

fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Save the ROC curve plot as an image file
plt.savefig('output_results/balanced/vgg16_roc_curve(B).png')

# Save the model architecture to JSON file
model_json = model.to_json()
with open('Pneumonia_VGG16(B).json', 'w') as json_file:
    json_file.write(model_json)
    print('Model saved to disk')

# Save the model weights
model.save_weights('Pneumonia_VGG16(B).weights.h5')
print('Weights saved to disk')