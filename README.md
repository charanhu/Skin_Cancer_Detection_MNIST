# Skin-Cancer-Classification

The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.

Play with Skin Cancer Images here: (https://skin-cancer-detection-cnn.herokuapp.com/)

## Model Architecture:

![alt_text](https://github.com/charanhu/Skin-Cancer-Detection-MNIST/blob/main/model_architecture.png)

## Data

https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000


## How to Run the App:

## Method1
•Run the app.py file

•Go to http://localhost:5000/ on your browser

•Use the Upload and button to browse and upload the image you want

•Hit submit to get the results.

## Method 2
•Depploy it to Azure Webapp or Heroku App through github repository

•Go to url generated after deployment on your browser

•Use the Upload and button to browse and upload the image you want

•Hit submit to get the results.



## Libraries Used: 

• numpy

• keras

• tensorflow-cpu==2.5.0

• pandas

• matplotlib

• pillow

• flask

• seaborn

• gunicorn

## Skin_Cancer_Detection.ipynb:
This is the Jupyter notebook used to define and train the model.

## app.py:
This is the flask app that needs to run in order to use the webapp

## skin_cancer_detection.py:
This contains the definition of the CNN model.

## best_model.h5:
Contains the weights of the best model.

## CNN model summary:

```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 16)        448       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 16)        0         
_________________________________________________________________
batch_normalization (BatchNo (None, 14, 14, 16)        64        
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 12, 12, 32)        4640      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 10, 10, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 5, 5, 64)          256       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 3, 128)         73856     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1, 1, 256)         295168    
_________________________________________________________________
flatten (Flatten)            (None, 256)               0         
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               65792     
_________________________________________________________________
batch_normalization_2 (Batch (None, 256)               1024      
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896     
_________________________________________________________________
batch_normalization_3 (Batch (None, 128)               512       
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256      
_________________________________________________________________
batch_normalization_4 (Batch (None, 64)                256       
_________________________________________________________________
dropout_2 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 32)                2080      
_________________________________________________________________
batch_normalization_5 (Batch (None, 32)                128       
_________________________________________________________________
dense_4 (Dense)              (None, 7)                 231       
=================================================================
Total params: 504,103
Trainable params: 502,983
Non-trainable params: 1,120
_________________________________________________________________
```


