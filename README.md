# Melanoma Detection with TensorFlow
This project is aimed at building a deep learning model using TensorFlow to detect melanoma, a type of skin cancer, from dermoscopy images. The dataset used for this project is the ISIC 2019 challenge dataset, which consists of 25,331 images with binary labels indicating whether a given image contains melanoma or not.

## Dataset
The ISIC 2019 challenge dataset can be downloaded from the official website: https://challenge.isic-archive.com/data. It contains two separate folders for training and testing, which include images in JPEG format and corresponding metadata in CSV format.

## Preprocessing
Before feeding the images to the model, we preprocess them by resizing them to a fixed size of 180*180 pixels and normalizing the pixel values to be between 0 and 1. Additionally, we augment the training dataset using techniques like rotation, zooming, and horizontal flipping to increase the size and diversity of the dataset.

## Model
The code snippet provided defines a CNN (Convolutional Neural Network) model for image classification. Let's break down the model architecture step by step:

1. Data Preprocessing:
   - The `layers.experimental.preprocessing.Rescaling` layer is applied to normalize the pixel values of the input images to the range [0, 1] by dividing them by 255.

2. Convolutional Layers:
   - The first `Conv2D` layer has 16 filters, a kernel size of 3x3, and applies the 'relu' activation function. It preserves the spatial dimensions of the input image by using 'same' padding.
   - After each `Conv2D` layer, a `MaxPooling2D` layer is added. This layer performs max pooling with a default pool size of 2x2, reducing the spatial dimensions of the feature maps.

3. Dropout Layer:
   - The `Dropout` layer is added with a dropout rate of 0.2. It randomly drops a fraction of the input units during training, which helps in reducing overfitting.

4. Flattening:
   - The `Flatten` layer is used to flatten the 2D feature maps into a 1D vector, preparing the data for the fully connected layers.

5. Dense Layers:
   - The `Dense` layer with 128 units and 'relu' activation is added. It performs a linear transformation on the input data followed by the 'relu' activation function.
   - The final output layer will depend on the specific task. It is not shown in the provided code snippet.

This model architecture consists of multiple convolutional layers with pooling, a dropout layer for regularization, and fully connected layers for classification. It follows the typical pattern of convolutional neural networks commonly used in image classification tasks.

## Results
The final model achieves an accuracy of Training and validation 
#### 96% and 90%

## Dependencies
TensorFlow 2.0
NumPy
Pandas
Matplotlib
Scikit-learn
Usage
To train the model, run the train.py script with the appropriate command-line arguments:

## Credits
This project was inspired by the following paper:

Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639), 115–118. https://doi.org/10.1038/nature21056

## Contributing
Vignesh G
Neha Sharma,
Priyanka pandey
