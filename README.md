# Image Classification

This repository contains code for various machine learning projects. Each project focuses on a specific dataset and utilizes different techniques for data analysis and modeling. Below is an overview of each project and the methods used.


## Dataset
The datasets used in the projects are:

- **MNIST**: A dataset of 60,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels.
- **CIFAR-10**: A dataset of 60,000 color images (32x32 pixels) across 10 different classes, with 6,000 images per class.
- **Iris**: A dataset of measurements for three different species of Iris flowers, including sepal length, sepal width, petal length, and petal width.

## Tools Used
* Pandas: A powerful data manipulation and analysis library used to handle data structures and perform data preprocessing tasks.
* NumPy: A library for numerical computations in Python, used for mathematical operations and array manipulation.
* Scikit-learn: A machine learning library that provides a wide range of algorithms and tools for data preprocessing, modeling, and evaluation.
* Matplotlib: A plotting library that provides a wide variety of visualization options for creating informative plots and figures.
* Seaborn: A data visualization library built on top of Matplotlib, used for creating attractive and informative statistical graphics.

## Methods Used
The following methods are employed in each project:

### Data Loading and Preprocessing
The dataset is loaded and preprocessed as follows:

- **MNIST**: The pixel values of the images are scaled between 0 and 1 by dividing them by 255.
- **CIFAR-10**: The pixel values of the images are normalized between 0 and 1.
- **Iris**: The features are extracted into an input matrix, while the target variable is encoded as integers.

### Modeling
The modeling techniques used for each project are as follows:

- **MNIST**: A neural network model is built using multiple dense layers. The model is trained using the Adam optimizer and evaluated based on accuracy.
- **CIFAR-10**: A convolutional neural network (CNN) model is constructed using convolutional and pooling layers, followed by fully connected layers. The model is trained using the Adam optimizer and evaluated based on accuracy.
- **Iris**: A support vector machine (SVM) classifier is trained on the Iris dataset. The model separates the data into different classes based on a hyperplane.

### Evaluation
The performance of the models is evaluated using the following metrics:

- **MNIST** and **CIFAR-10**: Accuracy is computed to measure the model's performance.
- **Iris**: Accuracy is computed, and a confusion matrix is generated for performance analysis.
---

## 📂 **MNIST Handwritten Digits Classification** 📂

-**Data Loading and Preprocessing**

The pixel values of the images are scaled between 0 and 1 by dividing them by 255. This normalization step ensures that the pixel values are within a consistent range for efficient training.

-**Neural Network Modeling**

A neural network model is constructed using multiple dense layers. The model architecture includes an input layer that flattens the 2D image into a 1D array, followed by several dense layers. The Adam optimizer is used to train the model, and accuracy is used as the evaluation metric.

-**Evaluation**

The trained model's performance is evaluated using accuracy, which measures the percentage of correctly classified digits. Additionally, a confusion matrix is generated to provide insights into the model's performance for each digit class.

---
## 📂 **CIFAR-10 Image Classification** 📂

-**Data Loading and Preprocessing**

The CIFAR-10 dataset is loaded, and the pixel values of the images are normalized between 0 and 1. This normalization step ensures that the pixel values are within a consistent range for efficient training.

-**Convolutional Neural Network Modeling**

A convolutional neural network (CNN) model is created, consisting of convolutional and pooling layers followed by fully connected layers. The model architecture captures the spatial relationships within the images. The Adam optimizer is used for training, and accuracy is used as the evaluation metric.

-**Evaluation**

The accuracy of the trained model is computed to measure its performance on the CIFAR-10 test set. Additionally, a confusion matrix is generated to visualize the model's classification performance for each class.

---
## 📂 **Iris Flower Classification** 📂

-**Data Loading and Preprocessing**

The Iris dataset is loaded, and the features are extracted into an input matrix. The target variable is encoded as integers to represent the different Iris species.

-**Random Forest Modeling**

A random forest classifier is trained on the Iris dataset. Random forests are ensemble learning methods that construct multiple decision trees during training and predict the class with the majority vote of the individual trees. The model is trained to predict the species of an Iris flower based on its measured features.

-**Evaluation**

The accuracy of the trained random forest model is computed to measure its performance on the test data. Additionally, a confusion matrix is generated to analyze the classification results and assess the model's performance.

