
# Neural Network Operations & Data Preprocessing

## Student Info
- Name: Nithin Lakide
- Student ID: 700764562
- Course: Deep Learning & Neural Networks (Home Assignment 2)


## Project Overview
This repository contains four tasks demonstrating convolution, pooling, edge detection, and data preprocessing using TensorFlow, OpenCV, and NumPy.

## Installation & Setup
Ensure you have Python installed. Install required libraries:
```bash
pip install numpy tensorflow scikit-learn matplotlib opencv-python

## Task 1: Convolution with Different Stride and Padding
Objective
To understand how stride and padding affect convolution operations.
Steps
- Define a 5×5 input matrix.
- Define a 3×3 kernel (edge detection kernel).
- Perform convolution using:
- Stride = 1, Padding = 'VALID'
- Stride = 1, Padding = 'SAME'
- Stride = 2, Padding = 'VALID'
- Stride = 2, Padding = 'SAME'
- Print the output feature maps for each configuration


## Task 2: Edge detection using Sobel filters, which are commonly used in image processing to highlight intensity changes along different directions.
Detailed Description: Edge Detection Using Sobel Filters
Objective
To apply Sobel filters for detecting edges in an image, using OpenCV (cv2).
Steps
- Load a grayscale image (any sample image).
- Apply a Sobel filter in the X-direction to detect vertical edges.
- Apply a Sobel filter in the Y-direction to detect horizontal edges.
- Display the original image and filtered edge-detected images using matplotlib



## Task 3: Max Pooling & Average Pooling
Objective:
To demonstrate Max Pooling and Average Pooling, which help reduce spatial dimensions while preserving essential features.
Steps
- Create a random 4×4 matrix as input data.
- Apply a 2×2 Max Pooling operation.
- Apply a 2×2 Average Pooling operation.
- Print the original matrix, max-pooled matrix, and average-pooled matrix.


## Task 4: Data Preprocessing – Normalization vs. Standardization
Objective
To preprocess the Iris dataset using Min-Max Normalization and Z-score Standardization, then compare their effects on model performance.
Steps
- Load the Iris dataset.
- Apply Min-Max Normalization to scale features between 0 and 1.
- Apply Z-score Standardization to center the data around a mean of 0 and a standard deviation of 1.
- Visualize distributions using histograms.
- Train a Logistic Regression model before and after preprocessing and compare accuracy.
