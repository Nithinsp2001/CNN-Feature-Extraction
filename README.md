# CNN-Feature-Extraction
# Task 1: Implement Edge Detection Using Convolution

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


# Load any sample grayscale image (you can replace the path with your own image)
image = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Error: Image not loaded. Please check the file path or if the image exists.")
else:
    # Define Sobel X and Y manually
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ], dtype=np.float32)

    # Apply filters
    sobel_x_output = cv2.filter2D(image, -1, sobel_x)
    sobel_y_output = cv2.filter2D(image, -1, sobel_y)

    # Display
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(sobel_x_output, cmap='gray')
    plt.title('Sobel X')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(sobel_y_output, cmap='gray')
    plt.title('Sobel Y')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# TASK 2: Max & Average Pooling (TensorFlow/Keras)

# Create 4x4 input matrix
input_matrix = np.random.randint(0, 10, (1, 4, 4, 1)).astype(np.float32)
print("Original 4x4 Matrix:\n", input_matrix[0, :, :, 0])

# MaxPooling
max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
max_result = max_pool(input_matrix)
print("\nMax Pooled 2x2:\n", max_result.numpy()[0, :, :, 0])

# AveragePooling
avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)
avg_result = avg_pool(input_matrix)
print("\nAverage Pooled 2x2:\n", avg_result.numpy()[0, :, :, 0])
