
# Cats vs Dogs Image Classification using CNN (TensorFlow)
This is an image classification project where the main goal was to build a basic Convolutional Neural Network (CNN) using TensorFlow and train it on the popular Cats vs Dogs dataset. The wanted to understand how deep learning models process images and learn patterns to distinguish between two classes, in this case, cats and dogs.

## About the Project
In this notebook, I used TensorFlow and TensorFlow Datasets to load the preprocessed Cats vs Dogs dataset. I resized all the images to a fixed size (160x160) and normalized the pixel values between 0 and 1 to make the training smoother. After that, I built a small CNN model using Keras layers, trained it for 5 epochs, and checked its performance on a validation set.
This project helped me understand how CNNs work, what layers like `Conv2D` and `MaxPooling2D` do, and how image data is handled in TensorFlow.

## Dataset Used

- **Name:** `cats_vs_dogs` from `tensorflow_datasets`
- **Split:** 80% training, 20% testing
- **Preprocessing done:**
  - Resized all images to 160x160 pixels
  - Normalized image pixel values to range [0, 1]
  - Used batching and shuffling for training

## About the model
**Conv2D Layer (32 filters, ReLU):**
This is the first layer of the model that applies 32 filters (small matrices) to the input images. These filters slide across the image and extract low-level features like edges, lines, or corners. The activation function used is ReLU (Rectified Linear Unit), which adds non-linearity to the model, helping it learn complex patterns.

**MaxPooling2D Layer:**
After the convolution, the max pooling layer reduces the size (dimensions) of the feature maps while keeping the most important information. It basically downsamples the image to reduce computation and help the model focus on key features. It also helps in reducing overfitting.

**Second Conv2D Layer (64 filters, ReLU):**
Now that some basic features are learned, we add another convolutional layer with more filters (64 this time) to learn more complex and detailed patterns in the image like textures or shapes. Again, ReLU is used for non-linearity.

**Second MaxPooling2D Layer:**
Just like before, this pooling layer reduces the spatial size of the output from the previous convolutional layer. This helps reduce the number of parameters and computations, which is essential for efficiency.

**Flatten Layer:**
Convolutional and pooling layers output 2D feature maps. The flatten layer converts these into a single long 1D vector, which can be fed into fully connected layers (Dense layers). Think of it as preparing the features for classification.

**Dense Layer (128 units, ReLU):**
This fully connected layer acts like a brain of the network where the extracted features are processed. It has 128 neurons with ReLU activation, allowing the model to learn complex combinations of the features for classification.

**Final Dense Layer (1 unit, Sigmoid):**
The last layer has only one neuron because of binary classification (cat or dog). The sigmoid activation function outputs a value between 0 and 1, where values closer to 0 mean “cat” and closer to 1 mean “dog” (depending on the training labels).

## Training Details
**Optimizer:Adam**
- Adam optimizer is chosen because it adapts the learning rate during training and generally performs well for image classification tasks without requiring a lot of manual tuning. It's efficient and combines the benefits of both RMSprop and SGD with momentum.
- **Loss Function: Binary Crossentropy**
- Since this is a binary classification problem (cat vs dog), binary crossentropy is the appropriate loss function. It measures how far the predicted probabilities are from the actual labels (0 or 1), and helps guide the model’s learning during backpropagation.
- **Epochs:5**
- The model was trained for 5 epochs, meaning it went through the entire training dataset 5 times. This is a reasonable starting point to see how well the model is learning without overfitting.
- **Batch Size:32**
- The data was divided into batches of 32 images. This size is a good balance between training speed and performance, and it helps the model generalize better.
- **Evaluation Metrics:Accuracy**
Accuracy was used as the main metric to evaluate how well the model performs. It gives a clear idea of the percentage of correct predictions on both training and validation data.

## What I Learned
This project taught me how to load and preprocess image datasets using TensorFlow Datasets, normalize and resize images properly, and build a simple CNN model for binary classification. I also learned how to compile the model with the right optimizer and loss function, handle training with batching and validation, and understand the overall structure of a deep learning workflow in TensorFlow.

## How to Open in Google Colab

Link for running this notebook easily on Colab ig given below:

## Open in Google Colab

## Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Diva015/Cats-vs-Dogs-Image-Classifier-Using-CNN-with-TensorFlow-/blob/main/Copy_of_Cats_vs_Dogs_Image_Classifier%28Using_CNN_with_TensorFlow%29.ipynb)

