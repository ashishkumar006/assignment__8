# CIFAR-10 Classification with Custom CNN Architecture

This project focuses on training a custom Convolutional Neural Network (CNN) to classify the CIFAR-10 dataset. The architecture includes various techniques such as **Depthwise Separable Convolutions**, **Dilated Convolutions**, and **Global Average Pooling (GAP)**. The goal is to achieve **85% accuracy** on the CIFAR-10 dataset with a parameter count under **200k**.


## Architecture Overview

The model follows the **C1C2C3C40** architecture with the following key features:
1. **No MaxPooling**: Instead of MaxPooling, strided convolutions are used to reduce the spatial size of the feature maps.
2. **Depthwise Separable Convolution**: One of the layers uses depthwise separable convolutions for computational efficiency.
3. **Dilated Convolution**: Some layers use dilated convolutions to increase the receptive field without reducing spatial size.
4. **Global Average Pooling (GAP)**: GAP is applied before a fully connected layer to make the model more efficient.
5. **Albumentations**: Data augmentation is applied using the Albumentations library, including horizontal flip, shift-scale-rotate, and coarse dropout.






## Model Overview

The model architecture is based on a custom Convolutional Neural Network (CNN) designed for the CIFAR-10 dataset. The architecture follows the **C1C2C3C40** structure with several unique features to meet the specified requirements. Below is a breakdown of the architecture:

### Architecture Components

1. **C1 - Initial Convolutional Layer:**
   - A 3x3 convolutional layer with padding `1` is used as the first layer.
   - **Output Channels:** 24
   - **Kernel Size:** 3x3
   - This layer captures basic features of the image.

2. **C2 - Depthwise Separable Convolution:**
   - The next layer employs **Depthwise Separable Convolution** to reduce computation complexity.
   - **Depthwise Convolution:** 3x3 kernel with padding `2` and **dilation** set to `2`.
   - **Pointwise Convolution:** A 1x1 convolution is used to combine the output of the depthwise convolution.
   - **Output Channels:** 48

3. **C3 - Strided Convolution:**
   - A 3x3 convolution with **stride=2** is used to reduce the spatial resolution while increasing the depth.
   - **Output Channels:** 64
   - **Kernel Size:** 3x3
   - The stride reduces the size of the feature maps.

4. **C4 - Dilated Convolution:**
   - This layer uses **Dilated Convolutions** with a kernel size of `3x3`, and dilation set to `4` to increase the receptive field without reducing the spatial dimensions.
   - **Output Channels:** 80
   - **Kernel Size:** 3x3
   - **Dilation:** 4

5. **C5 - Final Convolutional Layer with Stride 2:**
   - The last convolutional layer uses **stride=2** to further reduce the spatial size of the feature maps.
   - **Output Channels:** 96
   - **Kernel Size:** 3x3
   - **Stride:** 2
   - **Dilation:** 6

6. **Global Average Pooling (GAP):**
   - The output from the last convolutional layer is passed through **Global Average Pooling** to reduce the spatial dimensions to a single value per feature map.
   - GAP is an important step for dimensionality reduction.

7. **Fully Connected Layer:**
   - The output from GAP is passed through a **fully connected (FC) layer** to predict the class probabilities for the CIFAR-10 dataset (10 classes).
   - **Output Units:** 10 (for the 10 classes in CIFAR-10)

### Receptive Field (RF)

The final receptive field (RF) is calculated based on the layers used, ensuring that the total receptive field 49 pixels.

- The receptive field of the model grows with the dilation in the convolutional layers, and by the end of the model, the receptive field surpasses 44 pixels.

### Key Features

1. **No MaxPooling:** The model avoids using traditional MaxPooling layers. Instead, **strided convolutions** and **dilated convolutions** are used to reduce the spatial size of the feature maps.
   
2. **Depthwise Separable Convolutions:** One of the layers uses depthwise separable convolutions to make the network more computationally efficient, reducing the number of parameters.

3. **Dilated Convolutions:** **Dilated convolutions** are used in one of the layers to increase the receptive field without reducing spatial dimensions, improving the model’s ability to capture large-scale patterns.

4. **Global Average Pooling (GAP):** GAP is used before the final fully connected layer to reduce overfitting and improve generalization. This technique reduces the spatial dimensions of the output feature map to a single value per channel.

5. **Data Augmentation:** The model uses the **Albumentations** library for data augmentation, applying techniques like:
   - **Horizontal Flip**
   - **ShiftScaleRotate**
   - **CoarseDropout** (with specific parameters for the number of holes, size, and fill value)

### Total Parameters

The model has been carefully designed to keep the total number of parameters under **200k**, ensuring that it is both computationally efficient and capable of achieving good performance on the CIFAR-10 dataset.

---
### Results

This custom CNN architecture aims to achieve **85% accuracy** on the CIFAR-10 dataset while ensuring the model remains efficient with fewer than 200k parameters. The network is trained using **Cross-Entropy Loss** and optimized using **SGD** with momentum.
It acheives **85.19% accuracy** at epoch 34

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



