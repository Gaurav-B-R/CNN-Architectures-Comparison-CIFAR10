# Comparison of CNN Architectures on CIFAR-10 Dataset

This repository presents a comparison of two Convolutional Neural Network (CNN) architectures trained on the CIFAR-10 dataset. The project aims to evaluate the impact of adding an additional convolutional layer on the network's performance.

## Dataset
The CIFAR-10 dataset is used in this project, containing 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images.

## Architectures Compared

1. **Original CNN (Shallower Network):**
   - 2 convolutional layers with ReLU activation.
   - Max pooling after each convolutional layer.
   - Two fully connected layers before the final output.
   - Total training accuracy: **78.88%**  
   - Total testing accuracy: **60.05%**

2. **Modified CNN (Deeper Network):**
   - 3 convolutional layers with ReLU activation.
   - Max pooling after each convolutional layer.
   - Two fully connected layers before the final output.
   - Total training accuracy: **58.53%**  
   - Total testing accuracy: **54.80%**

## Project Structure

- **`cnn_comparison.ipynb`**: The Jupyter notebook containing the implementation and evaluation of both CNN architectures.
- **`data/`**: Directory to store the CIFAR-10 dataset (downloaded automatically).
- **`plots/`**: Contains plots of training loss and accuracy for each architecture.

## Installation & Setup

To run the project locally, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/CNN-Architectures-Comparison-CIFAR10.git
    cd CNN-Architectures-Comparison-CIFAR10
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook to train the models and compare the results:
    ```bash
    jupyter notebook cnn_comparison.ipynb
    ```

## Results

### Accuracy Comparison

| Architecture     | Training Accuracy | Testing Accuracy |
|------------------|-------------------|------------------|
| Original Network | 78.88%            | 60.05%           |
| Modified Network | 58.53%            | 54.80%           |

The results show that the original, shallower network performed better in both training and testing accuracy compared to the deeper network.

## Conclusion

This project demonstrates that adding an additional convolutional layer did not improve the performance of the CNN on the CIFAR-10 dataset. While deeper networks are often beneficial for complex datasets, this experiment shows that more layers can lead to overfitting or increased complexity without performance gains on smaller datasets.

## Future Work

- Experiment with regularization techniques to prevent overfitting.
- Test different activation functions and optimizers to improve performance.
