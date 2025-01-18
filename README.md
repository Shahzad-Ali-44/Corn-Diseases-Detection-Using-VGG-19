# Corn Diseases Detection Using VGG-19

## Overview
This project leverages the VGG-19 deep learning model for accurate detection of corn diseases. Fine-tuned on a curated dataset, the model achieves an impressive **94.24% accuracy** on the test set, providing a robust tool for agricultural disease diagnosis.

## Features
- **High Accuracy**: Achieves 94.24% accuracy on test data.
- **VGG-19 Backbone**: Uses the powerful VGG-19 architecture with fine-tuning for superior performance.
- **Data Augmentation**: Incorporates advanced augmentation techniques to improve generalization.
- **Visual Insights**: Confusion matrix, classification report, and random predictions provide detailed model evaluation.

## Dataset
- **Source**: Custom dataset of corn leaf images with healthy and diseased categories.
- **Classes**: Dataset comprises **6 classes**, representing different corn diseases and healthy plants.
- **Preprocessing**: Images resized to **224x224 pixels** and normalized for VGG-19.

## Methodology
1. **Data Preparation**:
   - Utilized `ImageDataGenerator` for rescaling, augmentation, and preprocessing.
   - Created separate training and testing sets from the dataset.

2. **Model Architecture**:
   - Base model: Pre-trained VGG-19 with ImageNet weights.
   - Added custom layers:
     - Flatten layer.
     - Dense layers with ReLU activation.
     - Dropout for regularization.
     - Final softmax layer for classification.

3. **Training**:
   - Optimizer: Adam with a learning rate of 0.001.
   - Loss function: Categorical cross-entropy.
   - Epochs: 50 with early stopping and learning rate reduction on plateau.

4. **Evaluation**:
   - Achieved **94.24% test accuracy**.
   - Confusion matrix and classification report provided detailed evaluation metrics.

## Dependencies
- Python 3.7+
- TensorFlow/Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- pandas

## Results
- **Accuracy**: 94.24%
- **Confusion Matrix**: Highlights model performance across all classes.
- **Visualizations**: Training/validation accuracy and loss graphs.

## Visualizations
- **Confusion Matrix**: Displays class-wise performance.
- **Training Metrics**: Line graphs showing accuracy and loss trends.
- **Predictions**: Sample images with actual labels, predicted labels, and confidence scores.

## Future Improvements
- Expand dataset to include more disease categories and variations.
- Experiment with other deep learning architectures for comparison.
- Deploy the model in a web or mobile application for real-time use.

## Contributing
Contributions are welcome! Fork the repository, implement your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The contributors of the corn leaf dataset.
- TensorFlow/Keras for providing robust deep learning tools.
- The open-source community for supporting essential libraries.