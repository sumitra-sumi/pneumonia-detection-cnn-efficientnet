# Pneumonia Detection from Chest X-ray Images using CNN and EfficientNetB0

## Project Overview

This project develops a deep learning model for classifying chest X-ray images into two categories:

- NORMAL
- PNEUMONIA

The project compares a custom Convolutional Neural Network (CNN) baseline with EfficientNetB0 using transfer learning and fine-tuning.

The main objective is to evaluate whether transfer learning can improve pneumonia classification compared with a CNN trained from scratch.

---

## Dataset

The dataset consists of chest X-ray images belonging to two classes:

- NORMAL
- PNEUMONIA

### Dataset Split

| Dataset | Number of Images |
|---|---:|
| Training | 5,216 |
| Validation | 16 |
| Test | 624 |

The training dataset contains class imbalance between NORMAL and PNEUMONIA images. Class weights were therefore used during model training to reduce the effect of this imbalance.

---

## Data Preprocessing

The following preprocessing techniques were applied:

- Images resized to 224 × 224 pixels
- Image normalization
- Data augmentation
- Rotation
- Zoom
- Horizontal flipping
- Class weights to handle class imbalance

Data augmentation was applied to introduce controlled variations into the training images and improve the model's ability to generalize to unseen images.

---

## Model 1: CNN Baseline

A custom Convolutional Neural Network (CNN) was developed as the baseline model.

The CNN learns image features directly from the chest X-ray training dataset and provides a reference point for evaluating the improvement obtained through transfer learning.

### CNN Baseline Performance

- Test Accuracy: approximately 74.5%
- Pneumonia Precision: 0.78
- Pneumonia Recall: 0.83

The baseline CNN provided reasonable performance but had limited feature-extraction capability compared with the pretrained EfficientNetB0 model.

---

## Model 2: EfficientNetB0 Transfer Learning

EfficientNetB0 pretrained on ImageNet was used for transfer learning.

The pretrained convolutional base was initially frozen so that previously learned image features could be reused for the pneumonia classification task.

Additional classification layers were added on top of the pretrained model:

- Global Average Pooling
- Dense layer
- Dropout
- Sigmoid output layer

Transfer learning produced a substantial improvement compared with the CNN baseline.

### EfficientNetB0 Performance

- Test Accuracy: approximately 91%
- Pneumonia Precision: 0.93
- Pneumonia Recall: 0.93

---

## Fine-Tuning EfficientNetB0

After the initial transfer-learning stage, EfficientNetB0 was further fine-tuned.

The base model was made trainable while most layers remained frozen. The final layers of EfficientNetB0 were unfrozen and trained using a very low learning rate.

This allowed higher-level pretrained features to adapt more specifically to chest X-ray images while reducing the risk of making large changes to useful pretrained representations.

### Fine-Tuned EfficientNetB0 Performance

- Test Accuracy: approximately 90.7%
- Test Loss: approximately 0.227
- Pneumonia Precision: 0.92
- Pneumonia Recall: 0.93
- Pneumonia F1-score: 0.93

Fine-tuning maintained performance comparable to the original EfficientNetB0 transfer-learning model while allowing the higher-level features to adapt further to the chest X-ray dataset.

---

## Final Model Results

The fine-tuned EfficientNetB0 model was evaluated on the independent test dataset containing 624 chest X-ray images.

### Overall Performance

- Test Accuracy: approximately 91%
- Test Loss: approximately 0.227

### Class-wise Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| NORMAL | 0.88 | 0.87 | 0.88 |
| PNEUMONIA | 0.92 | 0.93 | 0.93 |

### Confusion Matrix Summary

- True Negatives (NORMAL correctly predicted): 203
- False Positives (NORMAL predicted as PNEUMONIA): 31
- False Negatives (PNEUMONIA predicted as NORMAL): 27
- True Positives (PNEUMONIA correctly predicted): 363

The pneumonia recall of 0.93 indicates that the model correctly identified most pneumonia cases in the test dataset.

False negatives remain particularly important because they represent pneumonia cases incorrectly classified as NORMAL.

---

## Model Comparison

| Model | Accuracy | Pneumonia Precision | Pneumonia Recall |
|---|---:|---:|---:|
| CNN Baseline | 74.5% | 0.78 | 0.83 |
| EfficientNetB0 | 91% | 0.93 | 0.93 |
| Fine-Tuned EfficientNetB0 | 90.7% | 0.92 | 0.93 |

### Key Observations

- EfficientNetB0 significantly outperformed the CNN baseline.
- Transfer learning produced a major improvement in classification accuracy.
- Pneumonia recall improved from 0.83 with the CNN baseline to 0.93 with EfficientNetB0.
- Fine-tuning maintained performance comparable to the initial EfficientNetB0 model.
- The results demonstrate the effectiveness of pretrained feature extraction for chest X-ray classification.

---

## Training Performance Visualization

Training and validation accuracy and loss curves were plotted to monitor model learning.

These plots help identify:

- Improvement in training accuracy
- Validation performance
- Changes in training and validation loss
- Possible overfitting
- Differences between training and validation behaviour

The small validation dataset should be considered when interpreting fluctuations in validation accuracy and loss.

---

## Sample Prediction Visualization

Sample test images were visualized together with their actual and predicted labels.

The analysis included examples of:

### Correct NORMAL

The actual image is NORMAL and the model correctly predicts NORMAL.

### Correct PNEUMONIA

The actual image is PNEUMONIA and the model correctly predicts PNEUMONIA.

### False Positive

The actual image is NORMAL, but the model predicts PNEUMONIA.

False positives may result in unnecessary additional investigation.

### False Negative

The actual image is PNEUMONIA, but the model predicts NORMAL.

False negatives are particularly important because a pneumonia case may be missed.

This visualization helps identify both correctly classified and misclassified cases, providing insight into model strengths and limitations.

---

## Visualization and Analysis

The project includes:

- Chest X-ray sample visualization
- Class distribution analysis
- Training and validation accuracy curves
- Training and validation loss curves
- Confusion matrix
- Classification report
- Sample prediction visualization
- Correct and misclassified case analysis
- Image intensity heatmap

These visualizations support understanding of the dataset, training behaviour and model prediction errors.

---

## Key Insights

- EfficientNetB0 substantially outperformed the CNN baseline.
- Transfer learning was highly effective for this image-classification task.
- The final model achieved approximately 91% test accuracy.
- Pneumonia recall reached approximately 93%.
- Class weights helped address class imbalance during training.
- Fine-tuning enabled higher-level pretrained features to adapt to the chest X-ray dataset.
- False-negative analysis remains important because missing pneumonia cases is a critical error in this classification task.

---

## Limitations

One of the major limitations of the dataset is the very small validation set containing only 16 images.

Because of the small validation sample:

- Validation accuracy may fluctuate considerably.
- Validation performance may not provide a stable estimate of generalization.
- Small changes in predictions can produce large changes in validation accuracy.

Another limitation is that the dataset may not represent the full diversity of chest X-rays encountered across different hospitals, imaging equipment and patient populations.

Therefore, the model should be considered an experimental deep-learning classification system rather than a clinically validated diagnostic system.

---

## Future Improvements

Future improvements could include:

- Using a larger and more diverse chest X-ray dataset
- Increasing the size of the validation dataset
- Exploring cross-validation strategies where appropriate
- Performing additional hyperparameter tuning
- Testing other pretrained architectures such as DenseNet or Vision Transformers
- Applying Grad-CAM for model interpretability
- Performing more detailed false-negative analysis
- Evaluating the model on external chest X-ray datasets
- Comparing computational cost and inference time across models
- Deploying the model as a web or mobile demonstration application

---

## Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab
- Jupyter Notebook

---

## Repository Structure

```text
pneumonia-detection-cnn-efficientnet/
│
├── Pneumonia_Detection_CNN_EfficientNetB0.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The chest X-ray dataset and large trained model files are not included in the repository.

---

## Installation

Clone this repository:

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd pneumonia-detection-cnn-efficientnet
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Open the Jupyter/Google Colab notebook and execute the cells sequentially.

---

## Conclusion

This project demonstrates the application of deep learning and transfer learning for pneumonia classification from chest X-ray images.

A custom CNN was first developed as a baseline model and achieved approximately 74.5% test accuracy. EfficientNetB0 using transfer learning substantially improved performance to approximately 91% accuracy.

Fine-tuning was subsequently performed to allow higher-level pretrained features to adapt further to the chest X-ray dataset. The fine-tuned model maintained approximately 91% test accuracy and achieved a pneumonia recall of 0.93.

The comparison demonstrates the advantage of transfer learning over the custom CNN baseline for this dataset. At the same time, the small validation set and remaining false-negative predictions highlight areas requiring further investigation before considering real-world medical use.

---

## Disclaimer

This project was developed for educational and research purposes only.

The model is not clinically validated and should not be used as a substitute for professional medical diagnosis.
