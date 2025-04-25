# Facial Keypoints Detection Using Deep Learning

This project detects 15 key facial landmarks (30 coordinate points) on grayscale images using Convolutional Neural Networks (CNNs).

## Problem Type
- **Task:** Supervised Regression
- **Input:** 96x96 grayscale images
- **Output:** 30 floating-point keypoint values (x, y coordinates)

## Approach

### 1. Exploratory Data Analysis (EDA)
- Visualized missing keypoints
- Found missingness patterns (not random)
- Analyzed pixel value distribution

### 2. Data Preprocessing
- Converted image strings to 96x96 float arrays
- Normalized pixel values to [0, 1]
- Handled missing values by creating region-specific models

### 3. Modeling

#### Baseline CNN Model
- 2 Convolutional + Pooling layers
- Flatten + Dense + Dropout layers
- Final Dense(30) output layer
- Loss: Mean Squared Error (MSE)
- Evaluation Metric: Root Mean Squared Error (RMSE)

#### Region-Specific Models
- Split keypoints into regions (eyes/nose, mouth/eyebrows)
- Trained separate CNNs per region
- Reduced overfitting and improved validation RMSE

#### Multi-Branch CNN
- Separate convolutional branches for each facial region
- Merged features before output layer
- Improved modularity, interpretability, and robustness

### 4. Results
- Baseline model performed well but overfitted slightly
- Region-specific and multi-branch models improved generalization

## Future Work
- Implement heatmap-based landmark regression
- Use ensemble models to boost performance

## Requirements
- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib

## How to Run
1. Clone the repository or download the notebook.
2. Install the required packages.
3. Open the notebook and run all cells.
