# Cardiovascular Disease Prediction
## Project Overview
This project is part of an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. The primary goal is to predict whether a patient has a 10-year risk of future coronary heart disease (CHD) using various demographic, behavioral, and medical risk factors. The dataset consists of over 4,000 records and 15 attributes.

## Dataset
The dataset contains the following attributes:

- Age
- Sex
- Education
- Current Smoker
- Cigarettes Per Day
- BP Meds
- Prevalent Stroke
- Prevalent Hypertension
- Diabetes
- Total Cholesterol
- Systolic BP
- Diastolic BP
- BMI
- Heart Rate
- Glucose

## Project Structure
The project is organized into several key sections:
### 1. Problem Statement
This section provides a clear definition of the problem, outlining the objectives of the project. The goal is to predict the 10-year risk of coronary heart disease (CHD) using a variety of patient attributes.

### 2. Importing Libraries
In this section, all necessary libraries are imported. These include:

- 'pandas' and 'numpy' for data manipulation and analysis.
- 'scikit-learn' for machine learning algorithms and model evaluation.
- 'matplotlib' and 'seaborn' for data visualization.
- 'warnings' to ignore warnings for cleaner outputs.
  
### 3. Data Gathering
The dataset is loaded and initial data inspection is performed. This includes:

- Loading the dataset into a pandas DataFrame.
- Displaying the first few rows of the dataset.
- Checking for missing values and basic statistics of the dataset.

### 4. Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) is a crucial step in understanding the dataset and uncovering patterns, anomalies, and relationships between variables. In this project, EDA includes visualizations and statistical analyses to gain insights into the data.
#### - Gender Distribution
Count Plot: Shows the distribution of patients by gender.
<img width="856" alt="gender_distribution" src="https://github.com/Akshadakambale/ML_CHD_project/assets/166009612/6050547d-46f3-4441-b4a0-12a991a8abf4">
There are more than 1,750 female patients and approximately 1,500 male patients.

#### - CHD Risk Distribution
Count Plot: Displays the distribution of patients with and without CHD risk.
<img width="658" alt="chd_risk_distribution" src="https://github.com/Akshadakambale/ML_CHD_project/assets/166009612/413c6e41-8b6d-4b99-9204-c90c23903e94">
Approximately 2,800 patients have no risk of CHD, while around 500 patients are at risk.

#### - CHD Risk by Gender
Count Plot: Shows the distribution of CHD risk among males and females.
<img width="672" alt="chd_risk_by_gender" src="https://github.com/Akshadakambale/ML_CHD_project/assets/166009612/daa577a3-7a25-4b3d-b5bb-67a570437152">
Approximately 250 females and 300 males are at risk of CHD.

#### - Smoking Status and CHD Risk
Bar Plot: Illustrates the CHD risk based on smoking status.
<img width="659" alt="smoking_status_and_chd_risk" src="https://github.com/Akshadakambale/ML_CHD_project/assets/166009612/a34d98c8-87d5-4027-9ea7-4e0ddae1c895">
Patients who smoke have a higher risk of CHD compared to non-smokers.

#### - Outliers in Data
Box Plot: Identifies outliers in the dataset.
Values above 100 and below 50 in certain attributes are acting as outliers.

#### -Heart Rate by Hypertension Status
Box Plot: Examines the patients without hypertension tend to have a slightly higher median heart rate compared to patients with hypertension.The interquartile range (IQR) indicates the variability in heart rate within each group.
<img width="838" alt="heart_rate_by_hypertension" src="https://github.com/Akshadakambale/ML_CHD_project/assets/166009612/1f9fbb13-fab9-4c23-9f92-987eb1a1bd08">
The interquartile range (IQR) is within 70 to 87 heart rate for the patients having History of Hypertension, suggesting that most hypertensive patients have heart rates within this range, which is generally slightly higher than non-hypertensive patients.

#### - Stroke History and CHD Risk
Scatter Plot: Observes the prevalence of stroke across different age groups.
<img width="851" alt="prevalent_heart_stroke" src="https://github.com/Akshadakambale/ML_CHD_project/assets/166009612/d42fca5c-1c49-4f82-bdd3-26b68bd65243">
Patients with a history of stroke and age above 50 years have a higher risk of CHD.

#### - Blood Pressure and Hypertension
Scatter Plot: Displays the relationship between systolic blood pressure and hypertension.
<img width="659" alt="blood_pressure_and_hypertension" src="https://github.com/Akshadakambale/ML_CHD_project/assets/166009612/c6b0f153-bf92-4e63-bde0-7a5e69f207bf">
There is a strong positive correlation between high systolic blood pressure and hypertension.
The scatter plot shows a separation between individuals with and without hypertension based on their systolic blood pressure readings.

#### - Correlation Heatmap
Heatmap: Shows the correlation matrix of the dataset.
<img width="859" alt="heatmap" src="https://github.com/Akshadakambale/ML_CHD_project/assets/166009612/f1173228-d2c9-4b66-809d-926c644ec0ba">
Identifies the strength of relationships between different attributes.
Helps in understanding which features are strongly correlated with the target variable (CHD risk).

### Key Findings from EDA
- Gender and CHD Risk: Gender does not show a significant imbalance in CHD risk.
- Smoking: Smoking is a significant risk factor for CHD.
- Age: Older age groups (50-70 years) have a higher prevalence of CHD.
- Hypertension and Blood Pressure: High blood pressure is strongly correlated with hypertension, which in turn is a risk factor for CHD.
- Outliers: Certain attributes have outliers that need to be handled during preprocessing.

### 5. Data Preprocessing
This section involves preparing the data for model building. Key steps include:

#### Handling Missing Values :
Identify Missing Values: Use isnull() and sum() functions to identify missing values in the dataset.
#### Imputation: Missing values are imputed using appropriate strategies:
- For numerical attributes, missing values are filled using the mean or median of the respective column.
- For categorical attributes, missing values are filled using the mode of the respective column.
#### Encoding Categorical Variables :
- Label Encoding: Converts categorical text data into numerical data. For example, 'Male' and 'Female' are encoded as 0 and 1 respectively.
- One-Hot Encoding: Creates binary columns for each category in a categorical feature. This is useful for features with more than two categories.
#### Scaling Numerical Features :
- Normalization: Rescales the numerical attributes to a standard range (typically 0 to 1). This ensures that all features contribute equally to the model. Techniques such as Min-Max scaling or StandardScaler from scikit-learn are used.

### 6. Model Building Implementation
#### Data Splitting
Before training the models, we need to split the dataset into training and testing sets. This ensures that we can evaluate the model's performance on unseen data. The train_test_split function from scikit-learn is used for this purpose.

#### 1. Logistic Regression
Overview: Logistic Regression is a linear model used for binary classification. It predicts the probability of a binary outcome based on one or more predictor variables.

##### Implementation:

- Initialization: Import the LogisticRegression class from scikit-learn.
- Training: Fit the model on the training data using the fit() method.
- Prediction: Use the predict() method to make predictions on the test data.
- Evaluation: Assess the model's performance using metrics like accuracy, precision, recall, and ROC-AUC score.
 
#### 2. K-Nearest Neighbors (KNN)
Overview: KNN is a non-parametric method that classifies a data point based on the majority class of its k-nearest neighbors.

##### Implementation:

- Initialization: Import the KNeighborsClassifier class from scikit-learn.
- Training: Fit the model on the training data using the fit() method.
- Prediction: Use the predict() method to classify the test data.
- Evaluation: Assess the model's performance using metrics like accuracy, precision, recall, and ROC-AUC score.

#### 3. Decision Tree
Overview: Decision Trees split the data into branches based on feature values to make predictions. Each internal node represents a test, each branch represents an outcome, and each leaf node represents a class label.

##### Implementation:

- Initialization: Import the DecisionTreeClassifier class from scikit-learn.
- Training: Fit the model on the training data using the fit() method.
- Prediction: Use the predict() method to make predictions on the test data.
- Evaluation: Assess the model's performance using metrics like accuracy, precision, recall, and ROC-AUC score.

  
#### 4. Random Forest
Overview: Random Forest is an ensemble method that constructs multiple decision trees and merges their predictions to improve accuracy and control over-fitting.

##### Implementation:

- Initialization: Import the RandomForestClassifier class from scikit-learn.
- Training: Fit the model on the training data using the fit() method.
- Prediction: Use the predict() method to classify the test data.
- Evaluation: Assess the model's performance using metrics like accuracy, precision, recall, and ROC-AUC score.

### 7. Model Evaluation

After training machine learning models, it's crucial to evaluate their performance using various metrics. These metrics help us understand how well the model is performing and whether it's suitable for the task at hand.

* Logistic Regression (Original):
Precision = 0.70, Recall = 0.04,Accuracy = 83%
Confusion Matrix:
[[792 3]
[155 7]]

* K-Nearest Neighbors (Original):
Precision = 0.28, Recall = 0.05,Accuracy = 81%
Confusion Matrix:
[[774 21]
 [154 8]]

* Decision Tree (Original):
Precision = 0.26, Recall = 0.24,Acuracy = 75%
Confusion Matrix:
[[682 113]
 [123 39]]

* Random Forest (Original):
Precision = 1.0, Recall = 0.14,Accuracy = 83%
Confusion Matrix:
[[793 2]
 [157 5]]

These results indicate that the models have higher precision but lower recall, meaning they are making fewer false positives but missing more positive instances.So, we need to balance dataset to improve the recall.

### 8. Handling Imbalanced Dataset with SMOTE
Techniques such as SMOTE (Synthetic Minority Over-sampling Technique) are used to address class imbalance. The impact of these techniques on model performance is assessed. SMOTE (Synthetic Minority Over-sampling Technique) is an oversampling method used to balance class distribution in datasets. It works by generating synthetic samples for the minority class by interpolating between existing minority class samples.

* Steps to Use SMOTE
  * Install and Import SMOTE: Install the imbalanced-learn library if not already installed, and import SMOTE.
  * Apply SMOTE: Use SMOTE to create a balanced dataset by oversampling the minority class.
  *  Train and Evaluate Models: Train the models on both the original and balanced datasets and compare their performance.

* Logistic Regression (Balanced):
Precision = 0.32, Recall = 0.49, Accuracy = 73%
Confusion Matrix:
[[625 170]
 [ 82 80]]

* K-Nearest Neighbors (Balanced):
Precision = 0.23, Recall = 0.51, Accuracy = 62%
Confusion Matrix:
[[519 276]
 [ 80 82]]

* Decision Tree (Balanced):
Precision = 0.25, Recall = 0.37,Accuracy = 70%
Confusion Matrix:
[[616 179]
 [ 102 60]]

* Random Forest (Balanced):
Precision = 0.86, Recall = 0.93,Accuracy = 72%
Confusion Matrix:
[[616 179]
 [ 85 77]]
 
* After Balancing (Using SMOTE):

Models show improved recall at the expense of precision, meaning they can identify more positive instances but also produce more false positives.
Random Forest continues to perform well with a good balance between precision and recall, indicating it is robust to the effects of balancing.
Logistic Regression and Decision Tree also show improved recall, indicating their enhanced ability to detect positive cases after balancing.

### 9. Conclusion
Summarizes the findings and highlights the best-performing model. The Random Forest model is identified as the best performer with an accuracy of 72% and a recall of 93% for predicting CHD risk.

## Results
The Random Forest model showed the best performance with an accuracy of 72% and a recall of 93% for predicting the risk of CHD.
Applying SMOTE improved the recall for minority classes across various models.
