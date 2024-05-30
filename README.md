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
There are more than 1,750 female patients and approximately 1,500 male patients.
#### - CHD Risk Distribution
Count Plot: Displays the distribution of patients with and without CHD risk.
Approximately 2,800 patients have no risk of CHD, while around 500 patients are at risk.
#### - CHD Risk by Gender
Count Plot: Shows the distribution of CHD risk among males and females.
Approximately 250 females and 300 males are at risk of CHD.
#### - Smoking Status and CHD Risk
Bar Plot: Illustrates the CHD risk based on smoking status.
Patients who smoke have a higher risk of CHD compared to non-smokers.
#### - Outliers in Data
Box Plot: Identifies outliers in the dataset.
Values above 100 and below 50 in certain attributes are acting as outliers.
#### - Age and CHD Risk
Box Plot: Examines the distribution of CHD risk across different age groups.
Majority of patients with CHD fall within the age range of 50 to 60 years.
Some patients with CHD are older, ranging from 60 to 70 years old.
#### - Stroke History and CHD Risk
Scatter Plot: Observes the prevalence of stroke across different age groups.
Patients with a history of stroke and age above 50 years have a higher risk of CHD.
#### - Blood Pressure and Hypertension
Scatter Plot: Displays the relationship between systolic blood pressure and hypertension.
There is a strong positive correlation between high systolic blood pressure and hypertension.
The scatter plot shows a separation between individuals with and without hypertension based on their systolic blood pressure readings.
#### - Correlation Heatmap
Heatmap: Shows the correlation matrix of the dataset.
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
### Data Splitting
Before training the models, we need to split the dataset into training and testing sets. This ensures that we can evaluate the model's performance on unseen data. The train_test_split function from scikit-learn is used for this purpose.
from sklearn.model_selection import train_test_split

# Assuming x is the feature matrix and y is the target vector
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

### Model Building Implementation
#### 1. Logistic Regression
Overview: Logistic Regression is a linear model used for binary classification. It predicts the probability of a binary outcome based on one or more predictor variables.

##### Implementation:

- Initialization: Import the LogisticRegression class from scikit-learn.
- Training: Fit the model on the training data using the fit() method.
- Prediction: Use the predict() method to make predictions on the test data.
- Evaluation: Assess the model's performance using metrics like accuracy, precision, recall, and ROC-AUC score.
  from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# Initialize the model
log_reg = LogisticRegression()

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions
y_pred_lr = log_reg.predict(X_test)

# Evaluate the model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Logistic Regression Precision:", precision_score(y_test, y_pred_lr))
print("Logistic Regression Recall:", recall_score(y_test, y_pred_lr))
print("Logistic Regression ROC-AUC Score:", roc_auc_score(y_test, y_pred_lr))

#### 2. K-Nearest Neighbors (KNN)
Overview: KNN is a non-parametric method that classifies a data point based on the majority class of its k-nearest neighbors.

##### Implementation:

- Initialization: Import the KNeighborsClassifier class from scikit-learn.
- Training: Fit the model on the training data using the fit() method.
- Prediction: Use the predict() method to classify the test data.
- Evaluation: Assess the model's performance using metrics like accuracy, precision, recall, and ROC-AUC score.

from sklearn.neighbors import KNeighborsClassifier

# Initialize the model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred_knn = knn.predict(X_test)

# Evaluate the model
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNN Precision:", precision_score(y_test, y_pred_knn))
print("KNN Recall:", recall_score(y_test, y_pred_knn))
print("KNN ROC-AUC Score:", roc_auc_score(y_test, y_pred_knn))

#### 3. Decision Tree
Overview: Decision Trees split the data into branches based on feature values to make predictions. Each internal node represents a test, each branch represents an outcome, and each leaf node represents a class label.

##### Implementation:

- Initialization: Import the DecisionTreeClassifier class from scikit-learn.
- Training: Fit the model on the training data using the fit() method.
- Prediction: Use the predict() method to make predictions on the test data.
- Evaluation: Assess the model's performance using metrics like accuracy, precision, recall, and ROC-AUC score.

  from sklearn.tree import DecisionTreeClassifier

# Initialize the model
dt = DecisionTreeClassifier()

# Train the model
dt.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt.predict(X_test)

# Evaluate the model
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Decision Tree Precision:", precision_score(y_test, y_pred_dt))
print("Decision Tree Recall:", recall_score(y_test, y_pred_dt))
print("Decision Tree ROC-AUC Score:", roc_auc_score(y_test, y_pred_dt))

#### 4. Random Forest
Overview: Random Forest is an ensemble method that constructs multiple decision trees and merges their predictions to improve accuracy and control over-fitting.

##### Implementation:

- Initialization: Import the RandomForestClassifier class from scikit-learn.
- Training: Fit the model on the training data using the fit() method.
- Prediction: Use the predict() method to classify the test data.
- Evaluation: Assess the model's performance using metrics like accuracy, precision, recall, and ROC-AUC score.

  from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf = RandomForestClassifier(n_estimators=100)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Precision:", precision_score(y_test, y_pred_rf))
print("Random Forest Recall:", recall_score(y_test, y_pred_rf))
print("Random Forest ROC-AUC Score:", roc_auc_score(y_test, y_pred_rf))

### 7. Model Evaluation

After training machine learning models, it's crucial to evaluate their performance using various metrics. These metrics help us understand how well the model is performing and whether it's suitable for the task at hand.

#### Accuracy
- Definition: Accuracy measures the proportion of correctly classified instances out of the total instances.
- Usage: It's a general metric that indicates overall model performance but may not be suitable for imbalanced datasets.
#### Precision
- Definition: Precision measures the accuracy of positive predictions, indicating how many of the predicted positive instances are actually positive.
- Usage: It's useful when the cost of false positives is high, and we want to minimize false positive predictions.
#### Recall (Sensitivity or True Positive Rate)
- Definition: Recall measures the ability of the model to identify positive instances correctly, indicating how many actual positives were correctly predicted.
- Usage: It's crucial when the cost of false negatives is high, and we want to minimize missed positive predictions.
#### ROC-AUC Score
- Definition: ROC-AUC (Receiver Operating Characteristic - Area Under Curve) is a performance metric for binary classification models. It plots the true positive rate against the false positive rate.
- Usage: It provides an overall measure of how well the model can distinguish between positive and negative classes, irrespective of the chosen threshold.
#### Confusion Matrix
- Definition: A confusion matrix is a table that shows the true positives, false positives, true negatives, and false negatives of a classification model.
- Usage: It provides detailed insights into the model's performance across different classes and helps identify where the model is making errors.

#### Interpretation of Results
- Accuracy: High accuracy indicates overall correct predictions but may not be informative for imbalanced datasets.
- Precision: High precision means fewer false positive predictions.
- Recall: High recall indicates fewer false negatives, ensuring most positive instances are correctly predicted.
- ROC-AUC Score: A score close to 1 indicates a good model that can distinguish between classes effectively.
- Confusion Matrix: Provides detailed information about true positives, false positives, true negatives, and false negatives, helping understand the model's errors.

### 8. Handling Class Imbalance
Techniques such as SMOTE (Synthetic Minority Over-sampling Technique) are used to address class imbalance. The impact of these techniques on model performance is assessed.

### 9. Conclusion
Summarizes the findings and highlights the best-performing model. The Random Forest model is identified as the best performer with an accuracy of 72% and a recall of 93% for predicting CHD risk.

## Results
The Random Forest model showed the best performance with an accuracy of 72% and a recall of 93% for predicting the risk of CHD.
Applying SMOTE improved the recall for minority classes across various models.
