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
### - Gender Distribution
Count Plot: Shows the distribution of patients by gender.
There are more than 1,750 female patients and approximately 1,500 male patients.
### - CHD Risk Distribution
Count Plot: Displays the distribution of patients with and without CHD risk.
Approximately 2,800 patients have no risk of CHD, while around 500 patients are at risk.
### - CHD Risk by Gender
Count Plot: Shows the distribution of CHD risk among males and females.
Approximately 250 females and 300 males are at risk of CHD.
### - Smoking Status and CHD Risk
Bar Plot: Illustrates the CHD risk based on smoking status.
Patients who smoke have a higher risk of CHD compared to non-smokers.
### - Outliers in Data
Box Plot: Identifies outliers in the dataset.
Values above 100 and below 50 in certain attributes are acting as outliers.
### - Age and CHD Risk
Box Plot: Examines the distribution of CHD risk across different age groups.
Majority of patients with CHD fall within the age range of 50 to 60 years.
Some patients with CHD are older, ranging from 60 to 70 years old.
### - Stroke History and CHD Risk
Scatter Plot: Observes the prevalence of stroke across different age groups.
Patients with a history of stroke and age above 50 years have a higher risk of CHD.
### - Blood Pressure and Hypertension
Scatter Plot: Displays the relationship between systolic blood pressure and hypertension.
There is a strong positive correlation between high systolic blood pressure and hypertension.
The scatter plot shows a separation between individuals with and without hypertension based on their systolic blood pressure readings.
### - Correlation Heatmap
Heatmap: Shows the correlation matrix of the dataset.
Identifies the strength of relationships between different attributes.
Helps in understanding which features are strongly correlated with the target variable (CHD risk).
### Key Findings from EDA
- Gender and CHD Risk: Gender does not show a significant imbalance in CHD risk.
- Smoking: Smoking is a significant risk factor for CHD.
- Age: Older age groups (50-70 years) have a higher prevalence of CHD.
- Hypertension and Blood Pressure: High blood pressure is strongly correlated with hypertension, which in turn is a risk factor for CHD.
- Outliers: Certain attributes have outliers that need to be handled during preprocessing.
