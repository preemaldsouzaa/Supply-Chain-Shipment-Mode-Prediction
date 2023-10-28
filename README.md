# Shipment Mode Prediction Project

Welcome to the Shipment Mode Prediction project! In this project, we aim to predict the most suitable shipment mode for various shipments based on a comprehensive dataset. The project encompasses various stages, including data preprocessing, exploratory data analysis, and machine learning model development.
![image](https://github.com/preemaldsouzaa/Supply-Chain-Shipment-Mode-Prediction/assets/117831091/efd433e5-ea88-4586-b772-45a68ff282aa)

## Project Overview

- **Objective**: The goal of the project is to develop a robust and accurate machine learning model that can predict the most suitable shipment mode for a given set of variables. This will involve thorough data preprocessing and exploratory data analysis to ensure data quality, followed by the creation of predictive models.

- **Language:** Python
- **Main Libraries:** pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and more.

## Step 1: Data Preprocessing & EDA (Exploratory Data Analysis)

In this initial step, we perform data preprocessing and exploratory data analysis to better understand the dataset. This includes:

**Data Cleaning:**
- The dataset was loaded and thoroughly checked for inconsistencies, missing values, and duplicates.

**Data Visualization:**
- Data visualization techniques, including pair plots and count plots, were used to gain insights into the dataset.

**Handling Missing Values:**
- Missing values in key columns were handled by imputing the mean, mode, or zeros as appropriate.

**Changing Data Types of Columns:**
- Dates were converted to datetime format.
- Categorical columns were cast to categorical data types.
- Numerical columns were cast to integer data types.

**Feature Engineering:**
- Correlation analysis was conducted to select relevant features for the prediction model.
- A subset of columns was chosen for model development.

Let's explore some of the key findings from the dataset through data visualization:

### Shipment Mode Distribution

- The highest number of shipments (approximately 200 or more) were conducted by air, indicating that air transport is the most commonly used mode of shipping in the dataset.
- Truck shipments, while fewer in number compared to air, still represent a significant portion of the total shipments, with around 80 recorded in the dataset.
- Air charter shipments are still noteworthy with around 60 shipments, often used for specific logistics requirements.
- Ocean shipments have the fewest records, implying that this mode of transportation is the least commonly used among the options.

### Product Group Distribution

- A pie chart provides a quick visual overview of the distribution of product groups in the dataset.
- ARV (Antiretroviral) stands out as the largest slice of the pie, with 8,550 counts, representing over 82% of the dataset.
- HRDT is also prevalent, while other product groups have significantly fewer counts.

### Average Line Item Value by Country

- A graph displays the average 'Line Item Value' for the top 5 countries in the dataset.
- Namibia leads with the highest average 'Line Item Value' of approximately $643,807.83 USD, signifying its monetary significance.
- Nigeria follows closely with an average of about $446,731.94 USD.
- Pakistan, Tanzania, and South Africa also feature in the top 5 with varying average values, providing insights into transaction values across these countries.

### Line Item Value vs. Line Item Quantity

- A scatterplot reveals that most shipments consist of low-value, low-quantity items, but there is variability in the types of products and shipments.

### Managed By and Fulfill Via Analysis

- "PMO - US" dominates with a substantial count of 10,265 shipments, indicating a central role in managing shipments.
- "South Africa Field Office," "Ethiopia Field Office," and "Haiti Field Office" have significantly fewer shipments, suggesting specific regional operations with lower shipment volumes.

### Brand Analysis

- The "Generic" brand occurs 7,285 times in the dataset, indicating its popularity.
- Each brand has a unique market position:
    - Colloidal Gold has the highest total unit price, focusing on higher-priced products.
    - Generic caters to a larger customer base with lower-priced products.
    - Determines achieves a balance between unit price and count.

### Top Manufacturing Sites

- Aurobindo Unit III in India leads with 3,172 shipments, followed by Mylan (formerly Matrix) Nashik with 1,415 shipments, indicating the prominence of these manufacturing facilities.

## Step 2: Machine Learning Models

In the machine learning phase of the project, we explored a variety of classification models to predict the most suitable shipment mode based on the available features. The primary goal was to evaluate and fine-tune these models to achieve the best predictive performance.

### Model Evaluation (Without Hyperparameter Tuning)

Before hyperparameter tuning, we assessed the models using several key evaluation metrics:

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.3679 | 0.6471 | 0.3679 | 0.3487 |
| Support Vector Classifier | 0.8167 | 0.9124 | 0.8167 | 0.8424 |
| K-Nearest Neighbors | 0.9313 | 0.9534 | 0.9313 | 0.9376 |
| Decision Tree Classifier | 0.9367 | 0.9503 | 0.9367 | 0.9412 |
| Random Forest Classifier | 0.9474 | 0.9580 | 0.9474 | 0.9508 |
| Gradient Boosting Classifier | 0.9434 | 0.9604 | 0.9434 | 0.9483 |
| XGBoost Classifier | 0.9461 | 0.9607 | 0.9461 | 0.9504 |
| CatBoost Classifier | 0.9488 | 0.9619 | 0.9488 | 0.9526 |

### Model Evaluation (With Hyperparameter Tuning)

After careful hyperparameter tuning, we were able to enhance the performance of the models. Here are the updated evaluation results:

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| Cat Boost Classifier | 0.9488 | 0.9619 | 0.9488 | 0.9526 |
| Random Forest Classifier | 0.9501 | 0.9637 | 0.9501 | 0.9540 |
| XGBoost Classifier | 0.9474 | 0.9636 | 0.9474 | 0.9519 |
| Gradient Boosting Classifier | 0.9515 | 0.9640 | 0.9515 | 0.9551 |

These scores indicate the remarkable impact of hyperparameter tuning on model performance. The "Random Forest Classifier" emerged as the top-performing model, achieving an accuracy of 95% and a balanced combination of precision, recall, and F1-score.

Fine-tuning these models allowed us to create a more accurate and reliable prediction system for shipment modes. This level of model performance is invaluable for logistics and supply chain management, enabling efficient and cost-effective decision-making processes.

For in-depth details on the models and the hyperparameter tuning process, you can explore the code and documentation within the project repository on GitHub.
