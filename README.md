# Project harddrive life span fortcasting 

## Drive Lifespan Forecasting

### Overview
This project focuses on forecasting the lifespan of drives using machine learning techniques. It involves data analysis, visualization, trend review, correlation analysis, and the development of machine learning models for prediction.

### Business Understanding

The primary goal is to forecast the lifespan of drives based on various parameters, such as capacity, power on hours, total read bytes processed, etc.

### Data Understanding

The data is loaded from an Excel file (`SSA2 Enclosure Disks Endurance.xlsx`) and is explored using Pandas and Matplotlib. Initial data exploration includes examining the structure, unique values, statistics, and data types of different columns.

### Visualize Data

The data is visualized using violin plots and line plots to understand the distribution and trends of key parameters across different features.

### Review Trends

Line plots are used to review trends in parameters such as total read bytes processed, power on hours, MAX Spare Count, used spare count in core, and worst wear leveling count over time.

### Correlation

Correlation analysis is performed to identify relationships between different parameters. A new ID column is created, and the age of each drive is calculated based on the 'created' date.

### Data Preparation

Data preparation involves handling data types, creating dummy variables, and preparing the dataset for model training.

### Modeling

Five machine learning models (Random Forest, Gradient Boosting, Ridge, Lasso, ElasticNet) are trained using GridSearchCV to find the best hyperparameters. The models are evaluated using R2 score and Mean Absolute Error.

### Deployment

A simple deployment example is provided using a tkinter GUI, where users can input drive characteristics, and the model predicts the worst wear leveling count. The GUI also displays the predicted status (Red, Yellow, Green) based on outlier thresholds.

### Outlier Detection

Outliers are detected for key parameters such as total read bytes processed, power on hours, and worst wear leveling count, and a status (Red, Yellow, Green) is assigned based on the number of outliers.

### Instructions for Use

1. Install required libraries: `pip install pandas matplotlib seaborn scikit-learn==0.22 tk`
2. Execute the provided Python scripts (`project_analysis.py` and `prediction_script.py`).
3. Follow the GUI instructions for making predictions.

Feel free to explore and modify the code to suit your specific needs.
