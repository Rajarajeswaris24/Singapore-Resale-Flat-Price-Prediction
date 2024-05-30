DEMO VIDEO:https://drive.google.com/file/d/1ICjmXQou1GwRN9LY0emF3WcY7G8QM6pL/view?usp=drive_link

Problem Statement:

The resale flat market in Singapore is highly competitive, and it can be challenging to accurately estimate the resale value of a flat. There are many factors that can affect resale prices, such as location, flat type, floor area, and lease duration. A predictive model can help to overcome these challenges by providing users with an estimated resale price based on these factors.

The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.

Data Collection :

Collect a dataset of resale flat transactions of the Singapore Housing and Development Board (HDB) for the years 1990 to Till Date from https://beta.data.gov.sg/collections/189/view  concatenate all 5 datasets to a single dataset.

Preprocessing:

1.Remove duplicates.

2.Handle missing values with mean/median/mode or dropna or drop the column having null values.

3.convert the wrong datatype.

4.Treat Outliers using IQR and identify skewness if needed remove skewness.

5.Encode categorical variables using suitable techniques, such as one-hot encoding, label encoding, or ordinal encoding, based on their nature and relationship with the target variable.

6.EDA: Try visualizing outliers and skewness(before and after treating skewness) using Seaborn’s boxplot, distplot, violinplot.visualize heatmap for correlation and pairplot for relationships between columns.

Feature Engineering: 

Extract relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date. Create any additional features that may enhance prediction accuracy.

Model Selection and Training:

1.Split the dataset into training and testing/validation sets.

2.Choose an appropriate machine learning model for regression (e.g., linear regression, decision trees, or random forests). Train the model on the historical data, using a portion of the dataset for training.

3.Optimize model hyperparameters using techniques such as cross-validation to find the best-performing model.

Model Evaluation: 

Evaluate the model's predictive performance using regression metrics such R2 Score.

Streamlit Web Application:

Develop a user-friendly web application using Streamlit that allows users to input details of a flat (town, flat type, storey range, etc.). Utilize the trained machine learning model to predict the resale price based on user inputs.

Deployment on Render:

Deploy the Streamlit application on the Render platform to make it accessible to users over the internet.

Testing and Validation: 

Thoroughly test the deployed application to ensure it functions correctly and provides accurate predictions.

Regression R-squared score :

LinearRegression :  0.7801528825790175

DcisionTreeRegressor : 0.9721506325933871

RandomForestRegressor :  0.984231562972507

XGBRegressor :  0.9798490335911281

So the the best model is random forest 98% r2 score but I took xgboost model the second best model because random forest model is about 7.15 gb it takes more memory and time to load the model in streamlit so I choose xgboost.
