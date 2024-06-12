from typing import Any

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import time


# Load the dataset
def load_data(path):
    """ Load the dataset from the given path, rename the columns, calculate the mean score and filter the data
    :param path: Path to the dataset
    :columns race/ethnicity, parental level of education, lunch, test preparation course, gender"""
    dframe: DataFrame | Any = pd.read_csv(path)
    # Rename the columns
    dframe.rename(columns={'race/ethnicity': 'ethnicity', "parental level of education": "parental_level_of_education",
                           "test preparation course": "test_preparation_course"}, inplace=True)
    # Calculate mean score and filter the data
    dframe["mean score"] = ((dframe["math score"] + dframe["reading score"] + dframe["writing score"]) / 3).round()
    dframe.drop(columns=["math score", "reading score", "writing score"], inplace=True)
    dframe = dframe[dframe["mean score"] > 30]
    return dframe


# Encode categorical variables
def encode_categorical(df: DataFrame):
    """ Encode the categorical variables """
    columns_to_encode = ['ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'gender']
    encoders = {}

    for column in columns_to_encode:
        le = LabelEncoder()
        le.fit(df[column].astype(str))  # Ensure the data is in string format
        encoders[column] = le

    # Save the encoders for future use
    joblib.dump(encoders, f"label_encoders_{int(time.time())}.joblib")

    # Apply the encoders
    for column in columns_to_encode:
        le = encoders[column]
        df[column] = le.transform(df[column])

    return df



# Split the data into features and target variable
def split_data(df: DataFrame):
    """ Split the data into features and target variable """
    X = df.drop(['mean score'], axis=1)
    y = df['mean score']

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Define the preprocessing steps for numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', 'passthrough', categorical_features)
        ])

    return x_train, x_test, y_train, y_test, preprocessor

# Define the pipeline with preprocessing and the SGDRegressor model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', SGDRegressor())
])

# Train the model using the pipeline
pipeline.fit(x_train, y_train)

# Make predictions
predictions = pipeline.predict(x_test)

# Calculate the mean absolute difference
difference = abs(predictions - y_test)
mean_difference = difference.mean()

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Print the evaluation metrics
print(f"Mean Difference: {mean_difference}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")
