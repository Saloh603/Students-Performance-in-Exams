import joblib
import pandas as pd
from sklearn.linear_model import SGDRegressor
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Load the model
model = joblib.load('model/model_0.0.0.joblib')

# Predict
def predict(data):
    return model.predict(data)


class Item(BaseModel):
    ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    gender: str


app = FastAPI()

# Load the encoders
loaded_encoders = joblib.load('model/label_encoders.joblib')
columns_to_encode = ['ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'gender']


def category_converter(item: Item):
    new_data = item.dict()
    new_df = pd.DataFrame([new_data])

    for column in columns_to_encode:
        le = loaded_encoders[column]
        print(f"Column: {column}, Categories: {le.classes_}")
        new_df[column] = le.transform(new_df[column])
        print(f"Transformed {column}: {new_df[column]}")

    return new_df

@app.post("/items/")
async def create_item(item: Item):
    item = category_converter(item)
    item = list(item.values)
    print(item)
    prediction = predict(item)
    return {"overall_score_prediction": prediction.tolist()}  # Convert numpy array to list

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
