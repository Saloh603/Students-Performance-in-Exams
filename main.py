import joblib
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
    gender: str
    ethnicity: str
    education: str
    lunch: str
    course: str


app = FastAPI()

async def category_converter(data):
    data = data.dict()
    """ # Sample data form
    {"gender": "male",
    "ethnicity": "group A",
    "education": "high school",
    "lunch": "standard",
    "course": "completed"
    }
    """

    gender = {'male': 1, 'female': 0}
    ethnicity = {'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4}
    lunch = {'standard': 1, 'free/reduced': 0}
    course = {'completed': 1, 'none': 0}
    education = {'some high school': 5,
                 'high school': 2,
                 'some college': 4,
                 "associate's degree": 0,
                 "bachelor's degree": 1,
                 "master's degree": 3}

    # Converting categories to numerical values
    data['gender'] = gender[data['gender']]
    data['ethnicity'] = ethnicity[data['ethnicity']]
    data['lunch'] = lunch[data['lunch']]
    data['course'] = course[data['course']]
    data['education'] = education[data['education']]

    return data


@app.post("/items/")
async def create_item(item: Item):
    item = await category_converter(item)
    item = [list(item.values())]
    prediction = predict(item)
    return {"overall_score_prediction": prediction.tolist()}  # Convert numpy array to list

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
