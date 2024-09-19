import os
from typing import List, Optional

import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from model import generate_multiple_recipes, get_user_preferences
from pydantic import BaseModel, Field, conlist

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# class params(BaseModel):
#     n_neighbors:int=5
#     return_distance:bool=False


 
class NutritionalPreferences(BaseModel):
    calories: int = Field(..., ge=0, le=2000)
    fat: int = Field(..., ge=0, le=100)
    sugar: int = Field(..., ge=0, le=100)
    sodium: int = Field(..., ge=0, le=2000)
    fiber: int = Field(..., ge=0, le=50)
    carbs: int = Field(..., ge=0, le=300)
    protein: int = Field(..., ge=0, le=200)

class UserPreferences(BaseModel):
    dish_type: str = Field(..., description="Type of dish (Appetizer, Main Course, Dessert, Snack)")
    cuisine: str = Field(..., description="Type of cuisine (e.g., Italian, Indian, Mexican, Chinese)")
    flavor_profile: str = Field(..., description="Flavor profile (Spicy, Sweet, Savory, Tangy)")
    diet_restrictions: List[str] = Field(default_factory=list, description="Dietary restrictions")
    nutritional_preferences: NutritionalPreferences
    ingredients: List[str] = Field(..., description="List of preferred ingredients")
    exclude: Optional[List[str]] = Field(None, description="List of ingredients to exclude")
    num_recipes: int = Field(5, description="Number of recipes to generate")
    
class PredictionIn(BaseModel):
    
    userpreferences: UserPreferences

class Recipe(BaseModel):
    Name:str
    CookTime:str
    PrepTime:str
    TotalTime:str
    RecipeIngredientParts:list[str]
    Calories:float
    FatContent:float
    SaturatedFatContent:float
    CholesterolContent:float
    SodiumContent:float
    CarbohydrateContent:float
    FiberContent:float
    SugarContent:float
    ProteinContent:float
    RecipeInstructions:list[str]

class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None


@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict/" , response_model=PredictionOut)
def update_item(prediction_input:PredictionIn):
   #recommendation_dataframe=recommend(dataset,prediction_input.nutrition_input,prediction_input.ingredients,prediction_input.params.dict())
    output=generate_multiple_recipes(prediction_input.userpreferences)
    output=output.to_dict()
    return {"output":output}
