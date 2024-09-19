import os
from typing import List, Optional

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class UserPreferences(BaseModel):
    cuisine: str = Field(..., description="Type of cuisine (e.g., Italian, Mexican, Japanese)")
    diet: Optional[str] = Field(None, description="Dietary restrictions (e.g., vegetarian, vegan, gluten-free)")
    ingredients: List[str] = Field(..., description="List of preferred ingredients")
    exclude: Optional[List[str]] = Field(None, description="List of ingredients to exclude")

class Recipe(BaseModel):
    name: str
    ingredients: List[str]
    instructions: List[str]
    cooking_time: str
    servings: int

def get_user_preferences() -> UserPreferences:
    cuisine = input("Enter the type of cuisine: ")
    diet = input("Enter any dietary restrictions (press Enter if none): ")
    ingredients = input("Enter preferred ingredients (comma-separated): ").split(",")
    exclude = input("Enter ingredients to exclude (comma-separated, press Enter if none): ").split(",")
    
    return UserPreferences(
        cuisine=cuisine,
        diet=diet if diet else None,
        ingredients=[i.strip() for i in ingredients],
        exclude=[e.strip() for e in exclude] if exclude != [''] else None
    )

def generate_recipe(preferences: UserPreferences) -> Recipe:
    functions = [
        {
            "name": "create_recipe",
            "description": "Generate a recipe based on user preferences",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the recipe"},
                    "ingredients": {"type": "array", "items": {"type": "string"}, "description": "List of ingredients"},
                    "instructions": {"type": "array", "items": {"type": "string"}, "description": "List of cooking instructions"},
                    "cooking_time": {"type": "string", "description": "Total cooking time"},
                    "servings": {"type": "integer", "description": "Number of servings"}
                },
                "required": ["name", "ingredients", "instructions", "cooking_time", "servings"]
            }
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates recipes."},
            {"role": "user", "content": f"Generate a recipe based on these preferences: {preferences.json()}"}
        ],
        functions=functions,
        function_call={"name": "create_recipe"}
    )

    recipe_data = response.choices[0].function_call.arguments
    return Recipe.parse_raw(recipe_data)

def generate_image(recipe_name: str) -> str:
    response = openai.Image.create(
        prompt=f"A delicious plate of {recipe_name}",
        n=1,
        size="512x512"
    )
    return response['data'][0]['url']

def main():
    print("Welcome to the Recipe Generator!")
    preferences = get_user_preferences()
    
    print("\nGenerating recipe...")
    recipe = generate_recipe(preferences)
    
    print("\nGenerating image...")
    image_url = generate_image(recipe.name)
    
    print("\nHere's your generated recipe:")
    print(f"Name: {recipe.name}")
    print(f"Ingredients: {', '.join(recipe.ingredients)}")
    print("Instructions:")
    for i, instruction in enumerate(recipe.instructions, 1):
        print(f"{i}. {instruction}")
    print(f"Cooking Time: {recipe.cooking_time}")
    print(f"Servings: {recipe.servings}")
    print(f"\nImage URL: {image_url}")

if __name__ == "__main__":
    main()
    main()
