import os
from typing import List, Optional
from pydantic import BaseModel, Field
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

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

class Recipe(BaseModel):
    name: str
    ingredients: List[str]
    instructions: List[str]
    cooking_time: str
    servings: int
    nutritional_info: dict

def get_user_preferences() -> UserPreferences:
    dish_type = input("Enter dish type (Appetizer, Main Course, Dessert, Snack): ")
    cuisine = input("Enter cuisine type (e.g., Italian, Indian, Mexican, Chinese): ")
    flavor_profile = input("Enter flavor profile (Spicy, Sweet, Savory, Tangy): ")
    
    print("Enter dietary restrictions (comma-separated, press Enter if none):")
    print("Options: no sugar, no sodium, less oil, gluten free, cholesterol free, lactose free, spice")
    diet_restrictions = input().split(',')
    diet_restrictions = [restriction.strip() for restriction in diet_restrictions if restriction.strip()]
    
    print("\nEnter nutritional preferences (enter a number for each):")
    calories = int(input("Calories (0-2000): "))
    fat = int(input("Fat (0-100g): "))
    sugar = int(input("Sugar (0-100g): "))
    sodium = int(input("Sodium (0-2000mg): "))
    fiber = int(input("Fiber (0-50g): "))
    carbs = int(input("Carbs (0-300g): "))
    protein = int(input("Protein (0-200g): "))
    
    nutritional_preferences = NutritionalPreferences(
        calories=calories, fat=fat, sugar=sugar, sodium=sodium,
        fiber=fiber, carbs=carbs, protein=protein
    )
    
    ingredients = input("Enter preferred ingredients (comma-separated): ").split(",")
    exclude = input("Enter ingredients to exclude (comma-separated, press Enter if none): ").split(",")
    num_recipes = int(input("Enter the number of recipes to generate (minimum 5): "))
    
    return UserPreferences(
        dish_type=dish_type,
        cuisine=cuisine,
        flavor_profile=flavor_profile,
        diet_restrictions=diet_restrictions,
        nutritional_preferences=nutritional_preferences,
        ingredients=[i.strip() for i in ingredients],
        exclude=[e.strip() for e in exclude] if exclude != [''] else None,
        num_recipes=max(5, num_recipes)
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
                    "servings": {"type": "integer", "description": "Number of servings"},
                    "nutritional_info": {
                        "type": "object",
                        "properties": {
                            "calories": {"type": "integer"},
                            "fat": {"type": "integer"},
                            "sugar": {"type": "integer"},
                            "sodium": {"type": "integer"},
                            "fiber": {"type": "integer"},
                            "carbs": {"type": "integer"},
                            "protein": {"type": "integer"}
                        }
                    }
                },
                "required": ["name", "ingredients", "instructions", "cooking_time", "servings", "nutritional_info"]
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

def generate_multiple_recipes(preferences: UserPreferences) -> List[Recipe]:
    recipes = []
    for i in range(preferences.num_recipes):
        print(f"\nGenerating recipe {i+1}/{preferences.num_recipes}...")
        recipes.append(generate_recipe(preferences))
    return recipes

def generate_image(recipe_name: str) -> str:
    response = openai.Image.create(
        prompt=f"A delicious plate of {recipe_name}",
        n=1,
        size="512x512"
    )
    return response['data'][0]['url']

def main():
    print("Welcome to the Advanced Recipe Generator!")
    preferences = get_user_preferences()
    
    recipes = generate_multiple_recipes(preferences)
    
    for i, recipe in enumerate(recipes, 1):
        print(f"\n--- Recipe {i} ---")
        print(f"Name: {recipe.name}")
        print(f"Ingredients: {', '.join(recipe.ingredients)}")
        print("Instructions:")
        for j, instruction in enumerate(recipe.instructions, 1):
            print(f"{j}. {instruction}")
        print(f"Cooking Time: {recipe.cooking_time}")
        print(f"Servings: {recipe.servings}")
        print("Nutritional Information:")
        for key, value in recipe.nutritional_info.items():
            print(f"  {key.capitalize()}: {value}")
        
        print("\nGenerating image...")
        image_url = generate_image(recipe.name)
        print(f"Image URL: {image_url}")

if __name__ == "__main__":
    main()
