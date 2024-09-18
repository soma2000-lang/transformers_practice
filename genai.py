import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

def get_user_preferences():
    preferences = {}
    preferences['meat_type'] = input("What type of meat would you prefer? (chicken/mutton/fish/pork/beef/seafood): ").lower()
    preferences['cooking_style'] = input("What cooking style do you prefer? (grilled/baked/fried/curry): ").lower()
    preferences['spice_level'] = input("What spice level do you prefer? (mild/medium/hot): ").lower()
    preferences['cooking_time'] = int(input("Maximum cooking time in minutes: "))
    return preferences

# Initialize the language model
llm = OpenAI(temperature=0.7)

# Create the LLMChain
recipe_chain = LLMChain(llm=llm, prompt=RECIPE_PROMPT)

# Set up the output parser
response_schemas = [
    ResponseSchema(name="recipe_name", description="The name of the generated recipe"),
    ResponseSchema(name="ingredients", description="List of ingredients with quantities"),
    ResponseSchema(name="instructions", description="Step-by-step cooking instructions"),
    ResponseSchema(name="cooking_time", description="Estimated cooking time"),
    ResponseSchema(name="serving_size", description="Number of servings"),
    ResponseSchema(name="description", description="Brief description of the recipe")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

def main():
    print("Welcome to the AI-Driven Non-Vegetarian Recipe Generator!")
    print("Let's create a unique recipe based on your preferences.")
    
    preferences = get_user_preferences()
    recipe = generate_recipe(preferences)
    
    print("\nHere's your personalized recipe:\n")
    if isinstance(recipe, dict):
        for key, value in recipe.items():
            print(f"{key.replace('_', ' ').title()}:")
            print(value)
            print()
    else:
        print(recipe)
        
def generate_recipe(preferences):
    meat_type = preferences['meat_type']
    product = random.choice(LICIOUS_PRODUCTS[meat_type])
    
    recipe_input = {
        "meat_type": meat_type,
        "product": product,
        "cooking_style": preferences['cooking_style'],
        "spice_level": preferences['spice_level'],
        "cooking_time": preferences['cooking_time']
    }
    
    recipe_output = recipe_chain.run(recipe_input)
    
    try:
        parsed_recipe = output_parser.parse(recipe_output)
        return parsed_recipe
    except Exception as e:
        print(f"Error parsing recipe: {e}")
        return recipe_output
if __name__ == "__main__":
    main()
    



