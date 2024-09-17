import os

import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_recipe(ingredients):
    prompt = f"Create a unique recipe using these ingredients: {', '.join(ingredients)}. Include a title, ingredients list, and cooking instructions."
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a creative chef who creates unique recipes."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def generate_image(recipe_title):
    prompt = f"A high-quality, appetizing photo of {recipe_title}"
    
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )
    
    return response['data'][0]['url']

def main():
    print("Welcome to the AI Recipe Generator!")
    ingredients = input("Enter the raw meat ingredients (comma-separated): ").split(',')
    ingredients = [i.strip() for i in ingredients]
    
    print("\nGenerating recipe...")
    recipe = generate_recipe(ingredients)
    
    print("\nGenerated Recipe:")
    print(recipe)
    
    recipe_title = recipe.split('\n')[0]
    print(f"\nGenerating image for {recipe_title}...")
    image_url = generate_image(recipe_title)
    
    print(f"\nImage URL: {image_url}")
    print("\nYou can open this URL in a web browser to view the generated image.")

if __name__ == "__main__":
    main()    main()