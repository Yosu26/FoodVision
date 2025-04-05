
import os
import requests
import zipfile 

from pathlib import Path

data_path = Path("data")
image_path = data_path/"pizza_steak_sushi"

if image_path.is_dir():
  print(f"Directory {image_path} exists.")
else:
  print(f"Directory {image_path} does not exist, creating one...")
  image_path.mkdir(parents=True, exist_ok=True)

with open(data_path/"pizza_steak_sushi.zip","wb") as f:
  request = requests.get("https://github.com/Yosu26/FoodVision/raw/refs/heads/main/pizza_steak_sushi.zip")
  print(f"Downloading pizza, steak, sushi data...")
  f.write(request.content)

with zipfile.ZipFile(data_path/"pizza_steak_sushi.zip","r") as zip_ref:
  print(f"Unzipping pizza, steak, sushi data...")
  zip_ref.extractall(image_path)

os.remove(data_path/"pizza_steak_sushi.zip")
