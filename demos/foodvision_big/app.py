### 1. Imports and class names setup ###
import gradio as gr 
import os 
import torch 

from model import create_vit_model
from timeit import default_timer as timer 
from typing import Tuple, Dict 

# Setup class names 
with open("class_names.txt", "r") as f: 
    class_names = [food_name.strip() for food_name in f.readlines()]

### 2. Model and transforms preparation ###

# Create model 
vit, vit_transforms = create_vit_model(num_classes=len(class_names))

# Load saved weights
vit.load_state_dict(
    torch.load(
        f="pretrained_vit_food101.pth",
        map_location=torch.device("cpu")
    )
)

### 3. Predict function ### 

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer 
    start_time = timer() 

    # Transform the target image and add a batch dimension
    img = vit_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode 
    vit.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(vit(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time 
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time 

### 4. Gradio app ### 

title = "FoodVision Big 🍔👁"
description = "An ViT feature extractor computer vision model to classify images of food into [101 different classes]"

# Create example list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)")
    ],
    examples=example_list,
    title=title,
    description=description,
)

# Launch the app!
demo.launch(share=True)
