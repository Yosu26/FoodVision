# 🍔 FoodVision: Classifying 101 Foods with ViT
A computer vision project that classifies food images into 101 categories using a Vision Transformer (ViT). Trained on the Food-101 dataset, this model is deployed with a Gradio app for easy testing and sharing.

<br>

# 📷 Demo
Try the Gradio demo locally:
```bash
cd demos/foodvision_big
python app.py
```

Live demo: https://huggingface.co/spaces/Yosu26/foodvision_demo
<br>

# 🧠 Model Architecture
- Base model: `ViT-B/16` from `torchvision`
- Trained on: Food-101 dataset
- Features:

  - Predict food images across __101 different classes__
  - __High-accuracy predictions__ with a pretrained ViT model
  - Interactive UI with __Gradio__ for real-time demos

<br>

# 📁 Project Structure

```bash
FoodVision 
├── demos/ 
│   └── foodvision_big/ 
│       ├── examples/ # Sample images for Gradio demo
│       ├── app.py # Gradio app
│       ├── class_names.txt # 101 food class names
│       ├── model.py # ViT model for inference
│       └── requirements.txt # Python dependencies 
├── .gitattributes # Git settings (e.g., line endings, filters)
├── .gitignore # Files and folders Git should ignore
├── README.md # This file.gitignore # Files and folders Git should ignore
├── data_setup.py # DataLoader setup
├── engine.py # Training loop logic
├── get_data.py # Downloads Food-101 and Dataset setup
├── model_builder.py # Contains model creation logic
├── requirements.txt # Python dependencies
├── train.py # Training loop logic
└── utils.py # Helper functions
```
<br>

# 🚀 Getting Started

1. Clone the repo

    ```bash
    git clone https://github.com/Yosu26/FoodVision.git
    cd FoodVision
    ```
2. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Gradio app
    
    ```bash
    cd demos/foodvision_big
    python app.py
    ```
<br>

# 📊 Training

To train your own model, you can run the training script `train.py` with the following arguments:

```bash
python train.py --num_epochs 10 --batch_size 32 --learning_rate 0.001
```
Where:
- `--num_epochs` specifies the number of training epochs.
- `--batch_size` specifies the size of each training batch.
- `--learning_rate` specifies the learning rate for the optimizer.

You can also modify these values inside the train.py script directly if you prefer not to use the command line.
