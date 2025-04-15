
# <div align="center">DataSocks 🧦</div>

<div align="center">
  <p>
    <a href="#overview">Overview</a> •
    <a href="#the-challenge">Challenge</a> •
    <a href="#project-components">Components</a> •
    <a href="#getting-started">Get Started</a> •
    <a href="#results">Results</a> •
    </p>
<a href="https://huggingface.co/lerobot">
        <img src="https://img.shields.io/badge/Robotics-SO--100-blue" alt="SO-100">
</a>
<a href="https://phospho.ai">
        <img src="https://img.shields.io/badge/Framework-Phospho-green" alt="Phospho">
</a>
<a href="https://lu.ma/roboticshack?tk=kB537f">
    <img src="https://img.shields.io/badge/Hackathon-Mistral%20AI-purple" alt="Hackathon Mistral">
</a>
</div>

---

## Resources

### Dataset
The augmented dataset used in this project is publicly available on Hugging Face. It includes all the augmented videos and metadata generated during the hackathon. You can access it here:  
[![Dataset on Hugging Face](https://img.shields.io/badge/Dataset-Huggingface-blue)](https://huggingface.co/datasets/llinguini/augmented_datasocks_team_20)

### Model
The trained model, fine-tuned version of ACT model, is also available on Hugging Face, access the model here:  
[![Model on Hugging Face](https://img.shields.io/badge/Model-Huggingface-green)](https://huggingface.co/pravsels/augmented_datasocks_24k_steps)

## Overview

DataSocks is a robotics project developed during the Mistral AI Hackathon, focusing on improving robot performance across varying environmental conditions. The project addresses one of the most significant challenges in modern robotics: environmental sensitivity during training and inference.

Most robotic systems require consistent lighting and environmental conditions between training and deployment phases. When these conditions change, model performance degrades significantly. DataSocks demonstrates how data augmentation techniques can be used to create more robust robotic models that perform well across different environmental conditions.

<div align="center">
    <img src="./examples_augmented_data/roboengine_augmented_video_2.gif" alt="DataSocks Demo" width="600">
</div>

<div align="center">
    <em>Demo GIF showing the robot picking up socks in different environmental conditions</em>
</div>

## The Challenge

Robots trained in specific conditions often fail when:
- Lighting conditions change
- Backgrounds vary
- Shadows or reflections appear differently

Our solution focuses on a simple but representative task: **picking up socks and placing them in a container** using the SO-100 robotic arm and Phospho framework.

## Project Components

### Data Collection
- Used Phospho framework to collect original training data

### Data Augmentation
The core innovation of this project is the extensive data augmentation pipeline:

1. **Simple Image-Based Augmentations** (`simple_augmentations.py`)
   - Uses Kornia for color jittering, contrast adjustments, and perspective transformations
   - Applies consistent transformations across entire video sequences

2. **Advanced Segmentation-Based Augmentation** (`roboengine_script.py`, `roboengine_from_fixed_mask.py`)
   - Segments robot arm and target objects
   - Applies background replacements while maintaining foreground elements
   - Handles edge cases with mask fixing techniques

3. **Dataset Integration** (`insert_augmented_files_in_dataset.py`)
   - Seamlessly integrates augmented videos into the training dataset
   - Maintains proper parquet file structure for Phospho and Huggingface compatibility

### Demo Application
The demo system includes:
- Speech recognition using Whisper ([`demo/whisper.py`](demo/whisper.py))
- Text-to-speech using Kokoro ([`demo/main.py`](demo/main.py))
- Natural language conversation with Mistral Small
- Robot control via Phospho API ([`demo/client.py`](demo/client.py))  
    - Check the [Phospho documentation](https://docs.phospho.ai/learn/ai-models#train-an-act-model-locally-with-lerobot) to see how to train and load an ACT model!

## Getting Started

### Prerequisites
```bash
torch>=1.8.0
kornia>=0.6.0
opencv-python
numpy
tqdm
Pillow
diffusers 
transformers
```

### Installation
```bash
git clone https://github.com/yourusername/datasocks.git
cd datasocks
pip install -r requirements.txt
```

### Running Data Augmentation
```bash
python data_augmentation/simple_augmentations.py --runs_per_vid 5 --batch_size 16
```

For RoboEngine-based segmentation augmentation:
```bash
python data_augmentation/roboengine_script.py
```

### Running the Demo

The demo cannot actually be run without replicating the exact environment setup used during the hackathon. This includes specific hardware configurations, dependencies, and access to the SO-100 robotic arm and Phospho framework. For more details, please contact the contrbutors.

## Project Structure
```
README.md
requirements.txt
data_augmentation/           # Data augmentation scripts
  ├── simple_augmentations.py    # Kornia-based image transformations  
  ├── roboengine_script.py       # Segmentation-based augmentation
  ├── roboengine_from_fixed_mask.py
  ├── stitch_video.py
  └── insert_augmented_files_in_dataset.py
demo/                       # Demo application
  ├── whisper.py            # Speech recognition
  ├── main.py               # Demo orchestrator
  ├── client.py             # Phospho API client
  └── server.py             # ACT policy server
examples_augmented_data/    # Example outputs from augmentation
examples_original_video/    # Original training data samples
```

## Results

By augmenting a small original dataset with environment variations, we were able to create a model that successfully performs the sock retrieval task across multiple lighting conditions and backgrounds, by the lack of time the training cannot be done properly.


## Acknowledgments

This project was developed during the [Mistral AI Hackathon](https://mistral.ai) using the SO-100 robotic arm and [Phospho framework](https://phospho.ai) (Use it !) for data collection, training, and inference.


## Contributors

<div align="center">
  <a href="https://github.com/ramosleandre">
    <img src="https://github.com/ramosleandre.png" width="50px" alt="Your Name" style="border-radius:50%">
  </a>
  <!-- Add other contributors here -->
</div>