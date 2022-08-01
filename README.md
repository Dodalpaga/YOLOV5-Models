# Table of Contents

- [Table of Contents](#table-of-contents)
- [Project layout](#project-layout)
- [Create (and activate) a Virtual Environment](#create-and-activate-a-virtual-environment)
- [Requirements](#requirements)
- [Launch the server locally](#launch-the-server-locally)
- [Launch the server on heroku](#launch-the-server-on-heroku)
- [How it works ...](#how-it-works-)
  - [Inference Setion](#inference-setion)
  - [Webcam Section](#webcam-section)

# Project layout

```
📦YOLO-Object-Detection-Template
┣ 📦components
┃ ┣ 📂Test
┃ ┃ ┗ 📂Images_predites
┃ ┃   ┗ 📜.gitkeep
┃ ┗ 📜weights.pt
┣ 📦static
┃ ┣ 📂css
┃ ┃ ┣ 📜colorful.css
┃ ┃ ┣ 📜index.css
┃ ┃ ┣ 📜interface.css
┃ ┃ ┗ 📜notebook.css
┃ ┗ 📂images
┃ ┣ 📜logo.png
┃ ┗ 📜tensorboard.png
┣ 📦templates
┃ ┣ 📜doc.html
┃ ┣ 📜index.html
┃ ┣ 📜interface.html
┃ ┣ 📜live_streaming.html
┃ ┣ 📜training.html
┃ ┗ 📜training_notebook.html
┣ 🐍app.py
┣ 📜README.md
┗ 📜requirements.txt
```

# Create (and activate) a Virtual Environment

In order to create a new venv, type the following in a terminal :
`python3 -m venv /path/to/new/virtual/environment/`

Then, activate it so you can install the dependencies :
`source /path/to/new/virtual/environment/bin/activate`

# Requirements

Once the venv is activated, install the python dependencies
`pip install -r requirements.txt`

# Launch the server locally

To launch the flask server, type in a terminal :
`python app.py`

# Launch the server on heroku

To launch the flask server, type in a terminal :
`git push heroku main`
`heroku open`

# How it works ...

## Inference Setion

1. Import a yolo model (from a .pt file)
2. Import test images
3. Import test labels
4. Tune Confidence Threshold and IoU Threshold for Non-Maximum Suppression
5. Runs Inferences
6. (Optional) Clear cache when importing new images/labels

## Webcam Section

1. Import a yolo model (from a .pt file) in the [Inference Setion](#inference-setion)
2. Click on the "Webcam" tab and get live detection (~10/15 fps) using your own webcam
