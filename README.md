# PlateIQ - Food Recognition System

PlateIQ identifies food from photographs. It recognises all 101 dishes in the Food-101 dataset (pizza, sushi, samosa, tiramisu and many more) and includes a web app for trying it out.

## Results

Measured on the full Food-101 test set (25,250 images the model never saw during training):

| Metric | Score |
|---|---|
| Top-1 accuracy | 68.5% |
| Top-5 accuracy | 90.3% |

Top-5 accuracy means the correct food appears among the model's five best guesses.

## How it works

- Backbone: EfficientNetB0, pretrained on ImageNet
- Training in two phases: first the new classification layers, then fine-tuning of the top 150 layers of the backbone
- Data augmentation (flip, rotation, zoom, shift, contrast) and label smoothing
- Test-time averaging over each image and its mirror image
- Streamlit web app with top-5 predictions, a searchable food list and a performance dashboard

## Project structure

- plateiq.py: data preparation, training, evaluation, prediction and the web app (single file)
- results/: evaluation charts and metrics
- requirements.txt: Python dependencies

## Setup

```
pip install -r requirements.txt
```

## Usage

```
python plateiq.py --prepare-data
python plateiq.py --train
python plateiq.py --evaluate
streamlit run plateiq.py
```

Other commands:

```
python plateiq.py --train --resume
python plateiq.py --predict path/to/image.jpg
python plateiq.py --prepare-data --subset-size 300
```

- `--resume` continues training from the best saved model.
- `--predict` classifies a single image from the command line.
- `--subset-size` trains on fewer images per food for a quicker run.

The first run downloads the Food-101 dataset (about 5 GB). Training on a CPU takes several hours, and a GPU is much faster.

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/): 101 food categories, 101,000 images (750 training and 250 test images per category).

## Limitations

- The model only knows the 101 Food-101 categories. Other foods will be matched to the closest one, and the app shows a low-confidence warning when it is unsure.
- Visually similar dishes (for example steak and filet mignon) are sometimes confused.
