# Mushroom_Classification

# Introduction
Identifying edible mushrooms can be challenging for many people upon encountering them. 
My objective is to develop a machine learning model capable of classifying different types of fungi. 
Additionally, I aim to create a user-friendly platform where individuals can upload photos of mushrooms to receive an assessment of their edibility.

## Data
Mushroom world: 173 species with 10 images each

https://www.mushroom.world/mushrooms/namelist

Kaggle mushroom data: 215 species with 4 images each at least

https://www.kaggle.com/datasets/daniilonishchenko/mushrooms-images-classification-215/data

## Web scrape data
mushroom_script.ipynb

mushroom_uk_script.ipynb

These two notebooks allow us to scrape the image link from mushroom world webcite and uk wild mushroom webcite.

mushroom_images.csv

mushroom_info.csv

mushroom_uk_images.csv

Image csv file contains the image link and the name of the mushroom in the image. Info csv contains some information related to the mushroom, scraped from the mushroom world webcite.

download_image.ipynb

This notebook allows us to download the image from image path in the csv file to local folder.

### Data Preprocessing
data_augmentation.ipynb

data_aumentation.py

This notebook will filp the image to create more data for the model to train on. This is done because there are very little images I can find online with proper label.

### Vision Transformer model
my_dataset.py

predict.py

train.py

utils.py

vit_model.py

These python files contains vit models and their functions. If you have a image data folder with subfolders being each class like this, you can use train.py and set the input folder path to train the model. You will need to set the number of class.

## Results


## Potential Impact:


## Limitations:


# Future direction


# More information on Mushroom


# Reference
Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv preprint arXiv:2205.14135.

Gordić, A. (2022, July 18). ELI5: FlashAttention. Medium. https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad

Hugging Face. (2023). Flash Attention. Hugging Face Documentation. https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention

Tri Dao. (2023, January 1). FlashAttention - Tri Dao | Stanford MLSys #67. YouTube. https://www.youtube.com/watch?v=gMOAud7hZg4

