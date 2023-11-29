# Mushroom_Classification

## Introduction
- Identifying edible mushrooms can be challenging for many people upon encountering them.

- My objective is to develop a machine learning model capable of classifying different types of fungi.

- Additionally, I aim to create a user-friendly platform where individuals can upload photos of mushrooms to receive an assessment of their edibility.

## Method

### Vision Transformer

![1_-DBSfgxHUuknIqmyDVKwCg](https://github.com/Racso777/Mushroom_Classification/assets/111296013/9919daa6-e35e-46ac-94af-1f86db247fe0)

### Why VIT?

- Global Context Awareness: Unlike CNNs that primarily focus on local features due to their convolutional nature, Vision Transformers capture global dependencies in the image. Each transformer layer processes the entire image as a whole, enabling it to understand more complex, high-level relationships in the data.

- Scalability and Flexibility: Vision Transformers can easily scale with the amount of available data and computational resources. They tend to perform better as the dataset size increases, making them highly effective for large-scale applications.

- Transfer Learning and Generalization: ViTs demonstrate excellent transfer learning capabilities. A model trained on one large dataset can be fine-tuned on a smaller dataset for a different task, often outperforming models trained specifically for that task. This is partly due to their ability to generalize well from large-scale training.

-  Efficient Parallelization: Transformers are more amenable to parallel processing compared to CNNs. Unlike the sequential nature of RNNs or the local processing of CNNs, the self-attention mechanism in transformers can process all parts of the input simultaneously, leading to efficiency gains during training.

## Data
- Mushroom world: 173 species with 10 images each

  https://www.mushroom.world/mushrooms/namelist

- Kaggle mushroom data: 215 species with 4 images each at least

  https://www.kaggle.com/datasets/daniilonishchenko/mushrooms-images-classification-215/data

- huggingface dataset

  https://huggingface.co/datasets/Racso777/Mushroom_Dataset

## Web scrape data
- mushroom_script.ipynb

- mushroom_uk_script.ipynb

These two notebooks allow us to scrape the image link from mushroom world webcite and uk wild mushroom webcite.

![图片_20231129013247](https://github.com/Racso777/Mushroom_Classification/assets/111296013/610ae20c-ef48-4064-b45b-1a173dd4280e)

- mushroom_images.csv

- mushroom_info.csv

- mushroom_uk_images.csv

Image csv file contains the image link and the name of the mushroom in the image. Info csv contains some information related to the mushroom, scraped from the mushroom world webcite.

- download_image.ipynb

This notebook allows us to download the image from image path in the csv file to local folder.

## Data Preprocessing
- data_augmentation.ipynb

- data_aumentation.py

This notebook implements image flipping as a technique to augment the dataset for model training, addressing the issue of limited availability of appropriately labeled images found online.

![WeChat截图_20231129013323](https://github.com/Racso777/Mushroom_Classification/assets/111296013/ed82b4ef-6d05-47e4-b8b1-21a4a61c27c5)

## Vision Transformer model
- my_dataset.py

- predict.py

- train.py

- utils.py

- vit_model.py

The Python files provided include ViT (Vision Transformer) models and their associated functions. If your image data is organized into subfolders, each representing a class, you can utilize train.py to train the model. Simply set the input folder path in train.py and specify the number of classes.

![WeChat截图_20231129014823](https://github.com/Racso777/Mushroom_Classification/assets/111296013/6bfd97c0-efc6-4b89-b419-d0f711b45caa)

Train.py will generate the excel file for accuracy and loss and the model weights will save as best_model.path, which will be used by predict.py to predict mushroom image.

## Results

Vision Transformer with pretrained weight:

![WeChat截图_20231129014037](https://github.com/Racso777/Mushroom_Classification/assets/111296013/d1185443-2215-4ddd-a9d6-1c062e69a9ab)

![WeChat截图_20231129013938](https://github.com/Racso777/Mushroom_Classification/assets/111296013/5d55ad69-8231-4656-acfe-cfc46544fbbc)


## Code Demonstration:
Huggingface space:

https://huggingface.co/spaces/Racso777/Mushroom_Classification

Predict.py can also be used to predict image. Weights can be found in the link above.

## Limitations and Critical Analysis

### Finished:

- Implemented a Vision Transformer model for classifying mushroom images, achieving a training accuracy of 98%. 

- The correct class is typically among the top three predictions. 

- Developed an interface using HuggingFace for uploading images and receiving prediction outcomes.

### Problems:

- The model cannot predict exactly the class that mushroom belongs to.

- Only contains 370 classes of mushrooms, more classes will take longer to train and more resouces to collect and store.

## Future direction

- Finding more data with proper label

- Tranditional CNN model such as Resnet, Efficientnet, Alexnet.

## More information on Mushroom
mushroom world: https://www.mushroom.world/home/index

wild uk mushroom: https://www.wildfooduk.com/mushroom-guide/

ultimate mushroom guide:https://ultimate-mushroom.com/

## Reference
- Leonard2021. (2022, May 3). Vision Transformer实现图像分类+可视化+训练数据保存. CSDN. Retrieved from https://blog.csdn.net/weixin_51331359/article/details/124514770
- Raw, N. (2022, February 11). Fine-Tune ViT for Image Classification with Transformers. Hugging Face. Retrieved from https://huggingface.co/blog/fine-tune-vit.
- Han, K., Xiao, A., Wu, E., Guo, J., Xu, C., & Wang, Y. (2022). Recent Advances in Vision Transformer: A Survey and Outlook of Recent Work. arXiv preprint arXiv:2203.01536. Retrieved from https://arxiv.org/abs/2203.01536​​​​.
- Khan, S., Naseer, M., Hayat, M., Zamir, S. W., Khan, F. S., & Shah, M. (2021). Transformers in Vision: A Survey. arXiv preprint arXiv:2101.01169. Retrieved from https://arxiv.org/abs/2101.01169​​​​.
- Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). A survey on efficient vision transformers: algorithms, techniques, and performance benchmarking. arXiv preprint arXiv:2309.02031. Retrieved from https://arxiv.org/abs/2309.02031​​​​.

