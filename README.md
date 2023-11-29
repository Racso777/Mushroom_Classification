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

mushroom_images.csv
mushroom_info.csv
mushroom_uk_images.csv

download_image.ipynb

### Data Preprocessing
data_augmentation.ipynb
data_aumentation.py
image_size_distribution.ipynb

### Vision Transformer model
class_indices.json
my_dataset.py
predict.py
train.py
utils.py
vit_model.py

## Results


## Discussion Question: FlashAttention appears to be a versatile and valuable tool, especially when implemented on GPU-supported models, which encompasses a majority of current models. What potential drawbacks might be associated with this model?

## Limitations:
CUDA Compilation: We need a new CUDA kernel for each variant of attention, requiring low-level programming and extensive engineering, which may not be consistent across GPU architectures. A high-level language for writing attention algorithms, translatable to IO-aware CUDA implementations, is needed.

Multi-GPU IO-Aware Methods: While our attention implementation is nearly optimal for single-GPU use, extending and optimizing it for multi-GPU environments, including accounting for inter-GPU data transfers, represents an exciting area for future research.

Sparsity: In utilizing block-sparse FlashAttention for longer sequences, the sparsity increases, leading to more blocks and information being excluded in the training process as a compromise for enhanced performance. Although this method results in increased speed, the trade-offs resulting from the masked blocks remain uncertain. The authors have not explored or discussed the potential impacts of excluding blocks in this technique.

# Code Demonstration
If we want to train a model using this approach, we could clone the repo and run the python file: https://github.com/Dao-AILab/flash-attention/tree/main/training/run.py

The test dataset that we could use and train is in this file: https://github.com/Dao-AILab/flash-attention/blob/main/training/tests/datamodules/test_language_modeling_hf.py

Please refer to https://github.com/Dao-AILab/flash-attention/tree/main for the official demonstration and the source code of FlashAttention.

# More information on FlashAttention
Flashier Attention blog: https://www.adept.ai/blog/flashier-attention 

Tri Dao’s talk: https: //www.youtube.com/watch?v=gMOAud7hZg4

Tri Dao’s talk: https: //www.youtube.com/watch?v=FThvfkXWqtE

ELI5: FlashAttention: https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad

Huggingface: https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention

# Reference
Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv preprint arXiv:2205.14135.

Gordić, A. (2022, July 18). ELI5: FlashAttention. Medium. https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad

Hugging Face. (2023). Flash Attention. Hugging Face Documentation. https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention

Tri Dao. (2023, January 1). FlashAttention - Tri Dao | Stanford MLSys #67. YouTube. https://www.youtube.com/watch?v=gMOAud7hZg4

