# abg-comfyui
A Anime Character Segmentation node for comfyui, based on [this hf space](https://huggingface.co/spaces/skytnt/anime-remove-background), works same as [ABG extention in automatic1111](https://github.com/KutsuyaYuki/ABG_extension/tree/main)


# Installation
1. git clone this repo to the custom_nodes directory
```
git clone https://github.com/LyazS/abg-comfyui.git
```

2. Download dependencys on requirements.txt on comfyui
```
pip install -r requirements.txt
```
# Usage
Create a "mask/Anime Character Seg" node, and connect the images to input, and it would segment the anime character and output the masks.
