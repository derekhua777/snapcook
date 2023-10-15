# SnapCook

SnapCook automatically generates a recipe using images you take of your ingredients. A custom trained ResNet50 model is able to indentify 15 different fruits and vegetables. Using Google's large language model, PaLM 2, SnapCook is able to automatically generate recipes to inspire you for your next meal! 

## Model Training Dependencies

'pip install tensorflow'
'pip install jupyter'

Training Set: https://drive.google.com/drive/folders/1_crWOV6iA_ErSxn9dnu7D8tEvCuXCQem?usp=drive_link
Testing Set: https://drive.google.com/drive/folders/1zAC7hz--bWvRoFfnVBC6Mw-R77inG0mm?usp=drive_link
Trained Model: https://drive.google.com/drive/folders/1xeNrMxBwH4dD3PpjRR-wAZJyowwCCkR1?usp=drive_link

## snapcook.py Dependencies
'pip install google-generativeai'
Generate an API KEY: https://developers.generativeai.google/tutorials/setup for line 10 in snapcook.py
