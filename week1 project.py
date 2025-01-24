# %% [markdown]
# ## Waste Management of plastic using CNN model

# %%
pip install opencv-python

# %%
pip install pandas

# %%
pip install tensorflow

# %%
pip install tqdm

# %%
#importing necessary libraries
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# %%
train_path="DATASET/TRAIN"
test_path="DATASET/TEST"

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras.utils import plot_model
from glob import glob

# %%
#visualisation
from cv2 import cvtColor
x_data=[]
y_data=[]
for category in glob(train_path+'/*'):
    for file in tqdm(glob(category+'/*')):
        img_array=cv2.imread(file)
        img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
        x_data.append(img_array)
        y_data.append(category.split('/')[-1])
data=pd.DataFrame({'image':x_data,'label':y_data})


# %%
data.shape


