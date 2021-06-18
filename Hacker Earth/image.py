### Binary Image classification
## Pytorch
'''
pip install tez
'''
import os
import albumentations
import matplotlib.pyplot as plt

import tez
from tez.datasets import ImageDataset
from tez.callbacks import EarlyStopping

import torch
import torch.nn as nn

from sklearn import metrics, model_selection

## Read the dataste
df = pd.read_csv()
df.head() # image name class

df.Condition.value_count()
## Checked if biased or not --> Biased towards 1
df_train, df_valid = mmode_selection.train_test_split(
  df,
  test_size = 0.1,
  random_state = 42,
  stratify =df.Condition.values
)
df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)

## Shpesof the dataset
df_train.shape
df_valid.shape

## Loading the images
image_path = ''
train_image_paths=[
  os.path.join(image_path, x) for x in df_train.image_id.values
]

valid_image_paths=[
  os.path.join(image_path, x) for x in df_valid.image_id.values
]

train_targets = df_train.Condition.values
valid_targets = df_valid.Condition.values

train_dataset = ImageDataset(
  image_paths = train_image_paths,
  targets=train_targets,
  resize=(256,256),
  augmentations=None
)
train_dataset[0]

def plot_img(image_dict):
  image_tensor = image_dict['image']
  target = img_dict['Conditions']
  plt.figure(figsize=(10,10))
  image=img_tensor.permute(1,2,0)/255
  plt.imshow(image)

 plot_img(train_dataset[10])

# augmentations taken from: https://www.kaggle.com/khyeh0719/pytorch-efficientnet-baseline-train-amp-aug
train_aug = albumentations.Compose([
            albumentations.RandomResizedCrop(256, 256),
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
            albumentations.CoarseDropout(p=0.5),
            albumentations.Cutout(p=0.5)], p=1.)
  
        
valid_aug = albumentations.Compose([
            albumentations.CenterCrop(256, 256, p=1.),
            albumentations.Resize(256, 256),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            )], p=1.)


class Model(tez.Model):
  def __init__(self, num_classes, pretrained=True):
    super().__init__()
    self.convnet = torchvision.models.resnet18(pretrained=pretrained)
    self.convnet.fc = nn.Linear(512, num_classes)
    self.step_scheduler_after="epoch"
    
  def loss(self, outputs, targets):
    if targets is None:
      return None
    return nn.CrossEntropyLoss()(outputs, targets)
  def metrics(self, outputs, targets):
    outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    acc = metrics.f1_score(targets, outputs)
    return {
      "accuracy":acc
    }
  
  def fetch_optimizer(self):
    opt=torch.optim.Adam(self.parameters(), lr=1e-3)
    return opt
  
  def fetch_scheduler(self):
    sch=torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=0.7)
    return sch
  
  def forward(self, image, targets=None):
    outputs = self.convnet(image)
    if targets is not None:
      loss = self.loss(outputs, targets)
      m_metrics = self.monitor_metrics(outputs, targets)
      return outputs, loss, m_metrics
    return outputs, None, None

model =Model(num_classes=df.Condition.nunique(), pretrained=True)\

es = EarlyStopping(
    monitor="valid_accuracy", model_path="model.bin", patience=2, mode="max"
)
model.fit(
    train_dataset,
    valid_dataset=valid_dataset,
    train_bs=32,
    valid_bs=64,
    device="cuda",
    epochs=50,
    callbacks=[es],
    fp16=True,
)
