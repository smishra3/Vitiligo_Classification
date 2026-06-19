# Vitiligo_Classification
source code for vitiligo classification

## Pretrained Models

Pretrained weights are published as release assets under [v1.0.0](https://github.com/smishra3/Vitiligo_Classification/releases/tag/v1.0.0):

| Architecture | File | Size |
|---|---|---|
| DenseNet | [`densenet_vitiligo.pth`](https://github.com/smishra3/Vitiligo_Classification/releases/download/v1.0.0/densenet_vitiligo.pth) | 27 MB |
| ResNet | [`resnet_vitiligo.pth`](https://github.com/smishra3/Vitiligo_Classification/releases/download/v1.0.0/resnet_vitiligo.pth) | 43 MB |
| VGG | [`vgg_vitiligo.pth`](https://github.com/smishra3/Vitiligo_Classification/releases/download/v1.0.0/vgg_vitiligo.pth) | 492 MB |

Download:
```bash
wget https://github.com/smishra3/Vitiligo_Classification/releases/download/v1.0.0/vgg_vitiligo.pth
```

Load (weights are saved as state dicts):
```python
import torch
from torchvision import models

model = models.vgg16(num_classes=2)
model.load_state_dict(torch.load('vgg_vitiligo.pth', map_location='cpu'))
model.eval()
```

## Training & Evaluation

run the vitiligo_vgg.py file for training and validation

For testing, run the new_evaluation.py file

Path to data directory
  --> please use the format data/{train,val}/{Yes,No}
  --> use the data path in both training and evaluation files
