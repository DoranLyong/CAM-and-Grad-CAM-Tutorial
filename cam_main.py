# coding = <utf-8> 
"""
(ref) https://github.com/tyui592/class_activation_map/blob/master/cam.py
(ref) https://github.com/DoranLyong/DeepLearning_model_factory/blob/master/ML_tutorial/PyTorch/Basics/03_simple_CNN.py
(ref) https://github.com/frgfm/torch-cam
(ref) https://github.com/KangBK0120/CAM
(ref) https://github.com/chaeyoung-lee/pytorch-CAM

"""



# %%
import sys 
import os.path as osp 

import numpy as np
import matplotlib.pyplot as plt 
import torch 
from torch.utils.data import DataLoader 

import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms  

from model.models_cam import CAM



# ================================================================= #
#                         1. Set device                             #
# ================================================================= #
# %% 01. 프로세스 장비 설정 
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')




# ================================================================= #
#              2. Create and Initialize network                     #
# ================================================================= #
# %%  02. 모델 생성 및 초기화 
model = CAM('resnet50').to(device)










# ================================================================= #
#                       3. Hyperparameters                          #
# ================================================================= #
# %% 03. 하이퍼파라미터 설정 
in_channel = 1
num_classes = 10 
learning_rate = 0.001
batch_size = 64
num_epochs = 5



# ================================================================= #
#                         4.  Load Data                             #
# ================================================================= #
# %% 04. 데이터 로드
"""
(ref) https://blog.ees.guru/49
(ref) https://poddeeplearning.readthedocs.io/ko/latest/CNN/VGG19%20+%20GAP%20+%20CAM/
(ref) https://github.com/DoranLyong/DeepLearning_model_factory/blob/master/ML_tutorial/PyTorch/Basics/03_simple_CNN.py
"""
torch.manual_seed(42)


transform = transforms.Compose([    transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])




train_dataset = datasets.STL10( root='dataset/', 
                                split='train', 
                                transform = transform ,
                                download=True,
                                )

train_loader = DataLoader(  dataset=train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                        )


test_dataset = datasets.STL10( root='dataset/', split='test', transform = transform ,  download=True  )
test_loader = DataLoader(  dataset=test_dataset, batch_size=batch_size, shuffle=False, )






def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
classes =  ('airplance', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



# ================================================================= #
#                            5.  CAM                                #
# ================================================================= #
# %%  05. 손실 함수와 최적화 알고리즘 정의 
model.eval()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

for data in test_loader:    
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    outputs, f = model(images)
    _, predicted = torch.max(outputs, 1)
    break


params = list(model.parameters())

num = 0
for num in range(1):
    print("ANS :",classes[int(predicted[num])]," REAL :",classes[int(labels[num])],num)

    #print(outputs[0])

    overlay = params[-2][int(predicted[num])].matmul(f[num].reshape(512,49)).reshape(7,7).cpu().data.numpy()

    overlay = overlay - np.min(overlay)
    overlay = overlay / np.max(overlay)

    imshow(images[num].cpu())
    skimage.transform.resize(overlay, [224,224])
    plt.imshow(skimage.transform.resize(overlay, [224,224]), alpha=0.4,cmap='jet')
    plt.show()
    imshow(images[num].cpu())
    plt.show()



# ================================================================= #
#                      5.  Loss and optimizer                       #
# ================================================================= #

# %%  05. 손실 함수와 최적화 알고리즘 정의 
