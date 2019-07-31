from __future__ import print_function
import argparse
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
from torchvision import datasets, transforms
from lenet import Net
from PIL import Image #, ImageEnhance
#import matplotlib.pyplot as plt


use_custom_image = True
test_index = 0 # If use_custom_image is false, index of MNIST test point to use
if use_custom_image:
    # Put path to your image here
    image_path = 'C:/Users/lchen/OneDrive/Pictures/delete2.jpg'
    image_threshold = 0.2 # Value at which to split image into dark and light
    rotate_image = False
    
    #load the image
    image = Image.open(image_path)
    
    if rotate_image:
        image = image.transpose(Image.ROTATE_270) #In case the image is sideways
    #enhancer = ImageEnhance.Contrast(image)
    #image = enhancer.enhance(4.0)
    
    #image.show()
    #format image for network
    transformer = transforms.Compose([
    					transforms.Grayscale(), # Make black-and-white
                     #    transforms.CenterCrop(1024),
    					transforms.Resize((32,32)),
    					transforms.ToTensor(),
    				])
    image = transformer(image)
    image = 1-image # Flip background from light to dark
    image[image <= image_threshold] = 0 
    image[image > image_threshold] = 1
    #make a batch of 1 image
    image = image.unsqueeze(0)
    
else:
    # Use images from the MNIST test set
    testdata = datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
                               transforms.Pad(2),
                               transforms.ToTensor(),
                           ]))
    image, y = testdata[test_index]
    print("True class,", y)
    image = image.unsqueeze(0)

#load the saved model
model_path = 'models/lenet_epoch010.pth'
model = Net()
model.load_state_dict(torch.load(model_path)['state_dict'])
model.eval()


#apply the network
class_probs = model(image)
print(class_probs)
class_  = torch.argmax(class_probs)
print('prediction: ', class_.item())
processed_im = transforms.functional.to_pil_image(image.squeeze())
processed_im.show()
#plt.imshow(image.squeeze())
#plt.show()
