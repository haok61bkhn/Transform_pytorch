import cv2
import numpy as np
import torch
from torch.autograd import Variable
class Transform:
    def __init__(self,mean=[0.5,0.5,0.5],std=[0.5, 0.5, 0.5]):
        self.pixel_scale = float(1.0)
        self.pixel_means = np.array(mean, dtype=np.float32)
        self.pixel_stds = np.array(std, dtype=np.float32)
    

    def process_image(self,im):
        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
        for i in range(3):
            im_tensor[0, i, :, :] = (im[:, :, 2 - i] / self.pixel_scale - self.pixel_means[2 - i])/self.pixel_stds[2 - i]
        return im_tensor

    def transform(img):#cv2
        img=self.process_image(img)
        img /= 255.0  
        data = torch.from_numpy(img).float().to(self.device)
        data = Variable(data)
        return data


    
