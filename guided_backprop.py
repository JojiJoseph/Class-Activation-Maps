from flask import g
from jax import grad
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import argparse
import numpy as np
from torchvision import transforms

class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.hook_registered = False

    def hook_fn(self, module, grad_input, grad_output):
        self.gradients = torch.nn.functional.relu(grad_input[0])        
        grad_output = (self.gradients,)
        return grad_output

    def guided_backward(self, input_image, target_class):
        # Register hooks to modify gradients during backward pass
        if not self.hook_registered:
            hooks = []
            for layer in self.model.modules():
                if isinstance(layer, nn.ReLU):
                    layer.inplace = False
                    hooks.append(layer.register_backward_hook(self.hook_fn))
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()  # Set BatchNorm layer to evaluation mode
                    for param in layer.parameters():
                        param.requires_grad = False  # Freeze parameters
            self.hook_registered = True

        self.model.zero_grad()
        input_image.requires_grad = True

        output = self.model(input_image)
        print("Class idx", torch.argmax(output))
        print("target class", target_class)
        loss = output[0, target_class]  # Target class score
        # mask = torch.zeros_like(output)
        # mask[0, target_class] = 1
        # loss = torch.sum(output * mask)
#
        loss.backward()
        guided_gradients = input_image.grad  # Guided gradients
        return guided_gradients


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--class-idx', type=int, required=True)
    
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    # model.eval()
    guided_backprop = GuidedBackprop(model)

    # input_image = torch.randn(1, 3, 224, 224, requires_grad=True)  # Example input image
    # target_class = parser.class_idx

    image_path = parser.parse_args().image
    target_class = parser.parse_args().class_idx

    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

    image = cv2.imread(image_path)

    image_resized = cv2.resize(image, (1024, 1024))
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((1024, 1024)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    # image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0)/255.
    image_tensor = transform(image_resized[...,::-1].copy()).unsqueeze(0)
    # print("image tensor_ shape", image_tensor.shape, image_tensor.max(), image_tensor.min())
    # image_tensor = 2 * (image_tensor - 0.5)
    guided_gradients = guided_backprop.guided_backward(image_tensor, target_class)[0].permute(1, 2, 0).detach().numpy()
    output = model(image_tensor)
    print(output.shape)
    print(torch.argmax(output), "argmax")

    # print(guided_gradients.shape)
    # guided_gradients = np.clip(guided_gradients, 0, None)
    guided_gradients = guided_gradients - np.min(guided_gradients)  # Normalize
    guided_gradients = guided_gradients / np.max(guided_gradients)
    image_resized = cv2.resize(image, (224, 224))
    guided_gradients = np.uint8(guided_gradients*255)
    # print(guided_gradients.max(), guided_gradients.min(), guided_gradients.mean(), guided_gradients.std())
    import matplotlib.pyplot as plt
    # # Equalize histogram
    guided_gradients = np.max(guided_gradients, axis=-1, keepdims=True)
    print(guided_gradients.shape)
    # guided_gradients = cv2.merge([cv2.equalizeHist(guided_gradients[:,:,0]),cv2.equalizeHist(guided_gradients[:,:,0]),cv2.equalizeHist(guided_gradients[:,:,0])])
    # guided_gradients = cv2.merge([cv2.equalizeHist(guided_gradients[:,:,0]),cv2.equalizeHist(guided_gradients[:,:,1]),cv2.equalizeHist(guided_gradients[:,:,2])])
    guided_gradients = cv2.merge([guided_gradients,guided_gradients,guided_gradients])
    plt.imshow(guided_gradients)
    hist = cv2.calcHist([guided_gradients], [0], None, [256], [0, 256])
    plt.figure()
    plt.plot(hist)
    idx = np.where(hist == np.max(hist))[0]
    print(idx)
    guided_gradients[guided_gradients < idx[0]] = 0
    plt.show()
    plt.imshow(guided_gradients)
    # guided_gradients[guided_gradients > 128] = 255
    plt.show()
    # guided_gradients = cv2.applyColorMap(guided_gradients, cv2.COLORMAP_JET)
    # image_out = cv2.addWeighted(image_resized, 0.5, guided_gradients, 0.5, 0)
    image_out = cv2.multiply(image_resized, guided_gradients/255., dtype=cv2.CV_32F)
    # print(image_out.min(), image_out.max())
    image_out = np.uint8(image_out)
    plt.imshow(image_out)
    plt.show()

    """
    image net indices for cats,
    281: tabby, tabby cat
    282: tiger cat
    283: Persian cat

    463: Egyptian cat
    """

    # cam = generate_cam(model, image_tensor, class_idx)
    # cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    # cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    # cv2.namedWindow("Cam", cv2.WINDOW_NORMAL)
    # img_out = cv2.addWeighted(image, 0.5, cam, 0.5, 0)
    # cv2.imshow("Cam", img_out)
    # cv2.waitKey(0)
