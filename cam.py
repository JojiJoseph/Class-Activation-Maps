import argparse
import cv2
import numpy as np
import torch

def generate_cam(model, image_tensor, class_idx):
    model.eval()
    feature_map = None
    def activation_hook_func(module, inp, out):
        nonlocal feature_map
        feature_map = inp[0].detach()
    
    model.avgpool.register_forward_hook(activation_hook_func)

    scores = model(image_tensor)

    weights = model.fc.weight

    weights = weights[class_idx][None   , :, None, None]


    cam = weights * feature_map


    cam = cam.detach().numpy()

    cam = np.sum(cam[0],0 )

    cam = cv2.resize(cam, (image_tensor.shape[2], image_tensor.shape[3]))  # Resize to the input image size
    cam = cam - np.min(cam)  # Normalize
    cam = cam / np.max(cam)
    cam = np.uint8(cam*255)
    return cam
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--class-idx', type=int, required=True)

    image_path = parser.parse_args().image
    class_idx = parser.parse_args().class_idx

    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

    image = cv2.imread(image_path)

    image_resized = cv2.resize(image, (1024, 1024))
    image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0)/255.
    image_tensor = 2 * (image_tensor - 0.5)

    cam = generate_cam(model, image_tensor, class_idx)
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cv2.namedWindow("Cam", cv2.WINDOW_NORMAL)
    img_out = cv2.addWeighted(image, 0.5, cam, 0.5, 0)
    cv2.imshow("Cam", img_out)
    cv2.waitKey(0)
