import os
import numpy as np
import torch
from monai.networks.nets import DenseNet121
from PIL import Image
from scipy.io import loadmat
import json
import torch.nn.functional as F
import cv2
from monai.transforms import LoadImaged, EnsureChannelFirstd, Compose, ScaleIntensityRangeD
from monai.data import PILReader
from monai.apps.nuclick.transforms import AddLabelAsGuidanced
import matplotlib.pyplot as plt

def run_ai_model_inferencing(json_data):
    image_data = json_data.get("image")
    mask_data = json_data.get("mask")
    annot_data = json_data.get("nuclei")
    size_data = json_data.get("tilesize")
    image = np.array(image_data)
    mask = np.array(mask_data)
    mask = (mask > 0).astype(np.uint8)
    model_weights_path = "model.pt"

    class_names = {
        "0": "Other",
        "1": "Inflammatory",
        "2": "Epithelial",
        "3": "Spindle-Shaped",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = DenseNet121(spatial_dims=2, in_channels=4, out_channels=len(class_names))

    checkpoint = torch.load(model_weights_path, map_location=torch.device(device))
    model_state_dict = checkpoint.get("model", checkpoint)
    network.load_state_dict(model_state_dict, strict=True)
    
    img = image
    instances = mask
    nucleiClass = []
    gx, gy, gh, gw, xc, yc  = size_data[0], size_data[1], size_data[2], size_data[3], size_data[4], size_data[5]
    #print('image and mask shape',img.shape, mask.shape, h, w)

    patch_size = 64
    for element in annot_data[0]:
        xr, yr, xw, yh = element['center'][0], element['center'][1], element['width'], element['height']
        x = xr - xc
        y = yr - yc
        x_start = int(max(x - patch_size / 2, 0))
        y_start = int(max(y - patch_size / 2, 0))
        x_end = int(x_start + patch_size)# Ensure within image boundaries
        y_end = int(y_start + patch_size) # Ensure within image boundaries

        #remove any unwanted labels
        zero_label = np.zeros(instances.shape)
        zero_label[int(y-yh/2):int(y+yh/2),int(x-xw/2):int(x+xw/2)] = 1
        localized = instances * zero_label

        # Crop the image and label
        cropped_image_np = img[y_start: y_end, x_start: x_end, :]
        cropped_label_np = localized[y_start: y_end,x_start: x_end]
        
        if cropped_image_np.shape[0] == 64 and cropped_image_np.shape[1] == 64 and cropped_image_np.shape[2] == 3:
            # if cropped_image_np.shape[0] != 128 or cropped_image_np.shape[1] != 128:
            cropped_image_np = cv2.resize(cropped_image_np.astype(np.uint8), (128,128))
            cropped_label_np = cv2.resize(cropped_label_np.astype(np.uint8),(128,128))
            cv2.imwrite('image.png', cropped_image_np)
            plt.imsave('mask.png', cropped_label_np)
            transforms = Compose([
                            LoadImaged(keys="image", dtype=np.uint8, reader=PILReader(converter=lambda im: im.convert("RGB"))),
                            LoadImaged(keys="label", dtype=np.uint8, reader=PILReader(converter=lambda im: im.convert("L"))),
                            EnsureChannelFirstd(keys=("image", "label")),
                            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
                            AddLabelAsGuidanced(keys="image", source="label"),])
            
            input_data = {"image": 'image.png', "label":'mask.png'}
            output_data = transforms(input_data)
            network.eval()
            with torch.no_grad():
                pred = network(output_data["image"][None])[0]
                out = F.softmax(pred, dim=0)
                out = torch.argmax(out, dim=0)
                nucleiClass.append(out.item())
        else:
            print('skipped', cropped_image_np.shape, y_start, y_end)
            nucleiClass.append(4)
    print(nucleiClass)           
    return nucleiClass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run AI Model Inference")
    parser.add_argument("image_file", help="Path to the image file")
    parser.add_argument("json_file", help="Path to the JSON file")
    parser.add_argument("label_mat", help="Path to the label .mat file")
    args = parser.parse_args()

    results = run_ai_model_inferencing(args.image_file, args.json_file, args.label_mat)
    print(results)
