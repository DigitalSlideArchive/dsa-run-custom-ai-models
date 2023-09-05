import os
import numpy as np
import torch
from monai.networks.nets import DenseNet121
from PIL import Image
from scipy.io import loadmat
import json
import torch.nn.functional as F

def run_ai_model_inferencing(json_data):
    image_data = json_data.get("image")
    mask_data = json_data.get("mask")
    annot_data = json_data.get("nuclei")
    size_data = json_data.get("tilesize")
    image = np.array(image_data)
    mask = np.array(mask_data)
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

    #image = Image.open(image_file).convert("RGB")
    #m = loadmat(label_mat)
    #instances = m["inst_map"]
    
    # mask =
    img = image#np.asarray(image)
    instances = mask
    print('image and mask shape',img.shape, mask.shape)
    nucleiClass = []
    gx, gy, gh, gw = size_data[0], size_data[1], size_data[2], size_data[3]
    print('gx gy',gx, gy, gh, gw)

    patch_size = 128
    for element in annot_data[0]:
        xr, yr = element['center'][0], element['center'][1]
        print('xr yr',xr, yr)
        x = xr - (gx + gw / 2)
        y = yr - (gy - gh / 2)
        x_start = int(max(x - patch_size / 2, 0))
        y_start = int(max(y - patch_size / 2, 0))
        x_end = int(x_start + patch_size) # Ensure within image boundaries
        y_end = int(y_start + patch_size)  # Ensure within image boundaries
        print('start and end',x_start,x_end,y_start, y_end)

        # Crop the image and label
        cropped_image_np = img[x_start: x_end, y_start: y_end, :]
        cropped_label_np = instances[x_start: x_end, y_start: y_end]

        if cropped_image_np.shape[0] == 128 and cropped_image_np.shape[1] == 128 and cropped_image_np.shape[2] == 3:

            # Cropped Label
            zero_channel = torch.from_numpy(cropped_label_np.astype(np.float32)).unsqueeze(-1)

            # Zero label
            # zero_channel = torch.zeros(128, 128, 1)
            cropped_image_np = torch.cat((torch.from_numpy(cropped_image_np), zero_channel), dim=2)

            # Convert to torch tensors
            cropped_img = cropped_image_np.permute(2, 0, 1).unsqueeze(0)
            cropped_label = torch.from_numpy(cropped_label_np).unsqueeze(-1)

            # Ensure data types
            cropped_img = cropped_img.float()  # Convert to float if not already
            cropped_label = cropped_label.long()  # Assuming it's a label

            # Perform network inference
            network.eval()
            with torch.no_grad():
                pred = network(cropped_img)[0]
                out = F.softmax(pred, dim=0)
                out = torch.argmax(out, dim=0)
                nucleiClass.append(out.item())
        else:
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
