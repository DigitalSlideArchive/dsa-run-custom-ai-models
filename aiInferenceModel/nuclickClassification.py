import os
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.apps.nuclick.transforms import AddLabelAsGuidanced
from monai.data import PILReader
from monai.networks.nets import DenseNet121
from monai.transforms import (Compose, EnsureChannelFirstd, LoadImaged,
                              ScaleIntensityRangeD, Transform)


class BinarizeImage(Transform):
    """
    Custom MONAI transform to binarize image data.
    """

    def __init__(self, threshold=128, keys="label"):
        super().__init__()
        self.threshold = threshold
        self.keys = keys

    def __call__(self, data):
        data[self.keys] = (data[self.keys] >=
                           self.threshold).astype(np.float32)
        return data


def run_ai_model_inferencing(json_data):
    print("Running Nuclei Classification AI Model")
    """
    Run inference using an AI model on input data provided in JSON format.

    Args:
        json_data (dict): A dictionary containing image, mask, nuclei, and tilesize data.

    Returns:
        list: List of predicted class labels.
    """
    image_data = json_data.get("image")
    mask_data = json_data.get("mask")
    annot_data = json_data.get("nuclei")
    size_data = json_data.get("tilesize")
    image = np.array(image_data)
    mask = np.array(mask_data)
    mask = (mask > 0).astype(np.uint8)
    model_weights_path = "./models/nuclick.pt"
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    class_names = {
        "0": "Other",
        "1": "Inflammatory",
        "2": "Epithelial",
        "3": "Spindle-Shaped",
    }

    device = torch.device("cpu")
    network = DenseNet121(
        spatial_dims=2,
        in_channels=4,
        out_channels=len(class_names))

    checkpoint = torch.load(
        model_weights_path,
        map_location=torch.device(device))
    model_state_dict = checkpoint.get("nuclick", checkpoint)
    network.load_state_dict(model_state_dict, strict=True)

    img = image
    instances = mask
    nucleiClass = []
    _, _, _, _, xc, yc = size_data[0], size_data[1], size_data[2], size_data[3], size_data[4], size_data[5]

    colormap = {
                0: 'rgb(0,0,255)',
                1: 'rgb(0,255,0)',
                2: 'rgb(255,0,0)',
                3: 'rgb(255,255,0)',
                4: 'rgb(255,0,255)'}
    classnames = {
                0: "Other-Blue",
                1: "Inflammatory-Green",
                2: "Epithelial-Red",
                3: "Spindle-Shaped-Yellow",
                4: 'Cannot-be-processed-Pink'
            }
    patch_size = 64
    curated_annotation_data = []
    for element in annot_data[0]:
        xr, yr, xw, yh = element['center'][0], element['center'][1], element['width'], element['height']
        x = xr - xc
        y = yr - yc
        x_start = int(max(x - patch_size / 2, 0))
        y_start = int(max(y - patch_size / 2, 0))
        x_end = int(x_start + patch_size)
        y_end = int(y_start + patch_size)

        zero_label = np.zeros(instances.shape)
        zero_label[int(y - yh / 2):int(y + yh / 2),
                   int(x - xw / 2):int(x + xw / 2)] = 1
        localized = instances * zero_label

        cropped_image_np = img[y_start: y_end, x_start: x_end, :]
        cropped_label_np = localized[y_start: y_end, x_start: x_end]

        if cropped_image_np.shape[0] == patch_size and cropped_image_np.shape[1] == patch_size and cropped_image_np.shape[2] == 3:
            cropped_image_np = cv2.resize(
                cropped_image_np.astype(
                    np.uint8), (128, 128))
            cropped_label_np = cv2.resize(
                cropped_label_np.astype(
                    np.uint8), (128, 128))
            cv2.imwrite(os.path.join(temp_dir, 'image.png'), cropped_image_np)
            plt.imsave(os.path.join(temp_dir, 'label.png'), cropped_label_np)

            transforms = Compose([
                LoadImaged(
                    keys="image", dtype=np.uint8, reader=PILReader(
                        converter=lambda im: im.convert("RGB"))),
                LoadImaged(
                    keys="label", dtype=np.uint8, reader=PILReader(
                        converter=lambda im: im.convert("L"))),
                EnsureChannelFirstd(keys=("image", "label")),
                ScaleIntensityRangeD(
                    keys="image",
                    a_min=0.0,
                    a_max=255.0,
                    b_min=-1.0,
                    b_max=1.0),
                BinarizeImage(keys=("label"), threshold=128),
                AddLabelAsGuidanced(keys="image", source="label"),
            ])

            input_data = {
                "image": os.path.join(
                    temp_dir, 'image.png'), "label": os.path.join(
                    temp_dir, 'label.png')}
            output_data = transforms(input_data)

            network.eval()
            with torch.no_grad():
                pred = network(output_data["image"][None])[0]
                out = F.softmax(pred, dim=0)
                out = torch.argmax(out, dim=0)
                nucleiClass.append(out.item())
                #add information to the annotation
                element['lineColor'] = colormap[out.item()]
                curated_annotation_data.append(element)
        else:
            nucleiClass.append(4)
            element['lineColor'] = colormap[4]
            curated_annotation_data.append(element)
    return curated_annotation_data
