import os
import tempfile

import cv2
import numpy as np
import torch
from monai.apps.nuclick.transforms import AddClickSignalsd, PostFilterLabeld
from monai.data import PILReader
from monai.networks.nets import BasicUNet
from monai.transforms import (Activationsd, AsDiscreted, Compose,
                              EnsureChannelFirstd, LoadImage, LoadImaged,
                              ScaleIntensityRangeD, SqueezeDimd)
from skimage.measure import label, regionprops


def run_ai_model_inferencing(json_data):
    print(json_data.keys())
    image_data = json_data.get("image")
    foreground_data = eval(json_data.get("nuclei_location"))
    size_data = json_data.get("tilesize")
    image = np.array(image_data)
    model_weights_path = "./models/nuclickSegmentation.pt"
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    network = BasicUNet(
        spatial_dims=2,
        in_channels=5,
        out_channels=1,
        features=(
            32,
            64,
            128,
            256,
            512,
            32))
    checkpoint = torch.load(
        model_weights_path,
        map_location=torch.device(device))
    model_state_dict = checkpoint.get("nuclickSegmentation", checkpoint)
    network.load_state_dict(model_state_dict, strict=True)

    # Transforms
    pre_transforms = Compose([
        LoadImaged(
            keys="image",
            dtype=np.uint8,
            image_only=True,
            reader=PILReader(
                converter=lambda im: im.convert("RGB"))),
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityRangeD(
            keys="image",
            a_min=0.0,
            a_max=255.0,
            b_min=-1.0,
            b_max=1.0),
        AddClickSignalsd(
            image="image",
            foreground="foreground",
            gaussian=False),
    ])
    cv2.imwrite(os.path.join(temp_dir, 'image.png'), image)

    data = {
        "image": os.path.join(
            temp_dir,
            'image.png'),
        "foreground": foreground_data}

    # print(
    #     data["image"].shape,
    #     type(
    #         data["image"]),
    #     type(foreground),
    #     foreground_data)
    print(type(foreground_data), foreground_data)
    data = pre_transforms(data)

    # prediction
    network.eval()
    with torch.no_grad():
        pred = network(data["image"])

    # Transforms
    post_transforms = Compose([
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5, dtype=np.uint8),
        SqueezeDimd(keys="pred", dim=1),
        PostFilterLabeld(keys="pred"),])

    data["pred"] = pred
    output_predictions = post_transforms(data)

    nuclei_obj_props = regionprops(label(output_predictions["pred"]))

    nuclei_annot_list = []

    if len(nuclei_obj_props) > 1:
        for i in range(len(nuclei_obj_props)):
            cx = nuclei_obj_props[i].centroid[1]
            cy = nuclei_obj_props[i].centroid[0]
            width = nuclei_obj_props[i].bbox[3] - \
                nuclei_obj_props[i].bbox[1] + 1
            height = nuclei_obj_props[i].bbox[2] - \
                nuclei_obj_props[i].bbox[0] + 1
            print(cx, cy, width, height)

            cur_bbox = {
                'type': 'rectangle',
                'center': [cy, cx, 0],
                'width': width,
                'height': height,
                'rotation': 0,
                'fillColor': 'rgba(0,0,0,0)',
                'lineColor': 'rgb(0,255,0)'
            }

            nuclei_annot_list.append(cur_bbox)

    return nuclei_annot_list


if __name__ == "__main__":
    image_file = os.path.join('../debug/workspace/', 'test_12.png')
    foreground = [[190, 15], [218, 32], [296, 96]]
    reader = PILReader(converter=lambda im: im.convert("RGB"))
    image_np = LoadImage(
        image_only=True,
        dtype=np.uint8,
        reader=reader)(image_file)
    payload = {"image": image_np, "nuclei_location": str(foreground)}
    output = run_ai_model_inferencing(payload)
    print(output)