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


def run_ai_model_inferencing(json_data, network):
    print(json_data.keys())
    image_data = json_data.get("image")
    foreground_data = json_data.get("nuclei_location")
    size_data = json_data.get("tilesize")
    gx,gy,_,_,x,y = size_data
    image = np.array(image_data)
    temp_dir = tempfile.mkdtemp()

    # infereing nuclei location #TODO
    print('Tile reference',size_data)
    print("\n")
    print('nuclei locations', foreground_data)
    print("\n")
    print("image size", image.shape)

    #adding tile reference to the input cordinates
    for element in foreground_data:
        element[0] = int(np.abs(element[0] - x))
        element[1] = int(np.abs(element[1] - y))
    print('updated foreground ', foreground_data)
    ##############################################

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

    if len(nuclei_obj_props) >= 1:
        for i in range(len(nuclei_obj_props)):
            cx = nuclei_obj_props[i].centroid[1]
            cy = nuclei_obj_props[i].centroid[0]
            width = nuclei_obj_props[i].bbox[3] - \
                nuclei_obj_props[i].bbox[1] + 1
            height = nuclei_obj_props[i].bbox[2] - \
                nuclei_obj_props[i].bbox[0] + 1

            # generate contours
            zero_image = np.zeros(output_predictions["pred"].shape)

            zero_image[int(cy-width/2):int(cy+width/2), int(cx-height/2):int(cx + height/2)] = output_predictions["pred"][int(cy-width/2):int(cy+width/2), int(cx-height/2):int(cx + height/2)]
            contours, _ = cv2.findContours(zero_image.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            output_list = [[[x[0][1] + gx, x[0][0] + gy, 0]
                            for x in arr.tolist()] for arr in list(contours)]

            # create annotation json
            for record in output_list:
                cur_annot = {
                    'type': 'polyline',
                    'points': record,
                    'closed': True,
                    'fillColor': 'rgba(0,0,0,0)',
                    'lineColor': 'rgb(0,255,0)',
                    'group': 'Nuclick segmentation',
                }

                nuclei_annot_list.append(cur_annot)
    return nuclei_annot_list


if __name__ == "__main__":
    # Code for running the AI model locally
    image_file = ""
    foreground = [[190, 15], [218, 32], [296, 96]]
    reader = PILReader(converter=lambda im: im.convert("RGB"))
    image_np = LoadImage(
        image_only=True,
        dtype=np.uint8,
        reader=reader)(image_file)
    payload = {"image": image_np, "nuclei_location": str(foreground)}
    output = run_ai_model_inferencing(payload)
    print(output)
