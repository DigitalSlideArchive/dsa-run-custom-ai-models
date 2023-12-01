import matplotlib.pyplot as plt
import numpy as np


def run_ai_model_inferencing(json_data):
    """
    Template provided for developers to connect custom AI, segmentation 
    and classification models with DSA AI Adapter.

    Args:
        json_data (dict): A dictionary containing image, mask, nuclei, and tilesize data.

    Returns:
        list: List of predicted class labels.
    """
    image_data = json_data.get("image")
    # mask_data = json_data.get("mask")
    # annot_data = json_data.get("nuclei")
    # size_data = json_data.get("tilesize")
    image = np.array(image_data)
    # mask = np.array(mask_data)
    # mask = (mask > 0).astype(np.uint8)

    # Save image and mask
    plt.imsave('incoming_image.jpg', np.uint8(image))

    # write annotations to a file
    # with open('annotation.txt', 'a') as file:
    #     file.write(annot_data + size_data)

    # Add your AI model here for anayzing the following data send from DSA-AI-Adapter
    # 1- Image
    # 2 - Mask
    # 3 - Annotations

    # if you are sending nuclei annotations back to DSA
    # Please refer to the code - https://github.com/DigitalSlideArchive/dsa-run-custom-ai-models/blob/master/aiInferenceModel/ai_models/nuclickSegmentation.py
    ''''nuclei_annot_list = []
    cur_annot = {
                    'type': 'polyline',
                    'points': 'This is where the annotation has to be added',
                    'closed': True,
                    'fillColor': 'rgba(0,0,0,0)',
                    'lineColor': 'rgb(0,255,0)',
                    'group': 'Nuclick segmentation',
                }
    nuclei_annot_list.append(cur_annot)
    return nuclei_annot_list '''

    # if you are sending classification predictions back to DSA
    # Please refere to the classification code - https://github.com/DigitalSlideArchive/dsa-run-custom-ai-models/blob/master/aiInferenceModel/ai_models/nuclickClassification.py
    '''curated_annotation_data = []
    for element in annot_data:
        element['lineColor'] = colormap[out.item()]
        element['group'] = "Nuclick Classification"
        element['label'] = {"value": 'Predicted class label'}
    return curated_annotation_data '''

    return []
