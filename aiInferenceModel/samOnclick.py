from segment_anything import SamPredictor, sam_model_registry
import os
import tempfile

import cv2
import numpy as np

DEVICE = "cpu"  # Can be changed to a specific hardware device, e.g., "cuda:0" for GPU
MODEL_TYPE = "vit_h"  # The type of model to use, e.g., "vit_h"

# Import necessary modules from the "segment_anything" package


def run_ai_model_inferencing(json_data):
    """
    Run inference using a Semantic Segmentation model on input image data.

    Args:
        json_data (dict): JSON data containing image and size information.
            It should have keys: "image" and "tilesize".

    Returns:
        list: A list of dictionaries representing annotations for detected objects.

    Notes:
        - This function processes the input image using a Semantic Segmentation model.
        - Detected objects are converted into annotations with specified attributes.
    """
    """
    This code uses Facebook segment anything model.

    authors : Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, 
    Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo,
      Wan-Yen and Doll, Piotr and Girshick, Ross
    """

    print('Running Segment anything facebook model')

    # Extract image data and size information from JSON input
    image_data, size_data, foreground_data = json_data.get("image"), json_data.get("tilesize"), json_data.get("nuclei_location")
    gx, gy, _, _, x, y = size_data

    #adding tile reference to the input cordinates
    for element in foreground_data:
        element[0] = int(np.abs(element[0] - x))
        element[1] = int(np.abs(element[1] - y))
    print('updated foreground ', foreground_data)
    ##############################################

    # Convert image data to a NumPy array and create a temporary directory
    image_np, temp_dir = np.array(image_data), tempfile.mkdtemp()
    cv2.imwrite(os.path.join(temp_dir, 'image.png'), image_np)
    image = cv2.imread(os.path.join(temp_dir, 'image.png'))
    print(f"image shape {image.shape}")

    # Define the path to the pre-trained model checkpoint
    CHECKPOINT_PATH = "/home/local/KHQ/s.erattakulangara/Documents/HistomicsTK_EKS/dsa-run-custom-ai-models/debug/weights/vit_h_lm.pth"

    # Initialize the Semantic Segmentation model and move it to the specified
    # device
    sam = sam_model_registry[MODEL_TYPE](
        checkpoint=CHECKPOINT_PATH).to(
        device=DEVICE)

    # Generate masks for the input image using the model
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
    point_coords = np.array(foreground_data),
    point_labels = input_label,
    multimask_output = False)

    # Analyze the segmentation results and create annotations for detected
    print(f"masks shape: {masks.shape}")
    contours, _ = cv2.findContours(masks[0].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #select the largest contours
    output_list = [[[x[0][0] + gx, x[0][1] + gy, 0]
                            for x in arr.tolist()] for arr in list(contours)]
    nuclei_annot_list = []
    for record in output_list:
        cur_annot = {
            'type': 'polyline',
            'points': record,
            'closed': True,
            'fillColor': 'rgba(0,0,0,0)',
            'lineColor': 'rgb(0,255,0)',
            'group': 'Segment anything onclick'
        }
        nuclei_annot_list.append(cur_annot)

    return nuclei_annot_list


if __name__ == "__main__":
    # Example usage when run as a standalone script
    image_file = "/home/local/KHQ/s.erattakulangara/Documents/HistomicsTK_EKS/dsa-run-custom-ai-models/debug/workspace/test_12.png"
    image = cv2.imread(image_file)[:500,:500,:]
    payload = {"image": image, "tilesize":(0,0,0,0,0,0),"nuclei_location":[[288, 302]]}
    output = run_ai_model_inferencing(payload)
    print(output)
