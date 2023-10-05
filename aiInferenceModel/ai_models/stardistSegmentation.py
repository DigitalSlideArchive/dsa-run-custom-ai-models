import numpy as np
import tempfile
import cv2
import os
from csbdeep.utils import normalize

def run_ai_model_inferencing(json_data, model):
    # Extract image data and size information from JSON input
    image_data, size_data, foreground_data = json_data.get(
        "image"), json_data.get("tilesize"), json_data.get("nuclei_location")
    gx, gy, _, _, x, y = size_data

    # adding tile reference to the input cordinates
    for element in foreground_data:
        element[0] = int(np.abs(element[0] - x))
        element[1] = int(np.abs(element[1] - y))
    print('updated foreground ', foreground_data)
    ##############################################

    # Convert image data to a NumPy array and create a temporary directory
    image_np, temp_dir = np.array(image_data), tempfile.mkdtemp()
    cv2.imwrite(os.path.join(temp_dir, 'image.png'), image_np)
    image = cv2.imread(os.path.join(temp_dir, 'image.png'))
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"image shape {image.shape}")

    nuclei_annot_list = []

    #predict using the model
    _, data =  model.predict_instances(normalize(image))

    for element in data['coord']:
        xa = np.round(np.abs(element[0]) + y)
        ya = np.round(np.abs(element[1]) + x)
        za = np.zeros(element[0].shape)
        coord = np.vstack((ya,xa,za)).T.tolist()
        cur_annot = {
            'type': 'polyline',
            'points': coord,
            'closed': True,
            'fillColor': 'rgba(0,0,0,0)',
            'lineColor': 'rgb(0,255,0)',
            'group': 'Stardist H & E Segmentation'
        }
        nuclei_annot_list.append(cur_annot)

    print(nuclei_annot_list)
    return nuclei_annot_list

if __name__ == "__main__":
    # Example usage when run as a standalone script
    image_file = "/home/local/KHQ/s.erattakulangara/Documents/HistomicsTK_EKS/dsa-run-custom-ai-models/debug/workspace/test_12.png"
    image = cv2.imread(image_file)[:500, :500, :]
    payload = {"image": image, "tilesize": (
        0, 0, 0, 0, 0, 0), "nuclei_location": [[288, 302]]}
    output = run_ai_model_inferencing(payload)
    print(output)