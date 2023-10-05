import os
import urllib.request

import torch
from mobile_sam import sam_model_registry
from monai.networks.nets import BasicUNet, DenseNet121
from stardist.models import StarDist2D

ai_model_dir = "./model_weights"


def download_ai_models():
    urls = {'mobile_sam.pt': 'https://data.kitware.com/api/v1/file/hashsum/sha512/d2dcb4448e6f5443383dcd92dd00f0bfeeddad8d2ddabc3481297f67a8ca4095517a8225545c6c2461d9a00722f9277cf2ce420ba4f389f2e77a253cecf8f55f/download',
            'nuclick.pt': 'https://data.kitware.com/api/v1/file/hashsum/sha512/2c8fdba51313ed049ad13149547862bd4e1cbfef2bea7c3567c660019f48047788fa48f6e57c77d016e78040d56f188f44f15e1c1d6887ca1d03f2dea2580902/download',
            'nuclickSegmentation.pt': 'https://data.kitware.com/api/v1/file/hashsum/sha512/fd7b10e5aba63856e2cf24aa3ab15e6e6f8ef4d728812267843f37b68c0909d5bae52908eb6b089ffd88636d20dd1eb4684ed5ea98f1879652d345308bc83ada/download',
            'segmentAnything.pth': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            "microSegmentAnything.pth": "https://zenodo.org/record/8250299/files/vit_h_lm.pth?download=1"
            }
    if not os.path.exists(ai_model_dir):
        print('Downloading AI models')
        os.makedirs(ai_model_dir)

    # Download AI models
    for model_name, model_url in urls.items():
        model_path = os.path.join(ai_model_dir, model_name)
        if os.path.exists(model_path):
            print(f"'{model_name}' already exists in the directory.")
        else:
            print(f"Downloading '{model_name}'...")
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print(
                    f"'{model_name}' downloaded successfully to '{ai_model_dir}'.")
            except Exception as e:
                print(f"Failed to download '{model_name}': {e}")

    print("Finished Downloading")

# Pre-load small ai models for faster execution


def pre_load_ai_models():
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(DEVICE)

    # download all ai models
    download_ai_models()

    # mobile sam model
    MODEL_TYPE = "vit_t"
    mobile_sam = sam_model_registry[MODEL_TYPE](
        checkpoint=os.path.join(ai_model_dir, 'mobile_sam.pt'))
    mobile_sam.to(device=device)
    mobile_sam.eval()

    # nuclick classifier
    nuclick_class = DenseNet121(
        spatial_dims=2,
        in_channels=4,
        out_channels=4)
    checkpoint = torch.load(
        os.path.join(ai_model_dir, 'nuclick.pt'),
        map_location=torch.device(device))
    model_state_dict = checkpoint.get("nuclick", checkpoint)
    nuclick_class.load_state_dict(model_state_dict, strict=True)

    # nuclick segmentation
    nuclick_seg = BasicUNet(
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
        os.path.join(ai_model_dir, 'nuclickSegmentation.pt'),
        map_location=torch.device(device))
    model_state_dict = checkpoint.get("nuclickSegmentation", checkpoint)
    nuclick_seg.load_state_dict(model_state_dict, strict=True)

    #stardist ai model
    StarDist2D.from_pretrained()
    stardist_seg = StarDist2D.from_pretrained('2D_versatile_he')

    return mobile_sam, nuclick_class, nuclick_seg, stardist_seg
