from mobile_sam import sam_model_registry
from monai.networks.nets import DenseNet121
from monai.networks.nets import BasicUNet
import torch
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_TYPE = "vit_t"
mobile_sam_weights = "../debug/weights/mobile_sam.pt"
nuclick_class_weights = "../debug/weights/nuclick.pt"
nuclick_seg_weights = "../debug/weights/nuclickSegmentation.pt"

device = torch.device(DEVICE)

# Pre-load small ai models for faster execution
def pre_load_ai_models():
    # mobile sam model
    mobile_sam = sam_model_registry[MODEL_TYPE](checkpoint=mobile_sam_weights)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    
    # nuclick classifier
    nuclick_class = DenseNet121(
        spatial_dims=2,
        in_channels=4,
        out_channels=4)
    checkpoint = torch.load(
        nuclick_class_weights,
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
        nuclick_seg_weights,
        map_location=torch.device(device))
    model_state_dict = checkpoint.get("nuclickSegmentation", checkpoint)
    nuclick_seg.load_state_dict(model_state_dict, strict=True)


    return mobile_sam, nuclick_class, nuclick_seg