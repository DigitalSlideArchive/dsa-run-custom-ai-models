## DSA AI Adapter Developer Documentation

### Recommended Method to Build DSA AI Adapter

1. Install DSA AI Adapter by following the instructions [here](https://github.com/DigitalSlideArchive/dsa-run-custom-ai-models#step-1---install-dsa-adapater-for-ai-models).
2. Deploy the model using conda as explained [here](https://github.com/DigitalSlideArchive/dsa-run-custom-ai-models#method-2---using-conda).

Upon completion, Fast API will be initialized, and a URL for the hosted AI models will be generated.

#### Structure of the Fast API Root

Developers can access the AI model location by navigating to:

*Location-of-DSA-ai-adapter/aiInferenceModel/*

The code is organized as follows:

- **main.py**: Executes the Fast API code, creating endpoints for each AI model.
- **utils.py**: Contains instructions for downloading and pre-loading AI models.
- **ai_models folder**: Houses individual code sets for pre-built AI models.

##### Segmentation Model

Developers can use the following code as a template for building custom segmentation models. After launching the Fast API, modifications to the code will be automatically reflected at the endpoint:

- [nuclickSegmentation.py](https://github.com/DigitalSlideArchive/dsa-run-custom-ai-models/blob/master/aiInferenceModel/ai_models/nuclickSegmentation.py)
- [samOnclick.py](https://github.com/DigitalSlideArchive/dsa-run-custom-ai-models/blob/master/aiInferenceModel/ai_models/samOnclick.py)
- [StardistSegmentation.py](https://github.com/DigitalSlideArchive/dsa-run-custom-ai-models/blob/master/aiInferenceModel/ai_models/stardistSegmentation.py)

##### Classification Model

The following code serves as a template for building custom classification models:

- [nuclickClassification.py](https://github.com/DigitalSlideArchive/dsa-run-custom-ai-models/blob/master/aiInferenceModel/ai_models/nuclickClassification.py)