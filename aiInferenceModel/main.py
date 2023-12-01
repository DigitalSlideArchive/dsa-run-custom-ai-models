# Import AI model modules
from ai_models import (nuclickClassification, nuclickSegmentation, samMobile,
                       samOnclick, samSegmentation, stardistSegmentation, CustomAIModel)
from fastapi import FastAPI, HTTPException, Request
from utils import pre_load_ai_models

# You can import your custom AI models into this code
# Example: import customSegmentationModel
# Use the Endpoint for custom AI models

app = FastAPI()

# Pre-load ai models for faster execution
mobile_sam, nuclick_class, nuclick_seg, stardist_seg = pre_load_ai_models()

# Request count
request_count = 0


@app.get("/")
def read_root():
    return {"message": "DSA AI adapter for deploying AI models"}

# Endpoint for Nuclick AI model classification


@app.post("/nuclick_classification/")
async def process_ima(request: Request):
    global request_count
    print(f"Request number : {request_count}")
    request_count += 1
    try:
        json_data = await request.json()
        network_output = nuclickClassification.run_ai_model_inferencing(
            json_data, nuclick_class)
        return {"network_output": network_output}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail="Invalid data")

# Endpoint for Nuclick AI model segmentation


@app.post("/nuclick_segmentation/")
async def process_ima(request: Request):
    global request_count
    print(f"Request number : {request_count}")
    request_count += 1
    try:
        json_data = await request.json()
        network_output = nuclickSegmentation.run_ai_model_inferencing(
            json_data, nuclick_seg)
        return {"network_output": network_output}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail="Invalid data")

# Endpoint for Segment anything AI model segmentation


@app.post("/segment_anything/")
async def process_ima(request: Request):
    global request_count
    print(f"Request number : {request_count}")
    request_count += 1
    try:
        json_data = await request.json()
        network_output = samSegmentation.run_ai_model_inferencing(json_data)
        return {"network_output": network_output}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail="Invalid data")

# Endpoint for Segment anything AI model with user input


@app.post("/segment_anything_onclick/")
async def process_ima(request: Request):
    global request_count
    print(f"Request number : {request_count}")
    request_count += 1
    try:
        json_data = await request.json()
        network_output = samOnclick.run_ai_model_inferencing(json_data)
        return {"network_output": network_output}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail="Invalid data")

# Endpoint for Mobile segment anything model


@app.post("/segment_anything_mobile/")
async def process_ima(request: Request):
    global request_count
    print(f"Request number : {request_count}")
    request_count += 1
    try:
        json_data = await request.json()
        network_output = samMobile.run_ai_model_inferencing(
            json_data, mobile_sam)

        return {"network_output": network_output}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail="Invalid data")

# Endpoint for Stardist segmentation


@app.post("/stardist_h_and_e/")
async def process_ima(request: Request):
    global request_count
    print(f"Request number : {request_count}")
    request_count += 1
    try:
        json_data = await request.json()
        network_output = stardistSegmentation.run_ai_model_inferencing(
            json_data, stardist_seg)

        return {"network_output": network_output}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail="Invalid data")

# Endpoint for custom AI model


@app.post("/custom_ai_model/")
async def process_ima(request: Request):
    try:
        # Parse JSON data from the request
        json_data = await request.json()

        # Run segmentation model inference
        network_output = CustomAIModel.run_ai_model_inferencing(json_data)

        # Return the model's output
        return {"network_output": network_output}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail="Invalid data")

# Run the FastAPI application using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
