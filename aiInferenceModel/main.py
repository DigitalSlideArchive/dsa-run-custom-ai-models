from fastapi import FastAPI, HTTPException, Request

# Import AI model modules
import nuclickClassification as classificationModel
import nuclickSegmentation as segmentationModel

# You can import your custom AI models into this code
# Example: import customSegmentationModel as segmentationModel2
# Then uncomment the Endpoint for custom AI models

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "DSA AI adapter for deploying AI models"}

# Endpoint for Nuclick AI model classification
@app.post("/nuclick_classification/")
async def process_ima(request: Request):
    try:
        json_data = await request.json()
        network_output = classificationModel.run_ai_model_inferencing(
            json_data)
        return {"network_output": network_output}

    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid data")

# Endpoint for Nuclick AI model segmentation
@app.post("/nuclick_segmentation/")
async def process_ima(request: Request):
    try:
        json_data = await request.json()
        network_output = segmentationModel.run_ai_model_inferencing(json_data)
        return {"network_output": network_output}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail="Invalid data")

# Endpoint for custom AI model
"""""
@app.post("/custom_ai_model/")
async def process_ima(request: Request):
    try:
        # Parse JSON data from the request
        json_data = await request.json()

        # Run segmentation model inference
        network_output = segmentationModel2.run_ai_model_inferencing(json_data)

        # Return the model's output
        return {"network_output": network_output}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail="Invalid data")
"""""

# Run the FastAPI application using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
