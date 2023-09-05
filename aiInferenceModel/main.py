from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np
from typing import List
import asyncio
import nuclickAIModel as aimodel

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "This code allows the AI model to be packed as an API"}
    
@app.post("/process_img_msk_anot/")
async def process_ima(request: Request):
    try:
        json_data = await request.json()
        network_output = aimodel.run_ai_model_inferencing(json_data)
        return {"nuclei":network_output}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

