from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List
import asyncio

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "This code allows the AI model to be packed as an API"}

class NumpyArray(BaseModel):
    numpy_array: List[List[List[int]]]

@app.post("/process_tiles/")
async def process_data(input_data: NumpyArray):
    try:
        json_data = input_data.numpy_array
        numpy_array = np.array(json_data)

        # Example: Multiply each element of the NumPy array by 2
        processed_numpy_array = numpy_array * 2

        ############# Fit your tile processing model here ################
        # You can add your tile processing logic here.
        #################################################################

        return processed_numpy_array.tolist()

    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid data")

@app.post("/process_tiles/concurrent/")
async def process_multiple_data(input_data_list: List[NumpyArray]):
    """
    Process multiple NumPy arrays received as JSON input concurrently.

    Args:
        input_data_list (List[NumpyArray]): List of JSON inputs containing NumPy arrays.

    Returns:
        List[List[float]]: List of processed NumPy arrays.

    Raises:
        HTTPException: If there is an issue with the input data format.
    """
    try:
        async def process_single_data(input_data: NumpyArray):
            # Similar processing logic as the previous endpoint
            json_data = input_data.numpy_array
            numpy_array = np.array(json_data)
            processed_numpy_array = numpy_array * 2
            return processed_numpy_array.tolist()

        # Use asyncio.gather to concurrently process multiple input_data items
        results = await asyncio.gather(*(process_single_data(data) for data in input_data_list))
        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

