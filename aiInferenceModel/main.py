from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List

app = FastAPI()

@app.get("/")
def read_root():
    return {"This code allows the AI model to be packed as an API"}

class NumpyArray(BaseModel):
    """
    Pydantic model for receiving a NumPy array as JSON input.

    Attributes:
        numpy_array (List[List[float]]): A 2D list representing the NumPy array.
    """
    numpy_array: List[List[float]]

@app.post("/process_tiles/")
async def process_data(input_data: NumpyArray):
    """
    Process a NumPy array received as JSON input.

    Args:
        input_data (NumpyArray): JSON input containing a NumPy array.

    Returns:
        List[List[float]]: Processed NumPy array.

    Raises:
        HTTPException: If there is an issue with the input data format.
    """
    try:
        # Access the JSON data as a list and convert it to a NumPy array
        json_data = input_data.numpy_array
        numpy_array = np.array(json_data, dtype=np.float64)
        
        # Example: Multiply each element of the NumPy array by 2
        processed_numpy_array = numpy_array * 2

        ############# Fit your tile processing model here ################
        # You can add your tile processing logic here.

        #################################################################
        
        return processed_numpy_array.tolist()
    
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid data")
