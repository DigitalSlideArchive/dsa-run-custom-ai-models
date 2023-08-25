from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"This code allows the AI model to be packed as an API"}

class InputData(BaseModel):
    data: dict

@app.post("/aimodel/")
async def process_data(input_data: InputData):
    # You can process the input data here
    processed_data = input_data.data
    return processed_data