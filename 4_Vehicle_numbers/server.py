from fastapi import FastAPI, File, HTTPException
from PIL import Image
from io import BytesIO
import numpy as np
import logging
import requests
import sys
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from inference import (
    process_plate_images,
    PlateRecognition,
    LetterRecognition,
)
from cv2 import cvtColor, imwrite, COLOR_RGB2BGR
from fastapi.middleware.cors import CORSMiddleware

sys.setrecursionlimit(1500)
logging.basicConfig(
    filename="server.log",
    level=logging.DEBUG,
    filemode="a",
    format="%(asctime)s: %(funcName)s - %(levelname)s - %(message)s",
)

app = FastAPI()
app.mount("/_static", StaticFiles(directory="_static"), name="_static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

plate_detector = PlateRecognition()
letter_detector = LetterRecognition()


@app.get("/")
async def main():
    """
    ping method for api call
    Returns:
        200
    """
    return


@app.post("/CV_project/license_plates")
async def inference_demo(file: bytes = File(...),
    url: str = "https://nsa39.casimages.com/img/2018/09/12/180912122808840707.jpg",
    json: bool = False,
):
    """
    Public endpoint for demo fashion analysis by GET request

    Args:
        url: image url

    Returns:
        jpg image with recognized vehicle items
    """
    if file is not None:
        image = Image.open(BytesIO(file)).convert("RGB")
    else:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    np_image = np.array(image, dtype=np.uint8)
    result, result_image = process_plate_images(plate_detector, letter_detector, np_image)
    if json:
        return result
    if result_image is not None:
        # result_image = visualize_plate_results(image, result)
        imwrite("result.jpg", cvtColor(result_image, COLOR_RGB2BGR))
        return FileResponse("result.jpg")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8050, log_level="info")
