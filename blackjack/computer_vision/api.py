from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from pathlib import Path

import numpy as np
import cv2
import io
import os

app = FastAPI()

# Allow all requests (optional, good for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"status": "ok"}


@app.post("/upload_image")
async def receive_image(img: UploadFile = File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # type(cv2_img) => numpy.ndarray

    # Image directory (TODO add to params)
    directory = os.path.joinpath("blackjack", "computer_vision", "temp_image")
    # Temporary image file name
    filename = "input.png"

    cv2.imwrite(
        directory,
    )  # TODO temporarly save image

    ### Do cool stuff with your image.... For example face detection
    annotated_img = annotate_face()  # TODO call roboflow_predictions function

    ### Encoding and responding with the image
    im = cv2.imencode(".png", annotated_img)[
        1
    ]  # extension depends on which format is sent from Streamlit

    remove  # TODO remove temp image
    return Response(content=im.tobytes(), media_type="image/png")

if __name__ == '__main__':

    try:

    except:
        import ipdb, traceback, sys
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
