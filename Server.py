from fastapi import BackgroundTasks, FastAPI, UploadFile, File, Query, WebSocket
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import cv2
import tempfile
import os
import logging

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

from ultralytics import YOLO

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pothole-api")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_VALUE = os.getenv("MODEL_PATH", "best.pt")
MODEL_PATH = (
    MODEL_PATH_VALUE
    if os.path.isabs(MODEL_PATH_VALUE)
    else os.path.join(BASE_DIR, MODEL_PATH_VALUE)
)
model = None


def get_model():
    global model
    if model is None:
        logger.info("Loading YOLO model from %s", MODEL_PATH)
        model = YOLO(MODEL_PATH, task="detect")
        logger.info("YOLO model loaded")
    return model


def cleanup_file(path: str):
    if os.path.exists(path):
        os.unlink(path)


@app.get("/")
async def health_check():
    logger.info("Health check requested")
    return {
        "status": "ok",
        "message": "Pothole detection API is running",
        "model_path": MODEL_PATH,
        "model_loaded": model is not None
    }

# -----------------------------------
# IMAGE ENDPOINT
# -----------------------------------
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    logger.info("Image prediction requested: %s", file.filename)
    yolo_model = get_model()

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Resize for faster inference (reduce from 448x640 to 320x320)
    img_resized = cv2.resize(img, (320, 320))

    results = yolo_model.track(
        img_resized,
        conf=0.05,
        persist=True,
        imgsz=320
    )

    detections = []

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            track_id = (
                int(box.id[0])
                if box.id is not None
                else -1
            )

            x1,y1,x2,y2 = box.xyxy[0].tolist()

            detections.append({

                "id": track_id,

                "class": yolo_model.names[cls],

                "confidence": round(conf,2),

                "bbox":[
                    round(x1,2),
                    round(y1,2),
                    round(x2,2),
                    round(y2,2)
                ]
            })

    logger.info("Image prediction completed: %s detections", len(detections))

    return JSONResponse({
        "status":"success",
        "detections": detections
    })


# -----------------------------------
# VIDEO ENDPOINT (MP4)
# -----------------------------------
@app.post("/predict_video")
async def predict_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    download: bool = Query(False)
):
    logger.info("Video prediction requested: %s download=%s", file.filename, download)
    yolo_model = get_model()

    temp_input = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_input.write(await file.read())
    temp_input.close()

    cap = cv2.VideoCapture(temp_input.name)

    original_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    target_fps = 15

    frame_skip = max(
        1,
        int(original_fps / target_fps)
    ) if original_fps > target_fps else 1

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    output_path = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ).name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        target_fps,
        (width,height)
    )

    detections = []

    potholes = set()
    cracks = set()
    manholes = set()

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        results = yolo_model.track(
            frame,
            conf=0.05,
            persist=True
        )

        for r in results:
            for box in r.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                track_id = (
                    int(box.id[0])
                    if box.id is not None
                    else -1
                )

                label = yolo_model.names[cls]

                if label == "pothole":
                    potholes.add(track_id)
                elif label == "crack":
                    cracks.add(track_id)
                elif label == "manhole":
                    manholes.add(track_id)

                x1,y1,x2,y2 = box.xyxy[0].tolist()

                detections.append({

                    "frame": frame_count,

                    "id": track_id,

                    "class": label,

                    "confidence": round(conf,2),

                    "bbox":[
                        round(x1,2),
                        round(y1,2),
                        round(x2,2),
                        round(y2,2)
                    ]
                })

        annotated = results[0].plot()
        out.write(annotated)

        frame_count += 1

    cap.release()
    out.release()

    os.unlink(temp_input.name)

    logger.info(
        "Video prediction completed: %s detections, potholes=%s, cracks=%s, manholes=%s",
        len(detections),
        len(potholes),
        len(cracks),
        len(manholes)
    )

    if download:
        background_tasks.add_task(cleanup_file, output_path)
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="prediction.mp4",
            headers={
                "X-Processed-FPS": str(target_fps),
                "X-Total-Detections": str(len(detections)),
                "X-Potholes": str(len(potholes)),
                "X-Cracks": str(len(cracks)),
                "X-Manholes": str(len(manholes))
            }
        )

    return JSONResponse({

        "status":"success",

        "prediction_video": output_path,

        "processed_fps": target_fps,

        "counts":{
            "potholes": len(potholes),
            "cracks": len(cracks),
            "manholes": len(manholes)
        },

        "total_detections": len(detections),

        "detections": detections
    })


# -----------------------------------
# LIVE STREAM WEBSOCKET
# -----------------------------------
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):

    await websocket.accept()
    yolo_model = get_model()

    logger.info("Live client connected")

    try:

        while True:

            # receive JPEG frame bytes
            data = await websocket.receive_bytes()

            nparr = np.frombuffer(
                data,
                np.uint8
            )

            frame = cv2.imdecode(
                nparr,
                cv2.IMREAD_COLOR
            )

            if frame is None:
                continue

            results = yolo_model.track(
                frame,
                conf=0.05,
                persist=True
            )

            detections = []

            for r in results:
                for box in r.boxes:

                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    track_id = (
                        int(box.id[0])
                        if box.id is not None
                        else -1
                    )

                    x1,y1,x2,y2 = box.xyxy[0].tolist()

                    detections.append({

                        "id": track_id,

                        "class": yolo_model.names[cls],

                        "confidence": round(conf,2),

                        "bbox":[
                            round(x1,2),
                            round(y1,2),
                            round(x2,2),
                            round(y2,2)
                        ]
                    })

            await websocket.send_json({

                "status":"success",

                "detections": detections
            })

    except Exception as e:

        logger.info("Live client disconnected: %s", e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
