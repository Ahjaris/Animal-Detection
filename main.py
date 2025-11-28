from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# ==== Serve Frontend ====
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# ==== Load Model ====
model = YOLO("model/best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))

    results = model(img)[0]

    # Ambil bounding box + kelas
    if len(results.boxes) == 0:
        return {"prediction": "Tidak terdeteksi hewan", "box": None}

    box = results.boxes[0]
    cls_id = int(box.cls)
    label = model.names[cls_id]

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    return {
        "prediction": label,
        "box": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
    }
