from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from app.services.ml_interface import MLInterface
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import os
import json
from app.api import quiz

# APP SETUP
ml_interface = MLInterface()
app = FastAPI()
app.include_router(quiz.router, prefix="/quiz")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:19006"] for Expo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DIRECTORIES
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "assets/uploads")
DIST_DIR = os.path.join(ROOT_DIR, "dist")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# STATIC FILES
app.mount(
    "/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets"
)
app.mount(
    "/_expo", StaticFiles(directory=os.path.join(DIST_DIR, "_expo")), name="_expo"
)


@app.post("/camera")
async def make_horizon_prediction(request: Request, file: UploadFile = File(None)):
    print("Received request for horizon prediction")
    content_type = request.headers.get("content-type", "")

    try:
        # === MULTIPART MODE (MOBILE) ===
        if (
            "multipart/form-data" in content_type
            and file
            and isinstance(file.filename, str)
        ):
            contents = await file.read()
            img_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(img_path, "wb") as f:
                f.write(contents)

            # Try to extract form data (optional)
            form = await request.form()

            raw_depths = form.get("horizonDepths")
            raw_tabulars = form.get("horizonTabulars")
            try:
                horizon_depths = (
                    json.loads(raw_depths) if isinstance(raw_depths, str) else []
                )
                horizon_tabulars = (
                    json.loads(raw_tabulars) if isinstance(raw_tabulars, str) else []
                )
            except Exception:
                horizon_depths, horizon_tabulars = [], []

            pred_depths, pred_tabulars, pred_symbols = ml_interface.predict(
                img_path, horizon_depths, horizon_tabulars
            )

            return {
                "status": "ok",
                "horizonDepths": pred_depths,
                "horizonTabulars": pred_tabulars,
                "horizonSymbols": pred_symbols,
                "filename": file.filename,
                "source": "multipart",
            }

        # === JSON MODE (WEB) ===
        data = await request.json()

        base64_data = data.get("content", "").split(",")[-1]
        filename = data.get("filename", "image.png")
        if not base64_data or not isinstance(filename, str):
            raise ValueError("Missing or invalid base64 content/filename")

        img_path = os.path.join(UPLOAD_DIR, filename)
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(base64_data))

        horizon_depths = data.get("horizonDepths", [])
        horizon_tabulars = data.get("horizonTabulars", [])

        pred_depths, pred_tabulars, pred_symbols = ml_interface.predict(
            img_path, horizon_depths, horizon_tabulars
        )

        return {
            "status": "ok",
            "horizonDepths": pred_depths,
            "horizonTabulars": pred_tabulars,
            "horizonSymbols": pred_symbols,
            "filename": filename,
            "source": "base64",
        }

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# @app.post("/camera/sample")
# async def make_horizon_prediction_sample(request: Request):
#     print("Received request for horizon prediction")
#     content_type = request.headers.get("content-type", "")

#     # JSON base64 (Web)
#     try:
#         data = await request.json()

#         base64_data = data["content"].split(",")[-1]
#         filename = data.get("filename", "image.png")
#         img_path = os.path.join(SAMPLE_DIR, filename)

#         print("filename:", filename)
#         if not filename:
#             raise ValueError("Filename is required")
#         if not isinstance(filename, str):
#             raise ValueError("Filename must be a string")

#         with open(img_path, "wb") as f:
#             f.write(base64.b64decode(base64_data))

#         # Extract data
#         # horizon_depths = data["horizonDepths"]
#         # horizon_tabulars = data["horizonTabulars"]

#         pred_depths, pred_tabulars, pred_symbols = ml_interface.predict(img_path)#, horizon_depths, horizon_tabulars)
#         print(pred_depths)
#         print(pred_tabulars)
#         print(pred_symbols)

#         print(3)
#         return {
#             "status": "ok",
#             "horizonDepths": pred_depths,
#             "horizonTabulars": pred_tabulars,
#             "horizonSymbols": pred_symbols,
#             "filename": filename,
#             "source": "base64"
#         }
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"error": str(e)})


class HorizonFeedback(BaseModel):
    horizon_depths: list[float]
    horizon_label: str


@app.post("/camera/feedback")
def take_horizon_feedback(feedback: HorizonFeedback):
    """
    Endpoint to provide feedback on the predicted horizon.
    Accepts a list of horizon depths and a label.
    """
    # Here you would typically process the feedback, e.g., store it in a database
    return {
        "message": "Feedback received",
        "depths": feedback.horizon_depths,
        "label": feedback.horizon_label,
    }


# FRONT ROUTES
@app.get("/")
def serve_index():
    return FileResponse(os.path.join(DIST_DIR, "index.html"))


# @app.get("/{full_path:path}")
# def spa_router(full_path: str):
#     if full_path.startswith("camera") or full_path.startswith("quiz") or full_path.startswith("api"):
#         return JSONResponse(status_code=404, content={"error": "Not found"})
#     return FileResponse(os.path.join(DIST_DIR, "index.html"))
