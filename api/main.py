from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
from fastapi.middleware.cors import CORSMiddleware  # Add CORS

app = FastAPI()

# Enable CORS for all domains (Required for Render Deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = tf.keras.models.load_model("new_model/cnn_model.h5")  # Ensure this file exists in Render

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()  # Read the uploaded image file
    image = preprocess_image(image_bytes)  # Preprocess the image
    prediction = model.predict(image)[0][0]  # Get the prediction from the model
    
    # Apply threshold logic
    if prediction >= 0.6:
        predicted_label = "Sad"
    elif prediction <= 0.4:
        predicted_label = "Happy"
    else:
        predicted_label = "Uncertain"
    
    return {"prediction": float(prediction), "label": predicted_label}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)  # Update host and port



