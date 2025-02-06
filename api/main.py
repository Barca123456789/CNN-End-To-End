from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn  # Import uvicorn here

app = FastAPI()

# Load trained model
model = tf.keras.models.load_model("new_model/cnn_model.h5")

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

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
    
    # Apply the custom threshold logic for the label
    if prediction >= 0.6:
        predicted_label = "Sad"
    elif prediction <= 0.4:
        predicted_label = "Happy"
    else:
        predicted_label = "Uncertain"
    
    return {"prediction": float(prediction), "label": predicted_label}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)



