# # video
#
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from pydantic import BaseModel
# from typing import List
# import uvicorn
# from PIL import Image
# import io
# import numpy as np
# import tensorflow as tf
# import logging
# import json
#
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Load the TFLite model
# model_path = "C:\\Users\\ergou\\PycharmProjects\\pythonProject\\violence_nude_hate_api\\model\\yolo_voilence_float32.tflite"
# try:
#     interpreter = tf.lite.Interpreter(model_path=model_path)
#     interpreter.allocate_tensors()
# except Exception as e:
#     logger.error(f"Failed to load TFLite model: {e}")
#     raise RuntimeError(f"Failed to load TFLite model: {e}")
#
# # Get input and output details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#
# # Define the correct image size based on the model's expected input
# IMAGE_SIZE = (128, 128)  # Update to the correct size for your model
#
# # Initialize the FastAPI app
# app = FastAPI()
#
# class Prediction(BaseModel):
#     class_name: str
#     confidence: float
#
# async def process_image(image_data):
#     logger.info("Processing image...")
#     image = Image.open(io.BytesIO(image_data)).convert("RGB")
#     logger.info(f"Image size: {image.size}")
#     image = image.resize(IMAGE_SIZE)
#     logger.info(f"Resized image size: {image.size}")
#     image_array = np.array(image)
#     logger.info(f"Image array shape: {image_array.shape}")
#     image_array = image_array / 255.0
#     logger.info(f"Normalized image array shape: {image_array.shape}")
#     image_array = np.expand_dims(image_array, axis=0)
#     logger.info(f"Expanded image array shape: {image_array.shape}")
#     # Cast to FLOAT32
#     image_array = image_array.astype(np.float32)
#     interpreter.set_tensor(input_details[0]['index'], image_array)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     logger.info(f"Model output shape: {output_data.shape}")
#     confidence_scores = tf.nn.softmax(output_data[0])
#     logger.info(f"Confidence scores: {confidence_scores}")
#     class_names = ['non violence', 'violence']  # Replace with your actual class names
#     results = [Prediction(class_name=class_names[idx], confidence=float(confidence_scores[idx])) for idx in range(len(class_names))]
#     logger.info(f"Results: {results}")
#     return results
#
# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             data = await websocket.receive()
#             if data["type"] == "websocket.receive":
#                 if "bytes" in data:
#                     try:
#                         image_data = data["bytes"]
#                         results = await process_image(image_data)
#                         await websocket.send_text(json.dumps([result.dict() for result in results]))
#                     except Exception as e:
#                         logger.error(f"Error processing message: {e}")
#                         await websocket.send_text("Error processing message")
#                 else:
#                     logger.error("Invalid message format")
#                     await websocket.send_text("Invalid message format")
#     except WebSocketDisconnect:
#         logger.info(f"Client disconnected: {websocket.client}")
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         await websocket.send_text("Unexpected error")
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# image

# from fastapi import FastAPI, File, UploadFile
# from pydantic import BaseModel
# import uvicorn
# from PIL import Image
# import io
# import numpy as np
# import onnxruntime as ort
# import logging
# import json
#
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Load the ONNX model
# model_path = "C:\\Users\\ergou\\PycharmProjects\\pythonProject\\violence_nude_hate_api\\model\\yolo_voilence.onnx"
# try:
#     session = ort.InferenceSession(model_path)
#     logger.info("ONNX model loaded successfully.")
# except Exception as e:
#     logger.error(f"Failed to load ONNX model: {e}")
#     raise RuntimeError(f"Failed to load ONNX model: {e}")
#
# # Get input and output details
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name
#
# # Define the correct image size based on the model's expected input
# IMAGE_SIZE = (128, 128)  # Update to the correct size for your model
#
# # Initialize the FastAPI app
# app = FastAPI()
#
#
# class Prediction(BaseModel):
#     class_name: str
#     confidence: float
#
#
# def process_image(image_data):
#     try:
#         logger.info("Processing image...")
#         image = Image.open(io.BytesIO(image_data)).convert("RGB")
#         logger.info(f"Image size: {image.size}")
#         image = image.resize(IMAGE_SIZE)
#         logger.info(f"Resized image size: {image.size}")
#         image_array = np.array(image)
#         logger.info(f"Image array shape: {image_array.shape}")
#         image_array = image_array / 255.0
#         logger.info(f"Normalized image array shape: {image_array.shape}")
#         image_array = np.expand_dims(image_array, axis=0)
#         logger.info(f"Expanded image array shape: {image_array.shape}")
#         # Cast to FLOAT32
#         image_array = image_array.astype(np.float32)
#
#         # Run inference
#         outputs = session.run([output_name], {input_name: image_array})
#         output_data = outputs[0][0]  # Assuming batch size of 1
#
#         logger.info(f"Model output shape: {output_data.shape}")
#         confidence_scores = np.exp(output_data) / np.sum(np.exp(output_data))
#         logger.info(f"Confidence scores: {confidence_scores}")
#         class_names = ['class_0', 'class_1']  # Replace with your actual class names
#         results = [Prediction(class_name=class_names[idx], confidence=float(confidence_scores[idx])) for idx in
#                    range(len(class_names))]
#         logger.info(f"Results: {results}")
#         return results
#     except Exception as e:
#         logger.error(f"Error processing image: {e}")
#         return {"error": f"Error processing image: {e}"}
#
#
# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Log the content type
#         logger.info(f"Uploaded file content type: {file.content_type}")
#
#         # Ensure the uploaded file is an image
#         if file.content_type is None or not file.content_type.startswith("image/"):
#             return {"error": "Uploaded file is not an image."}
#
#         image_data = await file.read()
#         results = process_image(image_data)
#         if isinstance(results, dict) and "error" in results:
#             return results
#         return json.dumps([result.dict() for result in results])
#     except Exception as e:
#         logger.error(f"Error processing image: {e}")
#         return {"error": f"Error processing image: {e}"}
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
## working
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from pydantic import BaseModel
# import uvicorn
# from PIL import Image
# import io
# import numpy as np
# import onnxruntime as ort
# import logging
# import json
# import base64
# import cv2
#
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Load the ONNX model
# model_path = "C:\\Users\\ergou\\PycharmProjects\\pythonProject\\violence_nude_hate_api\\model\\yolo_voilence.onnx"
# try:
#     session = ort.InferenceSession(model_path)
#     logger.info("ONNX model loaded successfully.")
# except Exception as e:
#     logger.error(f"Failed to load ONNX model: {e}")
#     raise RuntimeError(f"Failed to load ONNX model: {e}")
#
# # Get input and output details
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name
#
# # Define the correct image size based on the model's expected input
# IMAGE_SIZE = (128, 128)  # Update to the correct size for your model
#
# # Initialize the FastAPI app
# app = FastAPI()
#
#
# class Prediction(BaseModel):
#     class_name: str
#     confidence: float
#
#
# def process_frame(frame):
#     try:
#         logger.info("Processing frame...")
#         image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         image = image.resize(IMAGE_SIZE)
#         image_array = np.array(image, dtype=np.float32) / 255.0
#
#         # Ensure the image has the right shape (batch_size, channels, height, width)
#         if image_array.shape[-1] == 3:  # Convert to (channels, height, width) if needed
#             image_array = np.transpose(image_array, (2, 0, 1))
#
#         image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
#
#         # Run inference
#         outputs = session.run([output_name], {input_name: image_array})
#         output_data = outputs[0][0]  # Assuming batch size of 1
#
#         confidence_scores = np.exp(output_data) / np.sum(np.exp(output_data))
#         class_names = ['class_0', 'class_1']  # Replace with your actual class names
#         results = [Prediction(class_name=class_names[idx], confidence=float(confidence_scores[idx])) for idx in
#                    range(len(class_names))]
#         return results
#     except Exception as e:
#         logger.error(f"Error processing frame: {e}")
#         return {"error": f"Error processing frame: {e}"}
#
#
# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             data = await websocket.receive_text()
#             if data:
#                 try:
#                     # Decode base64 to bytes
#                     frame_bytes = base64.b64decode(data)
#
#                     # Convert bytes to numpy array
#                     np_arr = np.frombuffer(frame_bytes, np.uint8)
#                     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#
#                     if frame is None:
#                         await websocket.send_text(json.dumps({"error": "Failed to decode frame."}))
#                         continue
#
#                     results = process_frame(frame)
#                     if isinstance(results, dict) and "error" in results:
#                         await websocket.send_text(json.dumps(results))
#                     else:
#                         await websocket.send_text(json.dumps([result.dict() for result in results]))
#
#                 except Exception as e:
#                     logger.error(f"Error processing frame: {e}")
#                     await websocket.send_text(json.dumps({"error": f"Error processing frame: {e}"}))
#     except WebSocketDisconnect:
#         logger.info(f"Client disconnected.")
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         await websocket.send_text(json.dumps({"error": f"Unexpected error: {e}"}))
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn
from PIL import Image
import numpy as np
import onnxruntime as ort
import logging
import json
import base64
import cv2
import os  # Import os for file operations
from datetime import datetime  # Import datetime for unique filenames

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the ONNX model
model_path = "C:\\Users\\ergou\\PycharmProjects\\pythonProject\\violence_nude_hate_api\\model\\yolo_voilence.onnx"
try:
    session = ort.InferenceSession(model_path)
    logger.info("ONNX model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load ONNX model: {e}")
    raise RuntimeError(f"Failed to load ONNX model: {e}")

# Get input and output details
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Define the correct image size based on the model's expected input
IMAGE_SIZE = (128, 128)  # Update to the correct size for your model

# Initialize the FastAPI app
app = FastAPI()

# Create an output directory if it doesn't exist
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

class Prediction(BaseModel):
    class_name: str
    confidence: float

def process_frame(frame):
    try:
        logger.info("Processing frame...")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Ensure the image has the right shape (batch_size, channels, height, width)
        if image_array.shape[-1] == 3:  # Convert to (channels, height, width) if needed
            image_array = np.transpose(image_array, (2, 0, 1))

        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Run inference
        outputs = session.run([output_name], {input_name: image_array})
        output_data = outputs[0][0]  # Assuming batch size of 1

        confidence_scores = np.exp(output_data) / np.sum(np.exp(output_data))
        class_names = ['non_violence', 'violence']  # Replace with your actual class names
        results = [Prediction(class_name=class_names[idx], confidence=float(confidence_scores[idx])) for idx in
                   range(len(class_names))]

        # Determine the class with the highest confidence
        best_result = max(results, key=lambda x: x.confidence)
        class_name = best_result.class_name
        confidence = best_result.confidence

        # # Save the frame with class and confidence in the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Unique filename with timestamp
        output_path = os.path.join(output_dir, f"frame_{timestamp}_{class_name}_{confidence:.2f}.jpg")
        cv2.imwrite(output_path, frame)
        logger.info(f"Frame saved to {output_path}")

        return results
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {"error": f"Error processing frame: {e}"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if data:
                try:
                    # Decode base64 to bytes
                    frame_bytes = base64.b64decode(data)

                    # Convert bytes to numpy array
                    np_arr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if frame is None:
                        await websocket.send_text(json.dumps({"error": "Failed to decode frame."}))
                        continue

                    results = process_frame(frame)
                    if isinstance(results, dict) and "error" in results:
                        await websocket.send_text(json.dumps(results))
                    else:
                        await websocket.send_text(json.dumps([result.dict() for result in results]))

                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    await websocket.send_text(json.dumps({"error": f"Error processing frame: {e}"}))
    except WebSocketDisconnect:
        logger.info(f"Client disconnected.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await websocket.send_text(json.dumps({"error": f"Unexpected error: {e}"}))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
