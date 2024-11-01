# # from fastapi import FastAPI, UploadFile, File, HTTPException
# # from fastapi.responses import JSONResponse, HTMLResponse
# # import tensorflow as tf
# # import tempfile
# # import numpy as np
# # import cv2

# # # Load your model here. Adjust the path and loading method based on your model type.
# # def load_model():
# #     model_path = "model_utils.h5"  # for TensorFlow/Keras
# #     model = tf.keras.models.load_model(model_path)
# #     return model

# # app = FastAPI()

# # # Initialize your model (adjust as needed)
# # model = load_model()
# # class_names = ['compressed', 'extended']

# # def classify_frame(frame, model):
# #     """
# #     Classify the given frame using the custom model.
# #     :param frame: Preprocessed frame.
# #     :param model: Custom model.
# #     :return: Tuple of (class_name, probability)
# #     """
# #     preds = model.predict(frame)

# #     # Assuming binary classification (A vs. B), where model output is a single probability
# #     prob_B = preds[0][0]      # Probability of class B ('extended')
# #     prob_A = 1 - prob_B        # Probability of class A ('compressed')
    
# #     # Determine the class with the highest probability
# #     if prob_A > prob_B:
# #         class_name = class_names[0]  # 'compressed'
# #         probability = prob_A
# #     else:
# #         class_name = class_names[1]  # 'extended'
# #         probability = prob_B
    
# #     return class_name, probability

# # @app.get("/", response_class=HTMLResponse)
# # async def root():
# #     html_content = """
# #     <!DOCTYPE html>
# #     <html lang="en">
# #     <head>
# #         <meta charset="UTF-8">
# #         <meta name="viewport" content="width=device-width, initial-scale=1.0">
# #         <title>Upload Video for Prediction</title>
# #     </head>
# #     <body>
# #         <h1>Upload a Video for Prediction</h1>
# #         <form action="/predict" method="post" enctype="multipart/form-data">
# #             <input type="file" name="file" accept="video/*" required>
# #             <button type="submit">Upload and Predict</button>
# #         </form>
# #     </body>
# #     </html>
# #     """
# #     return HTMLResponse(content=html_content)

# # @app.post("/predict")
# # async def predict_video(file: UploadFile = File(...)):
# #     if file.content_type not in ["video/mp4", "video/avi", "video/mov"]:
# #         raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

# #     with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_file:
# #         temp_file.write(await file.read())
# #         temp_file.flush()

# #         try:
# #             video_capture = cv2.VideoCapture(temp_file.name)
# #             frames = []
# #             while True:
# #                 ret, frame = video_capture.read()
# #                 if not ret:
# #                     break
# #                 frame = cv2.resize(frame, (224, 224))  # Adjust as needed
# #                 frames.append(frame)
            
# #             video_capture.release()

# #             frames_array = np.array(frames) / 255.0  # Normalizing frames to match model input preprocessing
# #             if len(frames_array.shape) == 3:
# #                 frames_array = np.expand_dims(frames_array, axis=0)

# #             # Predict class and probability for each frame
# #             predictions = [classify_frame(np.expand_dims(f, axis=0), model) for f in frames_array]

# #             # Aggregate the predictions using majority voting for the class
# #             class_votes = {}
# #             probability_sum = {}

# #             for class_name, prob in predictions:
# #                 class_votes[class_name] = class_votes.get(class_name, 0) + 1
# #                 probability_sum[class_name] = probability_sum.get(class_name, 0) + prob

# #             # Determine the final class by majority vote and calculate average probability for that class
# #             final_class = max(class_votes, key=class_votes.get)
# #             average_probability = probability_sum[final_class] / class_votes[final_class]

# #             response = {
# #                 "prediction": final_class,
# #                 "probability": average_probability*100
# #             }

# #         except Exception as e:
# #             raise HTTPException(status_code=500, detail=str(e))
        
# #     return JSONResponse(content=response)





# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse, HTMLResponse
# import tensorflow as tf
# import tempfile
# import numpy as np
# import cv2
# import logging

# logging.basicConfig(level=logging.INFO)

# # Load your model here. Adjust the path and loading method based on your model type.
# def load_model():
#     model_path = "model_utils.h5"  # for TensorFlow/Keras
#     model = tf.keras.models.load_model(model_path)
#     return model

# app = FastAPI()

# # Allow CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Change this to specific origins in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # Initialize your model (adjust as needed)
# model = load_model()
# class_names = ['compressed', 'extended']

# def classify_frame(frame, model):
#     """
#     Classify the given frame using the custom model.
#     :param frame: Preprocessed frame.
#     :param model: Custom model.
#     :return: Tuple of (class_name, probability)
#     """
#     preds = model.predict(frame)

#     # Assuming binary classification (A vs. B), where model output is a single probability
#     prob_B = preds[0][0]      # Probability of class B ('extended')
#     prob_A = 1 - prob_B        # Probability of class A ('compressed')
    
#     # Determine the class with the highest probability
#     if prob_A > prob_B:
#         class_name = class_names[0]  # 'compressed'
#         probability = prob_A
#     else:
#         class_name = class_names[1]  # 'extended'
#         probability = prob_B
    
#     return class_name, probability

# @app.get("/", response_class=HTMLResponse)
# async def root():
#     html_content = """
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <meta name="viewport" content="width=device-width, initial-scale=1.0">
#         <title>Upload Video for Prediction</title>
#     </head>
#     <body>
#         <h1>Upload a Video for Prediction</h1>
#         <form action="/predict" method="post" enctype="multipart/form-data">
#             <input type="file" name="file" accept="video/*" required>
#             <button type="submit">Upload and Predict</button>
#         </form>
#     </body>
#     </html>
#     """

#     return "Hello world"
#     return HTMLResponse(content=html_content)

# @app.post("/predict")
# async def predict_video(file: UploadFile = File(...)):
#     if file.content_type not in ["video/mp4", "video/avi", "video/mov"]:
#         raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

#     with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_file:
#         temp_file.write(await file.read())
#         temp_file.flush()

#         try:
#             video_capture = cv2.VideoCapture(temp_file.name)
#             frames = []
#             while True:
#                 ret, frame = video_capture.read()
#                 if not ret:
#                     break
#                 frame = cv2.resize(frame, (224, 224))  # Adjust as needed
#                 frames.append(frame)
            
#             video_capture.release()

#             frames_array = np.array(frames) / 255.0  # Normalizing frames to match model input preprocessing
#             if len(frames_array.shape) == 3:
#                 frames_array = np.expand_dims(frames_array, axis=0)

#             # Predict class and probability for each frame
#             predictions = [classify_frame(np.expand_dims(f, axis=0), model) for f in frames_array]

#             # Aggregate the predictions using majority voting for the class
#             class_votes = {}
#             probability_sum = {}

#             for class_name, prob in predictions:
#                 class_votes[class_name] = class_votes.get(class_name, 0) + 1
#                 probability_sum[class_name] = probability_sum.get(class_name, 0) + prob

#             # Determine the final class by majority vote and calculate average probability for that class
#             final_class = max(class_votes, key=class_votes.get)
#             average_probability = probability_sum[final_class] / class_votes[final_class]

#             response = {
#                 "prediction": final_class,
#                 "probability": float(average_probability) * 100  # Convert to float
#             }

#         except Exception as e:
#             logging.error("Error in predict_video: %s", str(e))
#             raise HTTPException(status_code=500, detail=str(e))
        
#     return JSONResponse(content=response)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import tensorflow as tf
import tempfile
import numpy as np
import cv2
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forces TensorFlow to use the CPU

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load your model here. Adjust the path and loading method based on your model type.
def load_model():
    print("Checking model loading")
    model_path = "model_utils.h5"  # for TensorFlow/Keras
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("GPUs available:", physical_devices)
    else:
        print("No GPUs available. TensorFlow will use the CPU.")
    return model

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your model (adjust as needed)
model = load_model()
print("Model loading done.")
class_names = ['compressed', 'extended']

def classify_frame(frame, model):
    """
    Classify the given frame using the custom model.
    :param frame: Preprocessed frame.
    :param model: Custom model.
    :return: Tuple of (class_name, probability)
    """
    preds = model.predict(frame)

    # Assuming binary classification (A vs. B), where model output is a single probability
    prob_B = preds[0][0]      # Probability of class B ('extended')
    prob_A = 1 - prob_B        # Probability of class A ('compressed')
    
    # Determine the class with the highest probability
    if prob_A > prob_B:
        class_name = class_names[0]  # 'compressed'
        probability = prob_A
    else:
        class_name = class_names[1]  # 'extended'
        probability = prob_B
    
    return class_name, probability

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload Video for Prediction</title>
    </head>
    <body>
        <h1>Upload a Video for Prediction</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="video/*" required>
            <button type="submit">Upload and Predict</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):

    logging.info("Received a request for prediction.")
    try:
        # Check file type
        logging.info("Checking file type")
        if file.content_type not in ["video/mp4", "video/avi", "video/mov"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
        logging.info("checking file type done")
        # Process the video
        logging.info("Checking with tempfile.namedTemporaryFile")
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_file:
            logging.info("Write")
            temp_file.write(await file.read())
            logging.info("Write done")
            temp_file.flush()
            logging.info("FLushed")
            logging.info("Video capture variable")
            video_capture = cv2.VideoCapture(temp_file.name)
            logging.info("Video capture done")
            frames = []
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    logging.info("Inside while true break")
                    break
                frame = cv2.resize(frame, (224, 224))  # Adjust size based on model input
                frames.append(frame)

            logging.info("Before release")
            video_capture.release()
            logging.info("Frames", frames)
            logging.info("After release")


            frames_array = np.array(frames) / 255.0  # Normalizing frames to match model input preprocessing
            logging.info("frames_array", frames_array)
            logging.info("Shape: ", len(frames_array.shape))
            if len(frames_array.shape) == 3:
                frames_array = np.expand_dims(frames_array, axis=0)
            else:
                print("Len > 3")

            # Predict class and probability for each frame
            print("Predict class and probability")
            predictions = [classify_frame(np.expand_dims(f, axis=0), model) for f in frames_array]
            print("Predict class and probability done")
            # Aggregate predictions using majority voting
            class_votes = {}
            probability_sum = {}
            
            logging.info("Before for class_name")
            for class_name, prob in predictions:
                class_votes[class_name] = class_votes.get(class_name, 0) + 1
                probability_sum[class_name] = probability_sum.get(class_name, 0) + prob
            logging.info("For class_name done")
            # Determine the final class by majority vote and calculate average probability
            final_class = max(class_votes, key=class_votes.get)
            average_probability = probability_sum[final_class] / class_votes[final_class]

            logging.info("After avg probability")
            logging.info("Before building response")
            response = {
                "prediction": final_class,
                "probability": float(average_probability) * 100  # Convert to float
            }

            logging.info("After building response")

    except Exception as e:
        logging.info("Exception")
        logging.error("Error in predict_video: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
    logging.info("Returning response")
    return JSONResponse(content=response)

@app.post("/test")
async def test(input_str: str):
    print("testing post test")
    return {"received": input_str}