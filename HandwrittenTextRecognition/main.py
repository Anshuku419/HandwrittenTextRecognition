import gradio as gr
from keras.models import load_model
import keras.backend as K
import cv2
import numpy as np

# Load the CRNN model
try:
    model = load_model('Text_recognizer_Using_CRNN.h5')
    print("✅ Model loaded successfully.")
    model.summary()
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

def process_image(img):
    """Preprocess the input image."""
    try:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        print("Grayscale image shape:", img.shape)

        h, w = img.shape
        new_h = 32
        new_w = int((new_h / h) * w)
        img = cv2.resize(img, (new_w, new_h))

        if new_w < 128:
            pad = np.full((32, 128 - new_w), 255)
            img = np.concatenate((img, pad), axis=1)
        else:
            img = cv2.resize(img, (128, 32))

        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        return img
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return None

def predict_image_text(input_img):
    """Predict text from an image using the CRNN model."""
    try:
        img = process_image(input_img)
        if img is None:
            return "⚠️ Error processing image."

        img = np.expand_dims(img, axis=0)
        print("Model input shape:", img.shape)

        prediction = model.predict(img)
        print("Raw model output:", prediction)

        decoded = K.ctc_decode(prediction,
                               input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                               greedy=True)[0][0]
        output_sequence = K.get_value(decoded)
        print("Decoded output:", output_sequence)

        char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        text_result = "".join([char_list[i] for seq in output_sequence for i in seq if i != -1])

        return text_result if text_result else "⚠️ No text detected."
    except Exception as e:
        return f"❌ Error: {e}"

# Gradio Interface
demo = gr.Interface(predict_image_text, gr.Image(), "text",
                     title="Image to Text Conversion",
                     description="Upload an image and get the extracted text.")

demo.launch()
