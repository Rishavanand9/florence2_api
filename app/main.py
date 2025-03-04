import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys

# Prevent `_bz2` import errors
sys.modules["_bz2"] = None
sys.modules["bz2"] = None


# Set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model and processor
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Function to process a single image
def process_image(image_path):
    try:
        import bz2  # Try importing `_bz2`
    except ImportError:
        bz2 = None  # Set to `None` if missing

    try:
        # Load image
        image = Image.open(image_path)
        
        # For OCR task, prompt must be just "<OCR>" with no additional text
        prompt = "<OCR>"
        
        # Process inputs
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        
        # Generate output
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=4096,
            num_beams=3,
            do_sample=False
        )
        
        # Decode generated text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Parse results based on task
        parsed_answer = processor.post_process_generation(generated_text, task="<OCR>", image_size=(image.width, image.height))
        
        return {
            "status": 200,
            "result": parsed_answer
        }
    
    except Exception as e:
        return {
            "status": 500,
            "result": f"Error processing image: {str(e)}"
        }


@app.post("/read-image")
async def read_image(file: UploadFile = File(...)):
    try:
        # Open image from uploaded file  
        image = Image.open(file.file)
        
        # Process image
        response = process_image(image)
        
        return response
    
    except Exception as e:
        return {"status": 500, "result": f"Error: {str(e)}"}
