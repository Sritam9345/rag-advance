from fastapi import FastAPI,UploadFile,File,Form
from ocr_engine import ocr_logic
from logic import pdf_checker
from vision import vision_response

app = FastAPI()


@app.get('/')
def home():
    return {"message":"hi there!"}


@app.post('/')
async def upload_pdf(file: UploadFile = File(...)):
    
    response = await pdf_checker(file.file)
    
    return response
    


@app.get('/OCR')
async def ocr_engine(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    text = ocr_logic(image_bytes)
    
    return text
    
    
    

@app.get('/VISION')
async def vision_engine(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    text = await vision_response(image_bytes)
    
    return text