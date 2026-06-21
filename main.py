from fastapi import FastAPI,UploadFile,File,Form,Query
from ocr_engine import ocr_logic
from logic import pdf_checker
from vision import vision_response
from index_builder import buildIndex,getEmbeddings
from typing import List
from retriever import retrieveChunks
from getAnswer import getAnswer

app = FastAPI()


@app.get('/')
def home():
    return {"message":"hi there!"}


@app.post('/')
async def upload_pdf(file: UploadFile = File(...)):
    
    response = await pdf_checker(file.file)
    
    message = buildIndex(response)
    
    return {"message":"Knowlege Base Created Sucessfully"}
    


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


@app.get('/embeddings')
async def generateEmbedding(queryList: List[str] = Query(...)):
    embeddings = getEmbeddings(queryList)
    return {"embeddings": embeddings.tolist()}


@app.post("/retrieve")
async def retrieve(data: str):
    chunks = await retrieveChunks(data)
    return {"result":chunks}


@app.get("/answer")
async def llmAnswer(data: str):
    response = await getAnswer(data)
    
    return response
    