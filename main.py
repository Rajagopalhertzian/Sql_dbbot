import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load model and tokenizer
model_name = "t5-small"  # Replace with your fine-tuned model
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Initialize FastAPI
app = FastAPI()

# API Input/Output Schemas
class QueryRequest(BaseModel):
    natural_language: str
    dataset: str

class QueryResponse(BaseModel):
    sql_query: str

# Endpoint to generate SQL
@app.post("/generate_sql/", response_model=QueryResponse)
async def generate_sql(request: QueryRequest):
    try:
        input_text = f"Translate to SQL: {request.natural_language} Schema: {request.dataset}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(input_ids, max_length=256, num_beams=4, early_stopping=True)
        sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return QueryResponse(sql_query=sql_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
