import asyncio
import logging
import time
from typing import List

from fastapi import FastAPI, Request
from models.RAG import RAG_pipeline
from models.rating_predict import predict_with_model
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    request_id: int
    results: List[str]

class ReviewRequest(BaseModel):
    review: str

class RatingResponse(BaseModel):
    request_id: int
    rating: float

    
app = FastAPI()
@app.post("/predict_rating", response_model=RatingResponse)
async def predict_rating(request: ReviewRequest):
    global request_id_counter, request_count
    request_id_counter += 1
    request_count += 1

    rating = predict_with_model(request.review)
    return RatingResponse(request_id=request_id_counter, rating=rating)

@app.post("/get_answer", response_model=QueryResponse)
async def get_answer(request: QueryRequest):
    global request_id_counter, request_count
    request_id_counter += 1
    request_count += 1

    results = RAG_pipeline(request.query)
    return QueryResponse(request_id=request_id_counter, results=results)

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
