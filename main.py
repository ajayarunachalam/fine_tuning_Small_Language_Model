from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import uvicorn
from typing import Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SmallMedLM Chatbot API",
    description="Medical Language Model Inference API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    max_length: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    num_return_sequences: Optional[int] = 1

class ChatResponse(BaseModel):
    response: str
    query: str
    success: bool

# Global model and tokenizer variables
model = None
tokenizer = None
device = None

@app.on_event("startup")
async def load_model():
    """Load the PyTorch model and tokenizer on startup"""
    global model, tokenizer, device
    try:
        logger.info("Loading model and tokenizer...")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token
        logger.info("Tokenizer loaded successfully!")
        
        # Download model if not present
        model_path = 'SmallMedLM.pt'
        if not os.path.exists(model_path):
            logger.info("Model not found locally, downloading...")
            import urllib.request
            
            # Replace with our actual model URL: The google drive link or other hosting link
            model_url = os.getenv('MODEL_URL', 'YOUR_MODEL_DOWNLOAD_URL') # eg: https://drive.google.com/file/view?usp=drive_link
            urllib.request.urlretrieve(model_url, model_path)
            logger.info("Model downloaded successfully!")
        
        # Load model
        model = torch.load(model_path, map_location=device)
        model.eval()
        model.to(device)
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(device) if device else "not set"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else "not loaded",
        "tokenizer_status": "loaded" if tokenizer is not None else "not loaded",
        "device": str(device) if device else "unknown",
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main inference endpoint for chatbot queries
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")
    
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Tokenize input
        inputs = tokenizer.encode(
            request.query,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                num_return_sequences=request.num_return_sequences,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3  # Avoid repetition
            )
        
        # Decode response
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input query from response if it's included
        if response_text.startswith(request.query):
            response_text = response_text[len(request.query):].strip()
        
        logger.info(f"Generated response: {response_text[:100]}...")
        
        return ChatResponse(
            response=response_text,
            query=request.query,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/batch-chat")
async def batch_chat(queries: list[str], max_length: Optional[int] = 256):
    """
    Batch inference endpoint for multiple queries
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")
    
    try:
        responses = []
        for query in queries:
            request = ChatRequest(query=query, max_length=max_length)
            result = await chat(request)
            responses.append(result)
        
        return {"responses": responses, "count": len(responses)}
        
    except Exception as e:
        logger.error(f"Error during batch inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")

@app.post("/generate")
async def generate_custom(
    prompt: str,
    max_new_tokens: Optional[int] = 100,
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = 0.9,
    repetition_penalty: Optional[float] = 1.2
):
    """
    Custom generation endpoint with more control
    """
    if model is None or tokenizer is not None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")
    
    try:
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "tokens_generated": len(outputs[0]) - len(inputs[0])
        }
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

if __name__ == "__main__":
    # For local testing

    uvicorn.run(app, host="0.0.0.0", port=8000)
