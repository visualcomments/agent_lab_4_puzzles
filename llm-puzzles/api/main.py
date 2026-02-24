from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
import os, queue

# Import local CallLLM helpers
try:
    from CallLLM import CONFIG, llm_query, get_models_list
except Exception as e:
    CONFIG, llm_query, get_models_list = None, None, None

app = FastAPI(
    title="LLM Query API",
    version="1.0.0",
    description="A minimal HTTP wrapper around CallLLM.llm_query, suitable for g4f-backed models."
)

def api_key_auth(x_api_key: Optional[str] = Header(default=None)):
    required = os.getenv("API_KEY")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

class AskRequest(BaseModel):
    model: str = Field(..., description="Model name recognized by g4f / AnyProvider")
    prompt: str = Field(..., description="User prompt / instruction")
    max_retries: int = Field(2, ge=0, le=10, description="Number of retry attempts")
    backoff_factor: float = Field(1.25, ge=0.5, le=10.0, description="Backoff multiplier between retries")
    stage: Optional[str] = Field("api", description="Arbitrary label used for logging in CallLLM")

class AskResponse(BaseModel):
    model: str
    output: str

class ModelsResponse(BaseModel):
    models: List[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models", response_model=ModelsResponse, dependencies=[Depends(api_key_auth)])
def models():
    if get_models_list is None or CONFIG is None:
        raise HTTPException(status_code=500, detail="CallLLM is not available")
    try:
        models = get_models_list(CONFIG)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse, dependencies=[Depends(api_key_auth)])
def ask(req: AskRequest):
    if llm_query is None or CONFIG is None:
        raise HTTPException(status_code=500, detail="CallLLM is not available")
    q = queue.Queue()
    try:
        output = llm_query(
            req.model,
            req.prompt,
            {'max_retries': req.max_retries, 'backoff_factor': req.backoff_factor},
            CONFIG,
            q,
            stage=req.stage or "api",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not output:
        raise HTTPException(status_code=502, detail="Empty response from provider")
    return {"model": req.model, "output": output}
