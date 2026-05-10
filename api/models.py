from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal, Union


class Message(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):                                                                                                                         
      model: str                                                                                                                                                  
      messages: List[Message]                                                                                                                                     
      max_tokens: Optional[int] = Field(default=256, ge=1)                                                                                                        
      temperature: Optional[float] = Field(default=0.7, ge=0, le=2)                                                                                               
      top_p: Optional[float] = Field(default=0.9, ge=0, le=1)                                                                                                     
      stream: Optional[bool] = False                                                                                                                              
      stop: Optional[Union[str, List[str]]] = None                                                                                                                
      priority: Literal["high", "normal", "low"] = "normal"
                                                                                                                                                                  
      # Accept but ignore for now, so OpenAI clients don't fail validation.                                                                                       
      n: Optional[int] = 1                                                                                                                                        
      presence_penalty: Optional[float] = 0                                                                                                                       
      frequency_penalty: Optional[float] = 0                                                                                                                      
      user: Optional[str] = None                                                                                                                                  
      metadata: Optional[Dict[str, Any]] = None 


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(default=256, ge=1)
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    top_p: Optional[float] = Field(default=0.9, ge=0, le=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    n: Optional[int] = 1
    user: Optional[str] = None
    priority: Literal["high", "normal", "low"] = "normal"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completions"
    created: int
    model: str
    choices: list[dict]
    usage: dict

class MetricsResponse(BaseModel):
    pages_used: int
    pages_total: int
    utilization_pct: float
    waiting_requests: int
    active_requests: int
    total_completed: int
