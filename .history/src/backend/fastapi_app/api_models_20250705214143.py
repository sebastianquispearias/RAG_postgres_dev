from enum import Enum
from typing import Any, Optional, List
from datetime import date
from pydantic import BaseModel, Field

# --- Modelos para la Petición de Chat ---

class AIChatRoles(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    content: str
    role: AIChatRoles = AIChatRoles.USER

class RetrievalMode(str, Enum):
    TEXT = "text"
    VECTORS = "vectors"
    HYBRID = "hybrid"

class ChatRequestOverrides(BaseModel):
    top: int = 3
    temperature: float = 0.3
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    use_advanced_flow: bool = True
    seed: Optional[int] = None

class ChatRequestContext(BaseModel):
    overrides: ChatRequestOverrides

class ChatRequest(BaseModel):
    messages: List[Message]
    context: Optional[ChatRequestContext] = None
    sessionState: Optional[Any] = None


# --- Modelos para los Filtros de Búsqueda ---

class AnoFilter(BaseModel):
    comparison_operator: str = Field(description="El operador para la comparación ('>', '<', '>=', '<=', '=')")
    value: int


# --- Modelos para la Respuesta al Usuario ---

class AbastecimentoPublic(BaseModel):
    id_veiculo: str
    placa: Optional[str] = None
    data: Optional[date] = None
    custo_combustivel: Optional[float] = None
    km_diesel: Optional[float] = None

    class Config:
        from_attributes = True

class SearchResults(BaseModel):
    query: str
    items: List[AbastecimentoPublic]
    filters: List[dict]

class ThoughtStep(BaseModel):
    title: str
    description: Optional[Any] = None
    props: Optional[dict[str, Any]] = None

class RAGContext(BaseModel):
    data_points: dict
    thoughts: List[ThoughtStep]

class RetrievalResponseDelta(BaseModel):
    delta: Message
    context: Optional[RAGContext] = None

class RetrievalResponse(BaseModel):
    message: Message
    context: RAGContext
    session_state: Optional[Any] = None