from enum import Enum
from typing import Any, Optional, List
from datetime import date
from openai.types.responses import ResponseInputItemParam
from pydantic import BaseModel, Field


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


class ChatRequest(BaseModel):
    messages: List[Message]
    
    # ESTA ES LA ESTRUCTURA ANIDADA CORRECTA
    class Overrides(BaseModel):
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
        top: int = 3
        temperature: float = 0.3
        use_advanced_flow: bool = True
        seed: Optional[int] = None

    context: dict[str, Any] = Field(default_factory=lambda: {"overrides": ChatRequest.Overrides().model_dump()})



class ItemPublic(BaseModel):
    id: int
    type: str
    brand: str
    name: str
    description: str
    price: float

    def to_str_for_rag(self):
        return f"Name:{self.name} Description:{self.description} Price:{self.price} Brand:{self.brand} Type:{self.type}"


class ItemWithDistance(ItemPublic):
    distance: float

    def __init__(self, **data):
        super().__init__(**data)
        self.distance = round(self.distance, 2)


class ThoughtStep(BaseModel):
    title: str
    description: Any
    props: dict = {}


class RAGContext(BaseModel):
    data_points: dict[int, ItemPublic]
    thoughts: list[ThoughtStep]
    followup_questions: Optional[list[str]] = None


class ErrorResponse(BaseModel):
    error: str


class RetrievalResponse(BaseModel):
    message: Message
    context: RAGContext
    sessionState: Optional[Any] = None


class RetrievalResponseDelta(BaseModel):
    delta: Optional[Message] = None
    context: Optional[RAGContext] = None
    sessionState: Optional[Any] = None


class ChatParams(ChatRequestOverrides):
    prompt_template: str
    response_token_limit: int = 1024
    enable_text_search: bool
    enable_vector_search: bool
    original_user_query: str
    past_messages: list[ResponseInputItemParam]


class Filter(BaseModel):
    column: str
    comparison_operator: str
    value: Any


class PriceFilter(Filter):
    column: str = Field(default="price", description="The column to filter on (always 'price' for this filter)")
    comparison_operator: str = Field(description="The operator for price comparison ('>', '<', '>=', '<=', '=')")
    value: float = Field(description="The price value to compare against (e.g., 30.00)")


class BrandFilter(Filter):
    column: str = Field(default="brand", description="The column to filter on (always 'brand' for this filter)")
    comparison_operator: str = Field(description="The operator for brand comparison ('=' or '!=')")
    value: str = Field(description="The brand name to compare against (e.g., 'AirStrider')")


class SearchResults(BaseModel):
    query: str
    """The original search query"""

    items: list[ItemPublic]
    """List of items that match the search query and filters"""

    filters: list[Filter]
    """List of filters applied to the search results"""
    ############################################################

class AnoFilter(BaseModel):
    comparison_operator: str = Field(description="The operator for price comparison ('>', '<', '>=', '<=', '=')")
    value: int

class SearchRequest(BaseModel):
    search_query: str
    id_veiculo_filter: str | None = None
    placa_filter: str | None = None
    fabricante_filter: str | None = None
    tipo_onibus_filter: str | None = None
    ano_filter: AnoFilter | None = None

# --- Modelos para la Respuesta al Usuario ---

class VeiculoPublic(BaseModel):
    id_veiculo: str
    garagem: str | None = None
    placa: str | None = None
    ano: int | None = None
    tipo_onibus: str | None = None
    fabricante: str | None = None
    modelo_chassi: str | None = None

    class Config:
        from_attributes = True

class AbastecimentoPublic(BaseModel):
    id_veiculo: str
    placa: str | None = None
    data: date | None = None
    custo_combustivel: float | None = None
    km_diesel: float | None = None

    class Config:
        from_attributes = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[VeiculoPublic | AbastecimentoPublic]