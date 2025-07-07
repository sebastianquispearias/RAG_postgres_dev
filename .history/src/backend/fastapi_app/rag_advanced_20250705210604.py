import json
from collections.abc import AsyncGenerator
from typing import Optional, Union

from agents import (
    Agent,
    ItemHelpers,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    ToolCallOutputItem,
    function_tool,
    set_tracing_disabled,
)
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.responses import EasyInputMessageParam, ResponseInputItemParam, ResponseTextDeltaEvent

from fastapi_app.api_models import (
    AIChatRoles,
    BrandFilter,
    ChatRequestOverrides,
    Filter,
    AbastecimentoPublic ,
    Message,
    PriceFilter,
    RAGContext,
    RetrievalResponse,
    RetrievalResponseDelta,
    SearchResults,
    ThoughtStep,
)
from fastapi_app.postgres_searcher import PostgresSearcher
from fastapi_app.rag_base import RAGChatBase

set_tracing_disabled(disabled=True)


class AdvancedRAGChat(RAGChatBase):
    query_prompt_template = open(RAGChatBase.prompts_dir / "query.txt").read()
    query_fewshots = open(RAGChatBase.prompts_dir / "query_fewshots.json").read()

    def __init__(
        self,
        *,
        messages: list[ResponseInputItemParam],
        overrides: ChatRequestOverrides,
        searcher: PostgresSearcher,
        openai_chat_client: Union[AsyncOpenAI, AsyncAzureOpenAI],
        chat_model: str,
        chat_deployment: Optional[str],  # Not needed for non-Azure OpenAI
    ):
        self.searcher = searcher
        self.chat_params = self.get_chat_params(messages, overrides)
        self.model_for_thoughts = (
            {"model": chat_model, "deployment": chat_deployment} if chat_deployment else {"model": chat_model}
        )
        openai_agents_model = OpenAIChatCompletionsModel(
            model=chat_model if chat_deployment is None else chat_deployment, openai_client=openai_chat_client
        )
        self.search_agent = Agent(
            name="Searcher",
            instructions=self.query_prompt_template,
            tools=[function_tool(self.search_database)],
            tool_use_behavior="stop_on_first_tool",
            model=openai_agents_model,
        )
        self.answer_agent = Agent(
            name="Answerer",
            instructions=self.answer_prompt_template,
            model=openai_agents_model,
            model_settings=ModelSettings(
                temperature=self.chat_params.temperature,
                max_tokens=self.chat_params.response_token_limit,
                extra_body={"seed": self.chat_params.seed} if self.chat_params.seed is not None else {},
            ),
        )

    async def search_database(
        self,
        search_query: str,
        price_filter: Optional[PriceFilter] = None,
        brand_filter: Optional[BrandFilter] = None,
    ) -> SearchResults:
        """
        Search PostgreSQL database for relevant products based on user query

        Args:
            search_query: English query string to use for full text search, e.g. 'red shoes'.
            price_filter: Filter search results based on price of the product
            brand_filter: Filter search results based on brand of the product

        Returns:
            List of formatted items that match the search query and filters
        """
        # Only send non-None filters
        filters: list[Filter] = []
        if price_filter:
            filters.append(price_filter)
        if brand_filter:
            filters.append(brand_filter)
        results = await self.searcher.search_and_embed(
            search_query,
            top=self.chat_params.top,
            enable_vector_search=self.chat_params.enable_vector_search,
            enable_text_search=self.chat_params.enable_text_search,
            filters=filters,
        )
        return SearchResults(
            query=search_query, items=[ItemPublic.model_validate(item.to_dict()) for item in results], filters=filters
        )

    async def prepare_context(self) -> tuple[list[ItemPublic], list[ThoughtStep]]:
# ... (el __init__ se queda igual)

    async def prepare_context(self) -> tuple[list, list[str]]: # Devuelve una lista genérica
        """
        Prepara el contexto para la respuesta, reescribiendo la consulta y buscando en la base de datos.
        """
        user_query = self.messages[-1].content
        history = [m.model_dump() for m in self.messages[:-1]]

        search_query, filters = await rewrite_query(user_query, history)

        if not search_query:
            return [], ["No se generó una consulta de búsqueda. Por favor, reformula tu pregunta."]

        results = await self.searcher.search_and_embed(
            query_text=search_query,
            filters=filters,
            top=self.overrides.top,
            enable_vector_search=self.overrides.retrieval_mode in ["vectors", "hybrid"],
            enable_text_search=self.overrides.retrieval_mode in ["text", "hybrid"],
        )
        
        thoughts = [
            f"Búsqueda semántica para '{search_query}'",
            f"Filtros aplicados: {filters}",
            f"Se encontraron {len(results)} resultados de la base de datos."
        ]
        return results, thoughts

    async def answer_stream(
        self,
        items: list, # Acepta una lista genérica
        earlier_thoughts: list[str],
    ) -> AsyncGenerator[dict, None]:
        """
        Genera la respuesta final en tiempo real (streaming).
        """
        # CORRECCIÓN: Creamos un ID único para cada fuente de 'abastecimento' para el contexto.
        # Usamos una combinación de placa y fecha para crear una clave.
        data_points = {f"{item.placa}-{item.data}": item.model_dump() for item in items}

        # Preparamos el prompt final para el LLM que generará la respuesta.
        rag_prompt = self.prepare_rag_request(self.messages[-1].content, items)
        
        # En una implementación completa, aquí se llamaría al LLM de chat.
        # Por ahora, simulamos una respuesta para probar el flujo.
        final_answer = f"Respuesta basada en {len(items)} fuentes encontradas."

        # Enviamos los datos de contexto al frontend
        yield {
            "context": {
                "data_points": data_points,
                "thoughts": earlier_thoughts,
            }
        }
        # Enviamos la respuesta final
        yield {"delta": {"role": "assistant", "content": final_answer}}