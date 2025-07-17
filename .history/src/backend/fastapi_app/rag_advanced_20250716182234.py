import json
import logging
from typing import Optional, Union, List, Tuple

from openai import AsyncAzureOpenAI, AsyncOpenAI
# CÓDIGO CORREGIDO
from openai.types.chat import ChatCompletion
from openai.types.responses import EasyInputMessageParam # <-- Se importa desde aquí# CAMBIO: Ya no importamos nada de la librería 'agents'

# Importaciones consistentes
from fastapi_app.api_models import (
    AnoFilter, AbastecimentoPublic, ChatRequest, ChatRequestOverrides, RAGContext,
    RetrievalResponse, RetrievalResponseDelta, SearchResults, ThoughtStep, Message, AIChatRoles
)
from fastapi_app.postgres_searcher import PostgresSearcher
from fastapi_app.rag_base import RAGChatBase
from fastapi_app.query_rewriter import build_search_function, extract_search_arguments # Importamos las funciones que necesitamos


class AdvancedRAGChat(RAGChatBase):

    query_fewshots = json.loads(open(RAGChatBase.prompts_dir / "query_fewshots.json").read())


    def __init__(
        self,
        *,
        messages: list[Message],
        overrides: ChatRequestOverrides,
        searcher: PostgresSearcher,
        openai_chat_client: Union[AsyncOpenAI, AsyncAzureOpenAI],
        chat_model: str,
        chat_deployment: Optional[str] = None,
    ):
        self.searcher = searcher
        self.openai_chat_client = openai_chat_client
        self.chat_params = self.get_chat_params(messages, overrides)
        self.chat_model = chat_model
        self.chat_deployment = chat_deployment


    async def prepare_context(self) -> tuple[list, list[ThoughtStep]]:
            """
            Prepara el contexto para la respuesta.
            Esta versión corregida soluciona el error 'model_dump'.
            """
            user_query = self.chat_params.original_user_query
            history = self.chat_params.past_messages

            # 1. Llama a la API de OpenAI para obtener los filtros
            tools = build_search_function()
            messages_for_llm = self.query_fewshots + history + [{"role": "user", "content": user_query}]

            chat_completion = await self.openai_chat_client.chat.completions.create(
                model=self.chat_deployment or self.chat_model,
                messages=messages_for_llm,
                tools=tools,
                tool_choice="auto",
            )

            # 2. Extrae los argumentos y filtros
            search_query, filters = extract_search_arguments(user_query, chat_completion)

            print("--- DEBUG: Plan de Búsqueda Generado ---")
            print(f"Search Query: {search_query}")
            print(f"Filters: {filters}")
            print("--------------------------------------")

            if not search_query:
                raise ValueError("El modelo no generó una consulta de búsqueda.")
            print(f"DEBUG >> voy a llamar a search_and_embed con top={top_k} y filters={filters}")

            # 3. Busca en la base de datos (devuelve objetos de base de datos)
            search_results = await self.searcher.search_and_embed(
                search_query,
                top=self.chat_params.top,
                enable_vector_search=self.chat_params.enable_vector_search,
                enable_text_search=self.chat_params.enable_text_search,
                filters=filters,
            )
            
            # 4. Prepara los "pensamientos" para el frontend
            thoughts = [
                ThoughtStep(title="Search query generated", description=search_query),
                ThoughtStep(title="Filters applied", description=filters),
                # CORRECCIÓN: Se convierten los objetos a su versión pública solo para esta descripción
                ThoughtStep(title="Search results", description=[AbastecimentoPublic.model_validate(item, from_attributes=True).model_dump() for item in search_results]),
            ]
            # Se devuelven los resultados originales (objetos de base de datos)
            return search_results, thoughts

    async def answer_stream(self, items: list, earlier_thoughts: list[ThoughtStep]):
            rag_prompt = self.prepare_rag_request(self.chat_params.original_user_query, items)
            
            # Prepara los mensajes para la API de OpenAI
            messages_for_llm = self.chat_params.past_messages + [{"role": "user", "content": rag_prompt}]
            print(f"DEBUG: Se encontraron {len(items)} resultados. Guion final enviado a la IA:\n---\n{rag_prompt}\n---")

            # Prepara el contexto para enviarlo al frontend
            web_friendly_items = [AbastecimentoPublic.model_validate(item, from_attributes=True) for item in items]
            data_points = {f"{item.placa}-{item.data}": item.model_dump() for item in web_friendly_items}
            
            yield RetrievalResponseDelta(
                delta=Message(content="", role=AIChatRoles.ASSISTANT),
                context=RAGContext(data_points=data_points, thoughts=earlier_thoughts)
            )
            
            # CORRECCIÓN: Se reemplaza la lógica del 'Runner' por una llamada directa a la API
            response_stream = await self.openai_chat_client.chat.completions.create(
                model=self.chat_deployment or self.chat_model,
                messages=messages_for_llm,
                temperature=self.chat_params.temperature,
                max_tokens=self.chat_params.response_token_limit,
                seed=self.chat_params.seed,
                stream=True
            )

            # Procesa la respuesta en "pedacitos" y la envía al frontend
            async for chunk in response_stream:
                content_delta = chunk.choices[0].delta.content
                if content_delta:
                    yield RetrievalResponseDelta(delta=Message(content=content_delta, role=AIChatRoles.ASSISTANT))