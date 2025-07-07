import json
import logging
from collections.abc import AsyncGenerator
from typing import Union, List

import fastapi
from openai import APIError
from fastapi.responses import StreamingResponse

# Importaciones adaptadas
from fastapi_app.api_models import (
    ChatRequest,
    ChatResponse,
    AbastecimentoPublic,
    ErrorResponse,
    RetrievalResponseDelta, # <-- Añadido para el stream
)
from fastapi_app.dependencies import ChatClient, CommonDeps, DBSession, EmbeddingsClient
from fastapi_app.postgres_searcher import PostgresSearcher
from fastapi_app.query_rewriter import rewrite_query
from fastapi_app.rag_advanced import AdvancedRAGChat
from fastapi_app.rag_simple import SimpleRAGChat

router = fastapi.APIRouter()
logger = logging.getLogger("ragapp")
ERROR_FILTER = {"error": "Your message contains content that was flagged by the content filter."}

async def format_as_ndjson(r: AsyncGenerator[RetrievalResponseDelta, None]) -> AsyncGenerator[str, None]:
    """
    Función de ayuda para formatear la respuesta de streaming.
    """
    try:
        async for event in r:
            yield event.model_dump_json() + "\n"
    except Exception as error:
        # ... (manejo de errores)
        yield json.dumps({"error": str(error)}, ensure_ascii=False) + "\n"


@router.post("/chat")
async def chat_handler(
    context: CommonDeps,
    database_session: DBSession,
    openai_chat: ChatClient,
    openai_embed: EmbeddingsClient,
    chat_request: ChatRequest,
) -> Union[ChatResponse, ErrorResponse]:
    """
    Maneja las peticiones de chat que esperan una respuesta completa al final.
    """
    try:
        search_query, filters = await rewrite_query(
            chat_request.messages[-1].content, 
            [m.model_dump() for m in chat_request.messages[:-1]]
        )

        if not search_query:
            return ChatResponse(answer="Por favor, haz una pregunta más específica.", sources=[])
        
        searcher = PostgresSearcher(
            db_session=database_session,
            openai_embed_client=openai_embed.client,
            embed_deployment=context.openai_embed_deployment,
            embed_model=context.openai_embed_model,
            embed_dimensions=context.openai_embed_dimensions,
            embedding_column="embedding_main",
        )
        
        results = await searcher.search_and_embed(
            query_text=search_query, 
            top=chat_request.context.overrides.top,
            filters=filters,
            enable_vector_search=True,
            enable_text_search=True,
        )

        sources = [AbastecimentoPublic.model_validate(result, from_attributes=True) for result in results]
        
        # En una implementación futura, aquí se usaría un LLM para generar la respuesta.
        # Por ahora, devolvemos una respuesta directa para probar el flujo.
        final_answer = f"Búsqueda para '{search_query}' completada. Se encontraron {len(sources)} resultados relevantes."

        return ChatResponse(answer=final_answer, sources=sources)

    except Exception as e:
        logger.exception("Exception in chat_handler: %s", e)
        return ErrorResponse(error=str(e))


@router.post("/chat/stream")
async def chat_stream_handler(
    context: CommonDeps,
    database_session: DBSession,
    openai_chat: ChatClient,
    openai_embed: EmbeddingsClient,
    chat_request: ChatRequest,
):
    """
    Maneja las peticiones de chat que esperan una respuesta en tiempo real (streaming).
    """
    searcher = PostgresSearcher(
        db_session=database_session,
        openai_embed_client=openai_embed.client,
        embed_deployment=context.openai_embed_deployment,
        embed_model=context.openai_embed_model,
        embed_dimensions=context.openai_embed_dimensions,
        embedding_column="embedding_main",
    )
    
    # El repositorio original usa las clases RAG para el streaming. Las reutilizamos.
    rag_flow: Union[SimpleRAGChat, AdvancedRAGChat]
    if chat_request.context.overrides.use_advanced_flow:
        rag_flow = AdvancedRAGChat(
            messages=chat_request.messages,
            overrides=chat_request.context.overrides,
            searcher=searcher,
            openai_chat_client=openai_chat.client,
            chat_model=context.openai_chat_model,
            chat_deployment=context.openai_chat_deployment,
        )
    else:
        rag_flow = SimpleRAGChat(
            messages=chat_request.messages,
            overrides=chat_request.context.overrides,
            searcher=searcher,
            openai_chat_client=openai_chat.client,
            chat_model=context.openai_chat_model,
            chat_deployment=context.openai_chat_deployment,
        )

    try:
        # La lógica de streaming del repositorio original
        items, thoughts = await rag_flow.prepare_context()
        result = rag_flow.answer_stream(items, thoughts)
        return StreamingResponse(content=format_as_ndjson(result), media_type="application/x-ndjson")
    except Exception as e:
        # ... (manejo de errores para el stream)
        return StreamingResponse(
            content=json.dumps({"error": str(e)}, ensure_ascii=False) + "\n",
            media_type="application/x-ndjson",
        )