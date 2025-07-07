import json
import logging
from collections.abc import AsyncGenerator
from typing import Union

import fastapi
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from openai import APIError
from sqlalchemy import select, text

from fastapi_app.api_models import (
    ChatRequest,
    ChatResponse,
    VeiculoPublic,      # <-- Corregido
    AbastecimentoPublic, # <-- Corregido
    ErrorResponse
)
from fastapi_app.dependencies import ChatClient, CommonDeps, DBSession, EmbeddingsClient
from fastapi_app.postgres_models import Veiculo, Abastecimento
from fastapi_app.postgres_searcher import PostgresSearcher
from fastapi_app.rag_advanced import AdvancedRAGChat
from fastapi_app.rag_simple import SimpleRAGChat

router = fastapi.APIRouter()


ERROR_FILTER = {"error": "Your message contains content that was flagged by the content filter."}


async def format_as_ndjson(r: AsyncGenerator[RetrievalResponseDelta, None]) -> AsyncGenerator[str, None]:
    """
    Format the response as NDJSON
    """
    try:
        async for event in r:
            yield event.model_dump_json() + "\n"
    except Exception as error:
        if isinstance(error, APIError) and error.code == "content_filter":
            yield json.dumps(ERROR_FILTER) + "\n"
        else:
            logging.exception("Exception while generating response stream: %s", error)
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
    Función principal que maneja la conversación del chat.
    """
    try:
        # 1. Re-escribe la consulta del usuario para obtener filtros
        search_query, filters = await rewrite_query(
            chat_request.messages[-1].content, 
            [m.model_dump() for m in chat_request.messages[:-1]]
        )

        if not search_query:
            return ChatResponse(answer="Por favor, haz una pregunta más específica.", sources=[])
        
        # 2. Crea una instancia de nuestro buscador adaptado
        searcher = PostgresSearcher(
            db_session=database_session,
            openai_embed_client=openai_embed.client,
            embed_deployment=context.openai_embed_deployment,
            embed_model=context.openai_embed_model,
            embed_dimensions=context.openai_embed_dimensions,
            embedding_column="embedding_main",
        )
        
        # 3. Busca en la base de datos (nuestro searcher ya está enfocado en 'abastecimento')
        results = await searcher.search_and_embed(
            query_text=search_query, 
            top=chat_request.context.overrides.top,
            filters=filters,
            enable_vector_search=True,
            enable_text_search=True,
        )

        # 4. Prepara las fuentes para la respuesta
        sources = [AbastecimentoPublic.model_validate(result, from_attributes=True) for result in results]
        
        # 5. Genera una respuesta simple para probar que el flujo funciona
        final_answer = f"Búsqueda para '{search_query}' completada. Se encontraron {len(sources)} resultados relevantes."

        return ChatResponse(answer=final_answer, sources=sources)

    except Exception as e:
        logging.exception("Exception in chat_handler: %s", e)
        return ErrorResponse(error=str(e))

@router.post("/chat/stream")
async def chat_stream_handler(
    context: CommonDeps,
    database_session: DBSession,
    openai_embed: EmbeddingsClient,
    openai_chat: ChatClient,
    chat_request: ChatRequest,
):
    searcher = PostgresSearcher(
        db_session=database_session,
        openai_embed_client=openai_embed.client,
        embed_deployment=context.openai_embed_deployment,
        embed_model=context.openai_embed_model,
        embed_dimensions=context.openai_embed_dimensions,
        embedding_column=context.embedding_column,
    )

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
        # Intentionally do search we stream down the answer, to avoid using database connections during stream
        # See https://github.com/tiangolo/fastapi/discussions/11321
        items, thoughts = await rag_flow.prepare_context()
        result = rag_flow.answer_stream(items, thoughts)
        return StreamingResponse(content=format_as_ndjson(result), media_type="application/x-ndjson")
    except Exception as e:
        if isinstance(e, APIError) and e.code == "content_filter":
            return StreamingResponse(
                content=json.dumps(ERROR_FILTER) + "\n",
                media_type="application/x-ndjson",
            )
        else:
            logging.exception("Exception while generating response: %s", e)
            return StreamingResponse(
                content=json.dumps({"error": str(e)}, ensure_ascii=False) + "\n",
                media_type="application/x-ndjson",
            )
