import json
import logging
from collections.abc import AsyncGenerator
from typing import Union, List

import fastapi
from openai import APIError
from fastapi.responses import StreamingResponse

# Estas son las importaciones correctas y adaptadas
from fastapi_app.api_models import (
    ChatRequest,
    ChatResponse,
    AbastecimentoPublic,
    ErrorResponse
)
from fastapi_app.dependencies import ChatClient, CommonDeps, DBSession, EmbeddingsClient
from fastapi_app.postgres_searcher import PostgresSearcher
from fastapi_app.query_rewriter import rewrite_query

router = fastapi.APIRouter()
logger = logging.getLogger("ragapp")
ERROR_FILTER = {"error": "Your message contains content that was flagged by the content filter."}


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
    Esta es la versión final que integra toda nuestra lógica adaptada.
    """
    try:
        # 1. Re-escribe la consulta del usuario para obtener una consulta semántica y filtros
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
        # En el futuro, aquí se podría llamar a un LLM para una respuesta más elaborada
        final_answer = f"Búsqueda para '{search_query}' completada. Se encontraron {len(sources)} resultados relevantes."

        return ChatResponse(answer=final_answer, sources=sources)

    except Exception as e:
        logger.exception("Exception in chat_handler: %s", e)
        return ErrorResponse(error=str(e))

# NOTA: La ruta de streaming (/chat/stream) se ha omitido por simplicidad.
# Es mejor asegurarnos de que esta funciona primero.