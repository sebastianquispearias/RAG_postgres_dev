from typing import List, Optional, Union

from openai import AsyncAzureOpenAI, AsyncOpenAI

# CAMBIO: Se usan los modelos correctos
from fastapi_app.api_models import ChatRequestOverrides, Message, AbastecimentoPublic, RetrievalResponse, RAGContext, AIChatRoles, ThoughtStep
from fastapi_app.postgres_searcher import PostgresSearcher
from fastapi_app.rag_base import RAGChatBase


class SimpleRAGChat(RAGChatBase):
    def __init__(
        self,
        *,
        messages: List[Message],
        overrides: ChatRequestOverrides,
        searcher: PostgresSearcher,
        openai_chat_client: Union[AsyncOpenAI, AsyncAzureOpenAI],
        chat_model: str,
        chat_deployment: Optional[str] = None,
    ):
        self.messages = messages
        self.overrides = overrides
        self.searcher = searcher
        self.openai_chat_client = openai_chat_client
        self.chat_model = chat_model
        self.chat_deployment = chat_deployment
        self.chat_params = self.get_chat_params(messages, overrides)

    async def prepare_context(self) -> tuple[list[AbastecimentoPublic], list[ThoughtStep]]:
        query = self.messages[-1].content

        results = await self.searcher.search_and_embed(
            query,
            top=self.chat_params.top,
            enable_vector_search=self.chat_params.enable_vector_search,
            enable_text_search=self.chat_params.enable_text_search,
        )

        thoughts = [ThoughtStep(title="Search results", description=[item.model_dump() for item in results])]
        return results, thoughts
    
    async def answer(self, items: list[AbastecimentoPublic], earlier_thoughts: list[ThoughtStep]) -> RetrievalResponse:
        rag_prompt = self.prepare_rag_request(self.chat_params.original_user_query, items)
        
        response = await self.openai_chat_client.chat.completions.create(
            model=self.chat_deployment or self.chat_model,
            messages=[{"role": "user", "content": rag_prompt}],
            temperature=self.chat_params.temperature,
            max_tokens=self.chat_params.response_token_limit,
            seed=self.chat_params.seed,
        )
        
        response_content = response.choices[0].message.content or ""
        
        return RetrievalResponse(
            message=Message(content=response_content, role=AIChatRoles.ASSISTANT),
            context=RAGContext(
                data_points={f"{item.placa}-{item.data}": item.model_dump() for item in items},
                thoughts=earlier_thoughts,
            ),
        )