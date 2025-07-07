import json
from typing import Optional, Union, List
from agents import Agent, ItemHelpers, ModelSettings, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.responses import EasyInputMessageParam, ResponseTextDeltaEvent

# CAMBIO: Importaciones adaptadas y consistentes
from fastapi_app.api_models import (
    AnoFilter, AbastecimentoPublic, ChatRequest, ChatRequestOverrides, RAGContext,
    RetrievalResponse, RetrievalResponseDelta, SearchResults, ThoughtStep, Message, AIChatRoles
)
from fastapi_app.postgres_searcher import PostgresSearcher
from fastapi_app.rag_base import RAGChatBase

set_tracing_disabled(disabled=True)

class AdvancedRAGChat(RAGChatBase):
    query_prompt_template = open(RAGChatBase.prompts_dir / "query.txt").read()
    answer_prompt_template = open(RAGChatBase.prompts_dir / "answer.txt").read()

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
        id_veiculo_filter: Optional[str] = None,
        placa_filter: Optional[str] = None,
        fabricante_filter: Optional[str] = None,
        tipo_onibus_filter: Optional[str] = None,
        ano_filter: Optional[AnoFilter] = None,
    ) -> SearchResults:
        filters: List[dict] = []
        if id_veiculo_filter: filters.append({"column": "id_veiculo", "operator": "=", "value": id_veiculo_filter})
        if placa_filter: filters.append({"column": "placa", "operator": "=", "value": placa_filter})
        if fabricante_filter: filters.append({"column": "fabricante", "operator": "=", "value": fabricante_filter})
        if tipo_onibus_filter: filters.append({"column": "tipo_onibus", "operator": "=", "value": tipo_onibus_filter})
        if ano_filter: filters.append({"column": "ano", "operator": ano_filter.comparison_operator, "value": ano_filter.value})

        results = await self.searcher.search_and_embed(
            search_query,
            top=self.chat_params.top,
            enable_vector_search=self.chat_params.enable_vector_search,
            enable_text_search=self.chat_params.enable_text_search,
            filters=filters,
        )
        return SearchResults(
            query=search_query, items=[AbastecimentoPublic.model_validate(item, from_attributes=True) for item in results], filters=filters
        )

    async def prepare_context(self) -> tuple[list[AbastecimentoPublic], list[ThoughtStep]]:
        user_query = f"Find search results for user query: {self.chat_params.original_user_query}"
        new_user_message = EasyInputMessageParam(role="user", content=user_query)
        # La librería 'agents' espera un formato específico, por eso la conversión
        past_messages_for_agent = [EasyInputMessageParam(role=m['role'], content=m['content']) for m in self.chat_params.past_messages]
        all_messages = past_messages_for_agent + [new_user_message]

        run_results = await Runner.run(self.search_agent, input=all_messages)
        most_recent_response = run_results.new_items[-1]

        if isinstance(most_recent_response, ItemHelpers.ToolCallOutputItem):
            search_results = most_recent_response.output
        else:
            raise ValueError("Error retrieving search results, model did not call tool properly")
        
        thoughts = [
             ThoughtStep(title="Search results", description=[item.model_dump() for item in search_results.items]),
        ]
        return search_results.items, thoughts

    async def answer_stream(self, items: list[AbastecimentoPublic], earlier_thoughts: list[ThoughtStep]):
        rag_prompt = self.prepare_rag_request(self.chat_params.original_user_query, items)
        # La librería 'agents' espera un formato específico
        past_messages_for_agent = [EasyInputMessageParam(role=m['role'], content=m['content']) for m in self.chat_params.past_messages]
        agent_dialog = past_messages_for_agent + [EasyInputMessageParam(role="user", content=rag_prompt)]
        
        data_points = {f"{item.placa}-{item.data}": item.model_dump() for item in items}
        
        yield RetrievalResponseDelta(context=RAGContext(data_points=data_points, thoughts=earlier_thoughts))
        
        runner = Runner(agent=self.answer_agent)
        async for event in runner.run_stream_async(agent_dialog):
            if isinstance(event, ResponseTextDeltaEvent):
                yield RetrievalResponseDelta(delta=Message(content=str(event.delta), role=AIChatRoles.ASSISTANT))