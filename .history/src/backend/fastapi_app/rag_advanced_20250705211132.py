import json
from typing import Optional, Union, List

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
from openai.types.responses import EasyInputMessageParam

# CAMBIO: Importamos los modelos y filtros correctos
from fastapi_app.api_models import (
    ChatRequest,
    ChatResponse,
    AnoFilter, # Nuestro filtro complejo
    AbastecimentoPublic,
    ErrorResponse,
    RAGContext,
    RetrievalResponseDelta,
    SearchResults,
    ThoughtStep,
)
from fastapi_app.postgres_searcher import PostgresSearcher
from fastapi_app.rag_base import RAGChatBase

set_tracing_disabled(disabled=True)


class AdvancedRAGChat(RAGChatBase):
    # Los prompts y el __init__ se quedan igual, ya que la estructura del agente no cambia
    query_prompt_template = open(RAGChatBase.prompts_dir / "query.txt").read()
    answer_prompt_template = open(RAGChatBase.prompts_dir / "answer.txt").read()

    def __init__(
        self,
        *,
        messages: list,
        overrides: ChatRequest.Overrides,
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
        # CORRECCIÓN: Adaptamos la herramienta search_database
        self.search_agent = Agent(
            name="Searcher",
            instructions=self.query_prompt_template,
            tools=[function_tool(self.search_database)], # La herramienta ahora está adaptada
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

    # CORRECCIÓN: La herramienta ahora acepta nuestros filtros personalizados
    async def search_database(
        self,
        search_query: str,
        id_veiculo_filter: Optional[str] = None,
        placa_filter: Optional[str] = None,
        fabricante_filter: Optional[str] = None,
        tipo_onibus_filter: Optional[str] = None,
        ano_filter: Optional[AnoFilter] = None,
    ) -> SearchResults:
        """
        Busca en la base de datos de abastecimientos usando los filtros especificados.
        """
        filters: List[dict] = []
        if id_veiculo_filter:
            filters.append({"column": "id_veiculo", "operator": "=", "value": id_veiculo_filter})
        if placa_filter:
            filters.append({"column": "placa", "operator": "=", "value": placa_filter})
        if fabricante_filter:
            filters.append({"column": "fabricante", "operator": "=", "value": fabricante_filter})
        if tipo_onibus_filter:
            filters.append({"column": "tipo_onibus", "operator": "=", "value": tipo_onibus_filter})
        if ano_filter:
            filters.append({
                "column": "ano",
                "operator": ano_filter.comparison_operator,
                "value": ano_filter.value
            })
        
        results = await self.searcher.search_and_embed(
            search_query,
            top=self.chat_params.top,
            enable_vector_search=self.chat_params.enable_vector_search,
            enable_text_search=self.chat_params.enable_text_search,
            filters=filters,
        )
        # CORRECCIÓN: Devuelve los resultados como AbastecimentoPublic
        return SearchResults(
            query=search_query, items=[AbastecimentoPublic.model_validate(item, from_attributes=True) for item in results], filters=filters
        )

    # El resto del archivo utiliza la lógica original del agente, que ahora funcionará
    # con nuestra herramienta adaptada.
    async def prepare_context(self) -> tuple[list, list[str]]:
        agent_dialog = [
            EasyInputMessageParam(role="user", content=self.chat_params.query),
        ]
        runner = Runner(agent=self.search_agent)
        result = await runner.run_async(agent_dialog)
        thoughts = self.get_thoughts(result.history)
        tool_call = ItemHelpers.get_tool_calls(result.history[-1])
        if not tool_call:
            return [], thoughts
        items = ItemHelpers.get_value(tool_call[0])
        return items, thoughts

    async def answer_stream(self, items: list, earlier_thoughts: list[str]):
        rag_prompt = self.prepare_rag_request(self.chat_params.query, items)
        agent_dialog = [
            EasyInputMessageParam(role="user", content=rag_prompt),
        ]
        # CORRECCIÓN: Creamos un ID único para cada fuente de 'abastecimento'
        data_points = {f"{item.placa}-{item.data}": item.model_dump() for item in items}
        
        yield RetrievalResponseDelta(
            context=RAGContext(
                data_points=data_points,
                thoughts=earlier_thoughts,
            ),
        )
        runner = Runner(agent=self.answer_agent)
        async for event in runner.run_stream_async(agent_dialog):
            if isinstance(event, ResponseTextDeltaEvent):
                yield RetrievalResponseDelta(delta={"role": "assistant", "content": event.text})