import json
from typing import Any, List, Tuple
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionToolParam,
)

def build_search_function() -> list[ChatCompletionToolParam]:
    """
    This function defines the 'search_database' tool that the AI agent can use.
    It specifies the parameters the agent can use to filter the search.
    This version is adapted for the 'veiculos' and 'abastecimento' tables.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Segitarches the company's vehicle and fueling database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "A semantic search query. For example: 'efficient urban bus'",
                        },
                        "id_veiculo_filter": {
                            "type": "string",
                            "description": "Filter by the exact vehicle ID (id_veiculo).",
                        },
                        "placa_filter": {
                            "type": "string",
                            "description": "Filter by the exact vehicle license plate (placa).",
                        },
"date_filter": {
                            "type": "object",
                            "description": "Filtrar resultados por un rango de fechas. Usa el formato AAAA-MM-DD.",
                            "properties": {
                                "start_date": {"type": "string", "description": "Fecha de inicio (e.g., '2025-02-01')"},
                                "end_date": {"type": "string", "description": "Fecha de fin (e.g., '2025-02-28')"},
                            },
                        },z

                        "ano_filter": {
                            "type": "object",
                            "description": "Filter results by the vehicle's manufacturing year.",
                            "properties": {
                                "comparison_operator": {
                                    "type": "string",
                                    "description": "Operator for comparison, can be '>', '<', '>=', '<=', '='.",
                                },
                                "value": {
                                    "type": "number",
                                    "description": "The year to compare against, e.g., 2020.",
                                },
                            },
                        },
                        "fabricante_filter": {
                            "type": "string",
                            "description": "Filter results by the vehicle manufacturer, e.g., 'Volvo', 'Mercedes-Benz'.",
                        },
                        "tipo_onibus_filter": {
                            "type": "string",
                            "description": "Filter results by the bus type, e.g., 'Urbano', 'Rodoviário'.",
                        },
                    },
                    "required": ["search_query"],
                },
            },
        }
    ]

def extract_search_arguments(original_user_query: str, chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    search_query = None
    filters = []
    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type != "function":
                continue
            function = tool.function
            if function.name == "search_database":
                arg = json.loads(function.arguments)
                # Even though its required, search_query is not always specified
                search_query = arg.get("search_query", original_user_query)

                if "id_veiculo_filter" in arg and arg["id_veiculo_filter"]:
                    filters.append({"column": "id_veiculo", "operator": "=", "value": arg["id_veiculo_filter"]})

                if "placa_filter" in arg and arg["placa_filter"]:
                    filters.append({"column": "placa", "operator": "=", "value": arg["placa_filter"]})

                if "fabricante_filter" in arg and arg["fabricante_filter"]:
                    filters.append({"column": "fabricante", "operator": "=", "value": arg["fabricante_filter"]})

                if "tipo_onibus_filter" in arg and arg["tipo_onibus_filter"]:
                    filters.append({"column": "tipo_onibus", "operator": "=", "value": arg["tipo_onibus_filter"]})

                if "ano_filter" in arg and arg["ano_filter"] and isinstance(arg["ano_filter"], dict):
                    ano_filter_args = arg["ano_filter"]
                    filters.append(
                        {
                            "column": "ano",
                            "operator": ano_filter_args["comparison_operator"],
                            "value": ano_filter_args["value"],
                        }
                    )
    elif query_text := response_message.content:
        search_query = query_text.strip()
    return search_query, filters

async def rewrite_query(query: str, history: List[dict]) -> Tuple[str | None, List[dict[str, Any]]]:
    """
    Función orquestadora que llama al agente de IA y procesa la respuesta.
    Esta versión corregida SÍ respeta la configuración del HOST.
    """
    tools = build_search_function()
    messages = history + [{"role": "user", "content": query}]

    # Lógica para seleccionar el modelo correcto basado en el host
    openai_chat_host = os.getenv("OPENAI_CHAT_HOST", "azure")
    if openai_chat_host == "azure":
        model_name = AZURE_OPENAI_CHAT_DEPLOYMENT
    else:
        # Usa el modelo definido para OpenAI.com
        model_name = os.getenv("OPENAICOM_CHAT_MODEL")

    # Llama a la API de OpenAI con el nombre del modelo correcto
    chat_completion = await client.chat.completions.create(
        model=model_name, # <-- CORREGIDO
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    print(f"DEBUG: Respuesta completa de OpenAI:\n{chat_completion.model_dump_json(indent=2)}")

    # Procesa la respuesta usando la función de extracción que ya tienes
    return extract_search_arguments(query, chat_completion)