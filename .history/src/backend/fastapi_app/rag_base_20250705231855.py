from pathlib import Path
from typing import Optional

# CAMBIO: Se usan los modelos correctos desde api_models
from fastapi_app.api_models import ChatParams, ChatRequestOverrides, Message, AbastecimentoPublic

class RAGChatBase:
    prompts_dir = Path(__file__).parent.resolve() / "prompts"
    answer_prompt_template: str = open(prompts_dir / "answer.txt").read()

    def get_chat_params(self, messages: list[Message], overrides: ChatRequestOverrides) -> ChatParams:
            """
            Processes the request messages and overrides to create a new ChatParams object.
            This version avoids the multiple values error.
            """
            # Create a dictionary from the overrides
            overrides_dict = overrides.model_dump()
            
            # Get the prompt_template from overrides, or use the default
            # Then remove it from the dictionary so we don't pass it twice
            prompt_template = overrides_dict.pop("prompt_template", self.answer_prompt_template)
            
            past_messages = [m.model_dump() for m in messages[:-1]]

            return ChatParams(
                **overrides_dict,
                original_user_query=messages[-1].content,
                past_messages=past_messages,
                prompt_template=prompt_template,
                response_token_limit=1024,
                enable_text_search=overrides.retrieval_mode in ("text", "hybrid"),
                enable_vector_search=overrides.retrieval_mode in ("vectors", "hybrid"),
            )
    
    def prepare_rag_request(self, query: str, results: list[AbastecimentoPublic]) -> str:
        # CAMBIO: Adaptado para formatear los resultados de 'Abastecimento'
        sources = ""
        for i, result in enumerate(results):
            sources += (
                f"[doc{i+1}]\n"
                f"Placa: {result.placa}\n"
                f"Fecha: {result.data}\n"
                f"Costo Combustible: {result.custo_combustivel}\n"
                f"Eficiencia: {result.km_diesel}\n\n"
            )
        return self.answer_prompt_template.format(query=query, sources=sources)