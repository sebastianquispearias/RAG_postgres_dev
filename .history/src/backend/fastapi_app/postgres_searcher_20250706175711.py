import os
from typing import Any, List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from openai import AsyncOpenAI, AsyncAzureOpenAI
from sqlalchemy import and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

# CAMBIO: Se usan los modelos correctos y consistentes
from fastapi_app.api_models import AnoFilter, Filter
from fastapi_app.postgres_models import Abastecimento, Veiculo


class PostgresSearcher:
    def __init__(
        self,
        db_session: AsyncSession,
        openai_embed_client: Union[AsyncOpenAI, AsyncAzureOpenAI],
        embed_deployment: str,
        embed_model: str,
        embed_dimensions: int,
        embedding_column: str,
    ):
        self.db_session = db_session
        self.openai_embed_client = openai_embed_client
        self.embed_deployment = embed_deployment
        self.embed_model = embed_model
        self.embed_dimensions = embed_dimensions
        self.embedding_column = embedding_column
        # Mapeo de nombres de columnas para la búsqueda
        self.searchable_columns = {
            "id_veiculo": Abastecimento.id_veiculo,
            "placa": Abastecimento.placa,
            "ano": Veiculo.ano,
            "fabricante": Veiculo.fabricante,
            "tipo_onibus": Veiculo.tipo_onibus,
        }

    async def search_and_embed(
        self,
        query_text: Optional[str],
        top: int = 5,
        filters: Optional[List[dict[str, Any]]] = None,
        enable_vector_search: bool = True,
        enable_text_search: bool = False,
    ) -> List[Any]:
        """
        Realiza la búsqueda vectorial y de texto completo.
        """
        # Genera el embedding para la pregunta del usuario
        query_embedding = (
            (
                await self.openai_embed_client.embeddings.create(
                    model=self.embed_deployment, input=query_text, dimensions=self.embed_dimensions
                )
            )
            .data[0]
            .embedding
        )

        # Construye las cláusulas WHERE a partir de los filtros
        where_clauses = self._build_where_clauses(filters)

        # Por ahora, la búsqueda se enfoca solo en 'abastecimento'
        # En el futuro, se podría hacer un JOIN con 'veiculos' para usar los filtros de año, etc.
        stmt = select(Abastecimento).where(and_(*where_clauses)).order_by(getattr(Abastecimento, self.embedding_column).l2_distance(query_embedding)).limit(top)

        results = await self.db_session.execute(stmt)
        return results.scalars().all()

    def _build_where_clauses(self, filters: Optional[List[dict[str, Any]]]) -> list:
        """
        Construye las cláusulas WHERE para la consulta SQL a partir de una lista de diccionarios.
        """
        where_clauses = []
        if not filters:
            return where_clauses

        for f in filters:
            # CORRECCIÓN: Se accede a los valores usando la sintaxis de diccionario (corchetes)
            column_name = f["column"]
            operator = f["operator"]
            value = f["value"]

            if column_name in self.searchable_columns:
                column = self.searchable_columns[column_name]
                # Construye la condición SQL dinámicamente
                if operator == "=":
                    where_clauses.append(column == value)
                elif operator == "!=":
                    where_clauses.append(column != value)
                elif operator == ">":
                    where_clauses.append(column > value)
                elif operator == "<":
                    where_clauses.append(column < value)
                elif operator == ">=":
                    where_clauses.append(column >= value)
                elif operator == "<=":
                    where_clauses.append(column <= value)
        
        return where_clauses