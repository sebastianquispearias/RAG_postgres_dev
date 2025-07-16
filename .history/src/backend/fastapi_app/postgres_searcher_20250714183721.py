from typing import Optional, Union, List, Any

import numpy as np
from openai import AsyncAzureOpenAI, AsyncOpenAI
from sqlalchemy import Float, Integer, String, column, select, text
from sqlalchemy.ext.asyncio import AsyncSession

# Importamos los modelos correctos
from fastapi_app.postgres_models import Abastecimento
from fastapi_app.embeddings import compute_text_embedding

class PostgresSearcher:
    def __init__(
        self,
        db_session: AsyncSession,
        openai_embed_client: Union[AsyncOpenAI, AsyncAzureOpenAI],
        embed_deployment: Optional[str],
        embed_model: str,
        embed_dimensions: Optional[int],
        embedding_column: str,
    ):
        self.db_session = db_session
        self.openai_embed_client = openai_embed_client
        self.embed_model = embed_model
        self.embed_deployment = embed_deployment
        self.embed_dimensions = embed_dimensions
        self.embedding_column = embedding_column

    def build_filter_clause(self, filters: Optional[List[dict]]) -> tuple[str, str]:
        """
        Construye la cláusula WHERE de SQL a partir de una lista de diccionarios de filtros.
        """
        if not filters:
            return "", ""
        
        filter_clauses = []
        for f in filters:
    for f in filters:
        # Si el filtro es un date_filter, lo manejamos de forma especial
        if 'start_date' in f.get('value', {}):
            date_filter = f['value']
            start_date = date_filter.get('start_date')
            end_date = date_filter.get('end_date')
            if start_date and end_date:
                filter_clauses.append(f"data BETWEEN '{start_date}' AND '{end_date}'")
        else:
            # Si no, es un filtro normal
            column_name = f['column']
            operator = f['operator']
            value = f['value']
            filter_value = f"'{value}'" if isinstance(value, str) else value
            filter_clauses.append(f"{column_name} {operator} {filter_value}")
        
        filter_clause_str = " AND ".join(filter_clauses)
        
        if filter_clause_str:
            return f"WHERE {filter_clause_str}", f"AND {filter_clause_str}"
        return "", ""

    async def search(
        self,
        query_text: Optional[str],
        query_vector: list[float],
        top: int = 5,
        filters: Optional[List[dict]] = None,
    ) -> list[Abastecimento]:
        filter_clause_where, filter_clause_and = self.build_filter_clause(filters)
        
        table_name = Abastecimento.__tablename__
        pk_column = "ctid" 

        vector_query = f"""
            SELECT {pk_column}, RANK () OVER (ORDER BY {self.embedding_column} <=> :embedding) AS rank
            FROM {table_name}
            {filter_clause_where}
            ORDER BY {self.embedding_column} <=> :embedding
            LIMIT 20
        """

        fulltext_query = f"""
            SELECT {pk_column}, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', placa), query) DESC)
            FROM {table_name}, plainto_tsquery('english', :query) query
            WHERE to_tsvector('english', placa) @@ query {filter_clause_and}
            ORDER BY ts_rank_cd(to_tsvector('english', placa), query) DESC
            LIMIT 20
        """

        hybrid_query = f"""
        WITH vector_search AS ({vector_query}),
        fulltext_search AS ({fulltext_query})
        SELECT
            COALESCE(vector_search.{pk_column}, fulltext_search.{pk_column}) AS {pk_column},
            COALESCE(1.0 / (:k + vector_search.rank), 0.0) +
            COALESCE(1.0 / (:k + fulltext_search.rank), 0.0) AS score
        FROM vector_search
        FULL OUTER JOIN fulltext_search ON vector_search.{pk_column} = fulltext_search.{pk_column}
        ORDER BY score DESC
        LIMIT 20
        """
        
        if query_text and query_vector:
            sql = text(hybrid_query).columns(column(pk_column), column("score", Float))
        elif query_vector:
            sql = text(vector_query).columns(column(pk_column), column("rank", Integer))
        elif query_text:
            sql = text(fulltext_query).columns(column(pk_column), column("rank", Integer))
        else:
            raise ValueError("Both query text and query vector are empty")
        
        results = (
            await self.db_session.execute(
                sql,
                {"embedding": np.array(query_vector), "query": query_text, "k": 60},
            )
        ).fetchall()

        row_models = []
        for row_data in results[:top]:
            ctid_value = row_data[0]
            query = select(Abastecimento).where(text(f"ctid = '{ctid_value}'"))
            abastecimento_obj = (await self.db_session.execute(query)).scalar_one_or_none()
            if abastecimento_obj:
                row_models.append(abastecimento_obj)
        return row_models

    async def search_and_embed(
        self,
        query_text: Optional[str] = None,
        top: int = 5,
        enable_vector_search: bool = False,
        enable_text_search: bool = False,
        filters: Optional[List[dict]] = None,
    ) -> list[Abastecimento]:
        vector: list[float] = []
        if enable_vector_search and query_text:
            vector = await compute_text_embedding(
                query_text,
                self.openai_embed_client,
                self.embed_model,
                self.embed_deployment,
                self.embed_dimensions,
            )
        
        text_query = query_text if enable_text_search else None

        return await self.search(text_query, vector, top, filters)