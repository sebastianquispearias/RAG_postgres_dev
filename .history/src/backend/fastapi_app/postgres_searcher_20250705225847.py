from typing import Optional, Union

import numpy as np
from openai import AsyncAzureOpenAI, AsyncOpenAI
from sqlalchemy import Float, Integer, column, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi_app.api_models import Filter
from fastapi_app.embeddings import compute_text_embedding
from fastapi_app.postgres_models import Abastecimento

class PostgresSearcher:
    def __init__(
        self,
        db_session: AsyncSession,
        openai_embed_client: Union[AsyncOpenAI, AsyncAzureOpenAI],
        embed_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
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

    def build_filter_clause(self, filters: Optional[list[Filter]]) -> tuple[str, str]:
        if filters is None:
            return "", ""
        filter_clauses = []
        for filter in filters:
            filter_value = f"'{filter.value}'" if isinstance(filter.value, str) else filter.value
            filter_clauses.append(f"{filter.column} {filter.comparison_operator} {filter_value}")
        filter_clause = " AND ".join(filter_clauses)
        if len(filter_clause) > 0:
            return f"WHERE {filter_clause}", f"AND {filter_clause}"
        return "", ""

    async def search(
        self,
        query_text: Optional[str],
        query_vector: list[float],
        top: int = 5,
        filters: Optional[list[Filter]] = None,
    ):
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

        # CAMBIO: La lógica híbrida ahora usa la columna correcta (pk_column)
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

        # Convert results to SQLAlchemy models
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
        filters: Optional[list[Filter]] = None,
    ) -> list[Abastecimento]:
        """
        Search rows by query text. Optionally converts the query text to a vector if enable_vector_search is True.
        """
        vector: list[float] = []
        if enable_vector_search and query_text is not None:
            vector = await compute_text_embedding(
                query_text,
                self.openai_embed_client,
                self.embed_model,
                self.embed_deployment,
                self.embed_dimensions,
            )
        if not enable_text_search:
            query_text = None

        return await self.search(query_text, vector, top, filters)
