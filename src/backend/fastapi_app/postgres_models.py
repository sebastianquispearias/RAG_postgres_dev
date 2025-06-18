from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from datetime import date

# Define the models
class Base(DeclarativeBase):
    pass


class Item(Base):
    __tablename__ = "items"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column()
    brand: Mapped[str] = mapped_column()
    name: Mapped[str] = mapped_column()
    description: Mapped[str] = mapped_column()
    price: Mapped[float] = mapped_column()
    # Embeddings for different models:
    embedding_3l: Mapped[Vector] = mapped_column(Vector(1024), nullable=True)  # text-embedding-3-large
    embedding_nomic: Mapped[Vector] = mapped_column(Vector(768), nullable=True)  # nomic-embed-text

    def to_dict(self, include_embedding: bool = False):
        model_dict = {column.name: getattr(self, column.name) for column in self.__table__.columns}
        if include_embedding:
            model_dict["embedding_3l"] = model_dict.get("embedding_3l", [])
            model_dict["embedding_nomic"] = model_dict.get("embedding_nomic", [])
        else:
            del model_dict["embedding_3l"]
            del model_dict["embedding_nomic"]
        return model_dict

    def to_str_for_rag(self):
        return f"Name:{self.name} Description:{self.description} Price:{self.price} Brand:{self.brand} Type:{self.type}"

    def to_str_for_embedding(self):
        return f"Name: {self.name} Description: {self.description} Type: {self.type}"


class Abastecimento(Base):
    __tablename__ = "abastecimentos"

    numero_veiculo: Mapped[str]  = mapped_column("Numero Veiculo", primary_key=True)
    placa:            Mapped[str]  = mapped_column("Placa")
    tipo_de_veiculo:  Mapped[str]  = mapped_column("Tipo de Veiculo")
    data:             Mapped[date] = mapped_column("Data")
    km_percorrido:    Mapped[float]= mapped_column("Km Percorrido")
    diesel:           Mapped[float]= mapped_column("Diesel")
    custo_combustivel:Mapped[float]= mapped_column("Custo Combustivel")
    embedding:        Mapped[Vector]= mapped_column(Vector(1536), nullable=True)

    def to_str_for_rag(self):
        return (
            f"Vehículo {self.numero_veiculo} placa {self.placa} "
            f"el {self.data}, cargó {self.diesel}L por {self.custo_combustivel}"
        )
"""
**Define HNSW index to support vector similarity search**

We use the vector_cosine_ops access method (cosine distance)
 since it works for both normalized and non-normalized vector embeddings
If you know your embeddings are normalized,
 you can switch to inner product for potentially better performance.
The index operator should match the operator used in queries.
"""

table_name = Item.__tablename__

index_3l = Index(
    f"hnsw_index_for_cosine_{table_name}_embedding_3l",
    Item.embedding_3l,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding_3l": "vector_cosine_ops"},
)

index_nomic = Index(
    f"hnsw_index_for_cosine_{table_name}_embedding_nomic",
    Item.embedding_nomic,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding_nomic": "vector_cosine_ops"},
)
