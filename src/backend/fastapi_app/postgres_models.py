from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import (Index,Integer,String,Date,Numeric,PrimaryKeyConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from datetime import date
from decimal import Decimal

class Base(DeclarativeBase):
    pass

class Veiculo(Base):
    __tablename__ = "veiculos"

    id_veiculo = mapped_column(String, primary_key=True)
    garagem = mapped_column(String, nullable=True)
    placa = mapped_column(String, nullable=True)
    ano = mapped_column(Integer, nullable=True)
    tipo_onibus = mapped_column(String, nullable=True)
    fabricante = mapped_column(String, nullable=True)
    modelo_chassi = mapped_column(String, nullable=True)

    embedding_main = mapped_column(Vector(1024), nullable=True)
    embedding_alt = mapped_column(Vector(768), nullable=True)

    def to_str_for_embedding(self) -> str:
        """
        FOR SEARCH (The Archivist):
        Text rich in keywords and concepts to find this row.
        """
        return f"Vehicle type {self.tipo_onibus} from brand {self.fabricante} {self.modelo_chassi} from year {self.ano}."

    def to_str_for_rag(self) -> str:
        """
        FOR THE RESPONSE (The Final Redactor):
        Clean and clear text for the LLM to use as a source of truth.
        """
        return (f"Vehicle ID: {self.id_veiculo}, Plate: {self.placa}, Manufacturer: {self.fabricante}, "
                f"Model: {self.modelo_chassi}, Year: {self.ano}, Type: {self.tipo_onibus}, Garage: {self.garagem}.")


class Abastecimento(Base):
    __tablename__ = "abastecimento"

    id_veiculo = mapped_column(String)
    placa = mapped_column(String)
    km_percorrido = mapped_column(Integer)
    diesel = mapped_column(Numeric)
    km_diesel = mapped_column(Numeric)
    data = mapped_column(Date)
    custo_combustivel = mapped_column(Numeric)
    preco_combustivel = mapped_column(Numeric)

    embedding_main = mapped_column(Vector(1024), nullable=True)
    embedding_alt = mapped_column(Vector(768), nullable=True)

    # Composite primary 
    __table_args__ = (
        PrimaryKeyConstraint('id_veiculo', 'data', 'km_percorrido', 'diesel', name='abastecimento_pk'),
    )

    def to_str_for_embedding(self) -> str:
            """
            FOR SEARCH (The Archivist):
            Describes the event with potential keywords like "anomaly", "high cost".
            """
            anomaly_text = ""
            # CORRECCIÓN: Comparamos Decimal con Decimal
            if self.km_diesel is not None and self.km_diesel < Decimal('1.0'):
                anomaly_text += " Potential low fuel efficiency anomaly."
            # CORRECCIÓN: Comparamos Decimal con Decimal
            if self.custo_combustivel is not None and self.custo_combustivel > Decimal('1000'):
                anomaly_text += " High total fueling cost."
            return f"Refueling record for vehicle plate {self.placa} on {self.data}. Efficiency: {self.km_diesel} km/l.{anomaly_text}"
    def to_str_for_rag(self) -> str:
        """
        FOR THE RESPONSE (The Final Redactor):
        Presents the event data clearly and directly.
        """
        return (f"Record from {self.data} for plate {self.placa}: {self.diesel} liters of diesel "
                f"cost {self.custo_combustivel}. The efficiency was {self.km_diesel} km/l.")

# Indexes 
index_veiculos_main = Index("hnsw_veiculos_main", Veiculo.embedding_main, postgresql_using="hnsw", postgresql_with={"m": 16, "ef_construction": 64}, postgresql_ops={"embedding_main": "vector_cosine_ops"})
index_veiculos_alt = Index("hnsw_veiculos_alt", Veiculo.embedding_alt, postgresql_using="hnsw", postgresql_with={"m": 16, "ef_construction": 64}, postgresql_ops={"embedding_alt": "vector_cosine_ops"})

index_abastecimento_main = Index("hnsw_abastecimento_main", Abastecimento.embedding_main, postgresql_using="hnsw", postgresql_with={"m": 16, "ef_construction": 64}, postgresql_ops={"embedding_main": "vector_cosine_ops"})
index_abastecimento_alt = Index("hnsw_abastecimento_alt", Abastecimento.embedding_alt, postgresql_using="hnsw", postgresql_with={"m": 16, "ef_construction": 64}, postgresql_ops={"embedding_alt": "vector_cosine_ops"})