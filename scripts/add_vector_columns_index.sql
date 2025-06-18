ALTER TABLE public.abastecimento     ADD COLUMN embedding_3l vector(1024);
ALTER TABLE public.manutencao       ADD COLUMN embedding_3l vector(1024);
ALTER TABLE public.quilometragem    ADD COLUMN embedding_3l vector(1024);
ALTER TABLE public.retorno_socorro  ADD COLUMN embedding_3l vector(1024);
ALTER TABLE public.veiculos         ADD COLUMN embedding_3l vector(1024);

CREATE INDEX ON public.abastecimento USING hnsw (embedding_3l vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Repite para cada tabla:
-- manutencao, quilometragem, retorno_socorro, veiculos
