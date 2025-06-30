-- prepare_db.sql
-- This script prepares the database for the RAG application.
-- It's safe to run this script multiple times.

-- Step 1: Enable the pgvector extension if it's not already enabled.
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: Add embedding columns to the 'veiculos' table if they don't exist.
ALTER TABLE public.veiculos ADD COLUMN IF NOT EXISTS embedding_main vector(1024);
ALTER TABLE public.veiculos ADD COLUMN IF NOT EXISTS embedding_alt vector(768);

-- Step 3: Add embedding columns to the 'abastecimento' table if they don't exist.
ALTER TABLE public.abastecimento ADD COLUMN IF NOT EXISTS embedding_main vector(1024);
ALTER TABLE public.abastecimento ADD COLUMN IF NOT EXISTS embedding_alt vector(768);

-- End of script.