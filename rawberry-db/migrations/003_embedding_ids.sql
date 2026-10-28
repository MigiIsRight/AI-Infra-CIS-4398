-- Apply after 002_user_scoping_conversations.sql.
-- Additive migration: generate embedding IDs so ingestion can store chunk vectors.
BEGIN;
SELECT pg_advisory_xact_lock(4398, 1);
CREATE SEQUENCE IF NOT EXISTS embedding_api_id_seq OWNED BY embedding.id;
ALTER TABLE embedding ALTER COLUMN id SET DEFAULT nextval('embedding_api_id_seq');
SELECT setval('embedding_api_id_seq', GREATEST(
    COALESCE((SELECT MAX(id) FROM embedding), 0),
    (SELECT last_value FROM embedding_api_id_seq), 1));

INSERT INTO schema_migration(version) VALUES ('003_embedding_ids') ON CONFLICT DO NOTHING;
COMMIT;
