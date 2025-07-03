-- Development database initialization script

-- Create development database if it doesn't exist
-- (This is already handled by POSTGRES_DB environment variable)

-- Create additional development users if needed
-- CREATE USER dr3am_dev WITH ENCRYPTED PASSWORD 'dr3am_dev_password';
-- GRANT ALL PRIVILEGES ON DATABASE dr3am_dev TO dr3am_dev;

-- Enable necessary PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for text search
-- These will be created by Alembic migrations, but included here for reference

-- Grant permissions for development
GRANT ALL ON SCHEMA public TO dr3am;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO dr3am;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO dr3am;