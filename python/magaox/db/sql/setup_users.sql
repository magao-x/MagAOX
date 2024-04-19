DO $$
BEGIN
CREATE ROLE xsup WITH PASSWORD 'extremeAO!';
EXCEPTION WHEN duplicate_object THEN RAISE NOTICE '%, skipping', SQLERRM USING ERRCODE = SQLSTATE;
END
$$;
DO $$
BEGIN
CREATE ROLE xtelem WITH PASSWORD 'extremeAO!';
EXCEPTION WHEN duplicate_object THEN RAISE NOTICE '%, skipping', SQLERRM USING ERRCODE = SQLSTATE;
END
$$;
GRANT ALL PRIVILEGES ON DATABASE xtelem TO xsup;
GRANT ALL PRIVILEGES ON DATABASE xtelem TO xtelem;