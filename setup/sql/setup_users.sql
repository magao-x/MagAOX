DO $$
BEGIN
CREATE ROLE xsup WITH LOGIN PASSWORD 'extremeAO!';
EXCEPTION WHEN duplicate_object THEN RAISE NOTICE '%, skipping', SQLERRM USING ERRCODE = SQLSTATE;
END
$$;
DO $$
BEGIN
CREATE ROLE xtelem WITH LOGIN PASSWORD 'extremeAO!';
EXCEPTION WHEN duplicate_object THEN RAISE NOTICE '%, skipping', SQLERRM USING ERRCODE = SQLSTATE;
END
$$;
GRANT pg_read_all_data TO xsup;
GRANT ALL PRIVILEGES ON DATABASE xtelem TO xtelem;
