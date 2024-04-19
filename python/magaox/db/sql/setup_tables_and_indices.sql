CREATE TABLE IF NOT EXISTS telem (
    ts TIMESTAMP,
    device VARCHAR(50),
    msg JSONB,
    prio VARCHAR(8),
    ec VARCHAR(20),
    PRIMARY KEY (ts, device, msg)
);

CREATE INDEX IF NOT EXISTS telem_device_ts ON telem (device, ts);

-- CREATE TABLE IF NOT EXISTS logs (
--     ts TIMESTAMP,
--     device VARCHAR(50),
--     msg JSONB,
--     prio VARCHAR(8),
--     ec VARCHAR(20),
--     PRIMARY KEY(ts, device, msg)
-- );

-- CREATE INDEX IF NOT EXISTS logs_device_ts ON logs (device, ts);

CREATE TABLE IF NOT EXISTS file_origins (
    origin_host VARCHAR(50),
    origin_path VARCHAR(1024),
    modification_time TIMESTAMP,
    creation_time TIMESTAMP,
    size_bytes BIGINT,
    PRIMARY KEY (origin_host, origin_path)
);

CREATE TABLE IF NOT EXISTS file_ingest_times (
    ts TIMESTAMP,
    device VARCHAR(50),
    ingested_at TIMESTAMP,
    origin_host VARCHAR(50),
    origin_path VARCHAR(1024),
    UNIQUE (ts, device),
    FOREIGN KEY (origin_host, origin_path) REFERENCES file_origins (origin_host, origin_path)
);

CREATE TABLE IF NOT EXISTS file_replicas (
    hostname VARCHAR(50),
    replica_path VARCHAR(1024),
    origin_modification_time TIMESTAMP,  -- intentionally denormalized because replicas may lag behind originals
    size_bytes BIGINT,
    origin_host VARCHAR(50),
    origin_path VARCHAR(1024),
    FOREIGN KEY (origin_host, origin_path) REFERENCES file_origins (origin_host, origin_path)
);
