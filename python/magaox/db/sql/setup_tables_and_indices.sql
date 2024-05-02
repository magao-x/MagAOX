CREATE TABLE IF NOT EXISTS telem (
    ts TIMESTAMPTZ,
    device VARCHAR(50),
    msg JSONB,
    prio VARCHAR(8),
    ec VARCHAR(20),
    PRIMARY KEY (ts, device)
);

CREATE INDEX IF NOT EXISTS telem_device_ts ON telem (device, ts);
CREATE INDEX IF NOT EXISTS observation_spans_fixup ON telem (ts, (msg->>'obsName')) WHERE device = 'observers' AND (msg->>'obsName' != '');

-- CREATE TABLE IF NOT EXISTS logs (
--     ts TIMESTAMPTZ,
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
    modification_time TIMESTAMPTZ,
    creation_time TIMESTAMPTZ,
    size_bytes BIGINT,
    PRIMARY KEY (origin_host, origin_path)
);

CREATE TABLE IF NOT EXISTS file_ingest_times (
    ts TIMESTAMPTZ,
    device VARCHAR(50),
    ingested_at TIMESTAMPTZ,
    origin_host VARCHAR(50),
    origin_path VARCHAR(1024),
    UNIQUE (ts, device),
    FOREIGN KEY (origin_host, origin_path) REFERENCES file_origins (origin_host, origin_path)
);

CREATE TABLE IF NOT EXISTS file_replicas (
    hostname VARCHAR(50),
    replica_path VARCHAR(1024),
    origin_modification_time TIMESTAMPTZ,  -- intentionally denormalized because replicas may lag behind originals
    size_bytes BIGINT,
    origin_host VARCHAR(50),
    origin_path VARCHAR(1024),
    FOREIGN KEY (origin_host, origin_path) REFERENCES file_origins (origin_host, origin_path)
);
