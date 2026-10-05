-- ------------------------telem_cooler--------------------------------
-- DO $$
-- DECLARE keys text;
-- BEGIN
--     DROP VIEW IF EXISTS telem_cooler;
--     SELECT string_agg(distinct format('msg->> %L as %I', jkey, jkey), ', ')
--         INTO keys
--         FROM telem, jsonb_object_keys("msg") as t(jkey)
--         WHERE ec = 'telem_cooler';
--     EXECUTE 'CREATE VIEW telem_cooler AS SELECT device, ts, prio, ec,' ||keys||' FROM telem WHERE ec = ''telem_cooler'' ';
-- END;
-- $$;

-- ---------------------------telem_fxngen------------------------------------
-- DO $$
-- DECLARE keys text;
-- BEGIN
--     DROP VIEW IF EXISTS telem_fxngen;
--     SELECT string_agg(DISTINCT format('msg ->> %L AS %I', jkey, jkey), ', ')
--         INTO keys
--         FROM telem, jsonb_object_keys("msg") as t(jkey)
--         WHERE ec = 'telem_fxngen';
--      EXECUTE 'CREATE VIEW telem_fxngen AS SELECT device, ts, prio, ec,' ||keys||' FROM telem WHERE ec = ''telem_fxngen'' ';

-- END;
-- $$;
-- --------------------------telem_observer-------------------------------------
-- DO $$
-- DECLARE keys text;
-- BEGIN
--     DROP VIEW IF EXISTS telem_observer cascade;
--     SELECT string_agg(DISTINCT format('msg ->> %L as %I', jkey, jkey), ', ')
--         INTO keys
--         FROM telem, jsonb_object_keys("msg") AS t(jkey)
--         WHERE ec = 'telem_observer';
--      EXECUTE 'CREATE VIEW telem_observer AS SELECT device, ts, prio, ec,' ||keys|| ' FROM telem WHERE ec = ''telem_observer''';
-- END;
-- $$;


CREATE VIEW observations AS
WITH
  obs AS (
    SELECT
    ts,
    (msg->>'observing')::bool as observing,
    msg->>'email' as email,
    msg->>'obsName' as obsName
    FROM telem
    WHERE
    device = 'observers' AND
    msg->>'obsName' != ''
    ORDER BY ts DESC
  ),
  edges AS (
    SELECT
      ts,
      observing,
      LAG(ts) OVER (ORDER BY ts DESC),
      LAG(observing) OVER (ORDER BY ts DESC) AS next_paired_observing,
      email,
      obsName
    FROM obs
  ),
  transitions AS (
        SELECT ts, email, obsName, next_paired_observing as observing FROM edges WHERE observing != next_paired_observing
    ),
  spans AS (
    SELECT
      ts,
      LAG(ts) OVER (ORDER BY ts DESC) AS next_edge_ts,
        email,
        obsName,
      observing
    FROM transitions
    )
SELECT
    ts as start_ts,
  next_edge_ts as end_ts,
  email,
  obsName
FROM spans
WHERE
    observing = true
ORDER BY ts DESC;