CREATE OR REPLACE FUNCTION public.observations_between(
  p_start timestamptz,
  p_end   timestamptz
)
RETURNS TABLE (start_ts timestamptz, end_ts timestamptz, email text, obsname text)
LANGUAGE sql
STABLE
AS $$
WITH obs AS (
  SELECT t.ts,
         (t.msg ->> 'observing')::boolean AS observing,
         t.msg ->> 'email'               AS email,
         t.msg ->> 'obsName'             AS obsname
  FROM telem t
  WHERE t.device = 'observers'
    AND t.msg ->> 'obsName' <> ''
    -- pushdown happens here:
    AND t.ts BETWEEN p_start AND p_end
),
edges AS (
  SELECT o.*,
         lag(o.observing) OVER (ORDER BY o.ts DESC) AS next_paired_observing
  FROM obs o
),
transitions AS (
  SELECT e.ts, e.email, e.obsname, e.next_paired_observing AS observing
  FROM edges e
  WHERE e.observing IS DISTINCT FROM e.next_paired_observing
),
spans AS (
  SELECT t.ts AS start_ts,
         lag(t.ts) OVER (ORDER BY t.ts DESC) AS end_ts,
         t.email, t.obsname, t.observing
  FROM transitions t
)
SELECT start_ts, end_ts, email, obsname
FROM spans
WHERE observing = true
  -- (optional redundancy; keeps semantics obvious)
  AND start_ts BETWEEN p_start AND p_end
  AND end_ts   BETWEEN p_start AND p_end
ORDER BY start_ts DESC;
$$;
