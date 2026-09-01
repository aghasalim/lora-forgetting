-- Recompute reports/summary_base.json and reports/summary_tuned.json from the
-- per-prediction files they were derived from.
--
-- The published rates come from task.aggregate() in Python, and every table in
-- the README and in RESULTS.md is rendered from that one dictionary. If the
-- averaging were wrong, nothing downstream would notice, because everything
-- downstream reads the same output. This redoes all four splits and the
-- by-difficulty breakdown in SQLite, straight from reports/preds_*.jsonl, and
-- prints a line for every value that disagrees.
--
-- Run from the repository root:
--   sqlite3 -init verify/summary.sql :memory: ""

CREATE TABLE p_base(line TEXT);
CREATE TABLE p_tuned(line TEXT);
CREATE TABLE p_base_synth(line TEXT);
CREATE TABLE p_tuned_synth(line TEXT);

-- Tabs mode, not CSV: a JSONL line is full of commas and quotes and the CSV
-- reader would tear it apart. The files contain no tab character, so one line
-- lands in one column.
.mode tabs
.import reports/preds_base.jsonl p_base
.import reports/preds_tuned.jsonl p_tuned
.import reports/preds_base_synth.jsonl p_base_synth
.import reports/preds_tuned_synth.jsonl p_tuned_synth

CREATE TEMP VIEW rows_all AS
              SELECT 'base_benchmark'  AS split, line FROM p_base
    UNION ALL SELECT 'tuned_benchmark',        line FROM p_tuned
    UNION ALL SELECT 'base_synthetic',         line FROM p_base_synth
    UNION ALL SELECT 'tuned_synthetic',        line FROM p_tuned_synth;

-- Long form, one row per (split, metric, prediction). Fields are pulled by
-- name out of the JSON, so a key added or reordered upstream cannot silently
-- shift what is averaged.
CREATE TEMP VIEW metric AS
              SELECT split, 'json_parsed' AS metric, json_extract(line,'$.score.json_parsed') AS v FROM rows_all
    UNION ALL SELECT split, 'schema_ok',  json_extract(line,'$.score.schema_ok')  FROM rows_all
    UNION ALL SELECT split, 'vendor',     json_extract(line,'$.score.vendor')     FROM rows_all
    UNION ALL SELECT split, 'amount',     json_extract(line,'$.score.amount')     FROM rows_all
    UNION ALL SELECT split, 'currency',   json_extract(line,'$.score.currency')   FROM rows_all
    UNION ALL SELECT split, 'date',       json_extract(line,'$.score.date')       FROM rows_all
    UNION ALL SELECT split, 'category',   json_extract(line,'$.score.category')   FROM rows_all
    UNION ALL SELECT split, 'all_correct',json_extract(line,'$.score.all_correct') FROM rows_all;

-- Where the published number for each split lives.
CREATE TEMP TABLE published(split TEXT, path TEXT, doc TEXT);
INSERT INTO published VALUES
    ('base_benchmark',  '$.benchmark', readfile('reports/summary_base.json')),
    ('tuned_benchmark', '$.benchmark', readfile('reports/summary_tuned.json')),
    ('base_synthetic',  '$.synthetic', readfile('reports/summary_base.json')),
    ('tuned_synthetic', '$.synthetic', readfile('reports/summary_tuned.json'));

-- Python rounds the rates to 4 places and the by-kind rates to 3, so the
-- comparison is on the rounded value rather than on a tolerance.
CREATE TEMP VIEW check_rates AS
    SELECT m.split || '.' || m.metric AS what,
           round(avg(m.v), 4) AS got,
           json_extract(p.doc, p.path || '.' || m.metric) AS want
    FROM metric m JOIN published p ON p.split = m.split
    GROUP BY m.split, m.metric;

CREATE TEMP VIEW check_counts AS
    SELECT r.split || '.n' AS what,
           CAST(count(*) AS REAL) AS got,
           CAST(json_extract(p.doc, p.path || '.n') AS REAL) AS want
    FROM rows_all r JOIN published p ON p.split = r.split
    GROUP BY r.split;

-- The by-difficulty table is all_correct grouped by the hand-labelled kind of
-- each benchmark case. Only the 45 hand-written cases carry a kind.
CREATE TEMP VIEW check_kinds AS
    SELECT k.label || '.by_kind.' || k.kind AS what, k.got,
           json_extract(k.doc, '$.by_kind.' || k.kind) AS want
    FROM (
        SELECT 'base' AS label, json_extract(line,'$.kind') AS kind,
               round(avg(json_extract(line,'$.score.all_correct')), 3) AS got,
               readfile('reports/summary_base.json') AS doc
        FROM p_base GROUP BY json_extract(line,'$.kind')
        UNION ALL
        SELECT 'tuned', json_extract(line,'$.kind'),
               round(avg(json_extract(line,'$.score.all_correct')), 3),
               readfile('reports/summary_tuned.json')
        FROM p_tuned GROUP BY json_extract(line,'$.kind')
    ) k;

CREATE TEMP VIEW all_checks AS
              SELECT * FROM check_rates
    UNION ALL SELECT * FROM check_counts
    UNION ALL SELECT * FROM check_kinds;

.mode list
.headers off
.separator " "
SELECT 'MISMATCH', what, 'recomputed', got, 'published', coalesce(want, 'absent')
FROM all_checks WHERE want IS NULL OR got IS NOT want;

SELECT count(*) || ' values recomputed from reports/preds_*.jsonl, '
       || sum(want IS NULL OR got IS NOT want) || ' mismatches'
FROM all_checks;
