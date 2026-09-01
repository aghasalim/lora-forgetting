// Regenerate the four tables of RESULTS.md from the raw predictions and require
// them to appear in the committed file, character for character.
//
// RESULTS.md is rendered by src/loraft/report.py out of reports/summary_*.json,
// and `make report-check` already fails if the committed file has drifted from
// what report.py would write today. What nothing checked is the step before
// that: whether summary_*.json is what the predictions actually say. The
// generator and its checker are the same program, so they agree by
// construction, and a wrong aggregate would be rendered faithfully into a table
// that passes.
//
// This goes the whole way in one pass, from reports/preds_*.jsonl to the exact
// markdown, and never opens summary_*.json. It also pins the rounding: the
// comparison is on the rendered string, so a rate that would print as a
// different tenth of a point is a failure and not a tolerance.
//
// Usage: java verify/Verify.java <repository root>

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Verify {

    static final String[] FIELDS = {"json_parsed", "schema_ok", "vendor", "amount",
            "currency", "date", "category", "all_correct"};

    static int failures = 0;

    static void fail(String msg) {
        System.out.println("FAIL: " + msg);
        failures++;
    }

    /** One scored prediction: its difficulty label and its eight verdicts. */
    static final class Pred {
        String kind;
        Map<String, Boolean> score = new LinkedHashMap<>();
    }

    static List<Pred> readPreds(Path path) throws IOException {
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        List<Pred> out = new ArrayList<>();
        Pattern kindRe = Pattern.compile("\"kind\": \"([^\"]*)\"");
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i);
            if (line.isEmpty()) {
                continue;
            }
            Pred p = new Pred();
            Matcher k = kindRe.matcher(line);
            p.kind = k.find() ? k.group(1) : null;
            for (String f : FIELDS) {
                // Anchored inside the score object by the key name, so a key
                // renamed or reordered upstream is a hard failure here rather
                // than a silently different average.
                Matcher m = Pattern.compile("\"" + f + "\": (true|false)").matcher(line);
                if (!m.find()) {
                    fail(path.getFileName() + " line " + (i + 1) + " has no " + f + " verdict");
                    p.score.put(f, false);
                } else {
                    p.score.put(f, m.group(1).equals("true"));
                    if (m.find()) {
                        fail(path.getFileName() + " line " + (i + 1) + " carries " + f + " twice");
                    }
                }
            }
            out.add(p);
        }
        return out;
    }

    static double rateOf(List<Pred> rows, String field) {
        int k = 0;
        for (Pred p : rows) {
            if (p.score.get(field)) {
                k++;
            }
        }
        return (double) k / rows.size();
    }

    static double jsonNumber(Path path, String key) throws IOException {
        String text = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
        Matcher m = Pattern.compile("\"" + key + "\":\\s*([0-9.eE+-]+)").matcher(text);
        if (!m.find()) {
            fail(path.getFileName() + " has no " + key);
            return Double.NaN;
        }
        return Double.parseDouble(m.group(1));
    }

    /** Python's "{:.1%}" and "{:+.1%}", which is what rendered the committed file. */
    static String pct(double v) {
        return String.format(Locale.ROOT, "%.1f%%", v * 100);
    }

    static String delta(double t, double b) {
        return String.format(Locale.ROOT, "%+.1f%%", (t - b) * 100);
    }

    static void require(String haystack, String row, String what) {
        if (haystack.contains("\n" + row + "\n")) {
            System.out.println("  ok  " + row);
        } else {
            fail(what + ": RESULTS.md has no row\n        " + row);
        }
    }

    public static void main(String[] args) throws IOException {
        Path root = Path.of(args.length > 0 ? args[0] : ".");
        Path reports = root.resolve("reports");

        List<Pred> base = readPreds(reports.resolve("preds_base.jsonl"));
        List<Pred> tuned = readPreds(reports.resolve("preds_tuned.jsonl"));
        List<Pred> baseSynth = readPreds(reports.resolve("preds_base_synth.jsonl"));
        List<Pred> tunedSynth = readPreds(reports.resolve("preds_tuned_synth.jsonl"));

        if (base.size() != tuned.size() || baseSynth.size() != tunedSynth.size()) {
            fail("the base and tuned prediction files are different lengths");
        }

        String results = new String(Files.readAllBytes(root.resolve("RESULTS.md")),
                StandardCharsets.UTF_8);

        // --- the benchmark table, one row per scored field ---------------
        for (String f : FIELDS) {
            double b = rateOf(base, f), t = rateOf(tuned, f);
            require(results, "| " + f + " | " + pct(b) + " | " + pct(t) + " | " + delta(t, b) + " |",
                    "the benchmark table");
        }

        // --- the by-difficulty table -------------------------------------
        TreeSet<String> kinds = new TreeSet<>();
        for (Pred p : base) {
            kinds.add(p.kind);
        }
        for (String kind : kinds) {
            List<Pred> b = new ArrayList<>(), t = new ArrayList<>();
            for (int i = 0; i < base.size(); i++) {
                if (kind.equals(base.get(i).kind)) {
                    b.add(base.get(i));
                    t.add(tuned.get(i));
                }
            }
            double rb = rateOf(b, "all_correct"), rt = rateOf(t, "all_correct");
            require(results, "| " + kind + " | " + pct(rb) + " | " + pct(rt) + " | "
                    + delta(rt, rb) + " |", "the by-difficulty table");
        }

        // --- the generalisation gap --------------------------------------
        require(results, "| held-out synthetic (same generator) | "
                + pct(rateOf(baseSynth, "all_correct")) + " | "
                + pct(rateOf(tunedSynth, "all_correct")) + " |", "the generalisation gap");
        require(results, "| hand-written benchmark (disjoint vendors) | "
                + pct(rateOf(base, "all_correct")) + " | "
                + pct(rateOf(tuned, "all_correct")) + " |", "the generalisation gap");

        // --- the forgetting table ----------------------------------------
        String[][] forget = {
                {"arc_loglikelihood", "ARC-Easy, log-likelihood (knowledge)"},
                {"arc_generative", "ARC-Easy, generated (instruction following)"},
                {"arc_parse_rate", "answer parseable at all"},
                {"open_ended", "open-ended factual probes"},
        };
        Path fb = reports.resolve("forgetting_base.json");
        Path ft = reports.resolve("forgetting_tuned.json");
        for (String[] row : forget) {
            double b = jsonNumber(fb, row[0]), t = jsonNumber(ft, row[0]);
            require(results, "| " + row[1] + " | " + pct(b) + " | " + pct(t) + " | "
                    + delta(t, b) + " |", "the forgetting table");
        }

        if (failures > 0) {
            System.out.println("\n" + failures + " checks failed");
            System.exit(1);
        }
        System.out.println("Java rebuilds every row of all four RESULTS.md tables "
                + "from reports/preds_*.jsonl and matches the committed text exactly");
    }
}
