// Structural validation of everything under reports/, and a check that the
// percentages printed in README.md and RESULTS.md are the ones in those files.
//
// Two different failures are covered here, neither of which any test caught.
// The first is a malformed results file: a truncated write, a column that
// drifted, a NaN out of a division, a prediction file that lost a line. Every
// number in this repository is read out of these files, so a broken one is
// invisible until someone reads a table. The second is drift between the
// documents and the data. RESULTS.md is generated, but the README is written by
// hand, and until this ran nothing compared its percentages to reports/.
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// A printed percentage carries one decimal, so it can sit half a tenth of a
// point away from the exact rate and still be the correct rendering.
const printTol = 0.05

var scoreKeys = []string{"json_parsed", "schema_ok", "vendor", "amount",
	"currency", "date", "category", "all_correct"}

var goldKeys = []string{"vendor", "amount", "currency", "date", "category"}

type problem struct {
	where string
	what  string
}

var problems []problem

func fail(where, format string, a ...interface{}) {
	problems = append(problems, problem{where, fmt.Sprintf(format, a...)})
}

// --- structural validation ------------------------------------------------

func checkCSV(path string) int {
	f, err := os.Open(path)
	if err != nil {
		fail(path, "unreadable: %v", err)
		return 0
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		fail(path, "not a rectangular CSV: %v", err)
		return 0
	}
	if len(rows) < 2 {
		fail(path, "only %d rows", len(rows))
		return 0
	}
	header, body := rows[0], rows[1:]

	seen := map[string]bool{}
	for _, h := range header {
		if strings.TrimSpace(h) == "" {
			fail(path, "a column has an empty name")
		}
		if seen[h] {
			fail(path, "duplicate column %q", h)
		}
		seen[h] = true
	}

	for i, row := range body {
		for j, cell := range row {
			low := strings.ToLower(strings.TrimSpace(cell))
			if low == "nan" || low == "inf" || low == "-inf" || low == "" {
				fail(path, "row %d column %s is %q", i+2, header[j], cell)
				continue
			}
			v, err := strconv.ParseFloat(low, 64)
			if err != nil {
				fail(path, "row %d column %s is not a number: %q", i+2, header[j], cell)
			} else if math.IsNaN(v) || math.IsInf(v, 0) {
				fail(path, "row %d column %s is %v", i+2, header[j], v)
			}
		}
	}
	return len(body)
}

// checkPreds walks one predictions file and returns how many records it holds.
func checkPreds(path string) int {
	text, err := os.ReadFile(path)
	if err != nil {
		fail(path, "unreadable: %v", err)
		return 0
	}
	lines := strings.Split(strings.TrimRight(string(text), "\n"), "\n")
	n := 0
	for i, line := range lines {
		if strings.TrimSpace(line) == "" {
			fail(path, "line %d is blank", i+1)
			continue
		}
		var rec map[string]json.RawMessage
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			fail(path, "line %d is not JSON: %v", i+1, err)
			continue
		}
		for _, k := range []string{"received", "text", "gold", "raw", "score"} {
			if _, ok := rec[k]; !ok {
				fail(path, "line %d has no %q", i+1, k)
			}
		}
		var score map[string]interface{}
		if err := json.Unmarshal(rec["score"], &score); err != nil {
			fail(path, "line %d has an unreadable score: %v", i+1, err)
		} else {
			if len(score) != len(scoreKeys) {
				fail(path, "line %d scores %d fields, expected %d",
					i+1, len(score), len(scoreKeys))
			}
			for _, k := range scoreKeys {
				v, ok := score[k]
				if !ok {
					fail(path, "line %d has no score.%s", i+1, k)
					continue
				}
				if _, ok := v.(bool); !ok {
					fail(path, "line %d score.%s is %T, not a boolean", i+1, k, v)
				}
			}
		}
		var gold map[string]interface{}
		if err := json.Unmarshal(rec["gold"], &gold); err != nil {
			fail(path, "line %d has an unreadable gold record: %v", i+1, err)
		} else {
			if len(gold) != len(goldKeys) {
				fail(path, "line %d has %d gold fields, expected %d",
					i+1, len(gold), len(goldKeys))
			}
			for _, k := range goldKeys {
				if _, ok := gold[k]; !ok {
					fail(path, "line %d gold has no %q", i+1, k)
				}
			}
			if amt, ok := gold["amount"].(float64); ok {
				if math.IsNaN(amt) || math.IsInf(amt, 0) {
					fail(path, "line %d gold amount is %v", i+1, amt)
				}
			}
		}
		n++
	}
	return n
}

func readJSON(path string) map[string]interface{} {
	text, err := os.ReadFile(path)
	if err != nil {
		fail(path, "unreadable: %v", err)
		return nil
	}
	var doc map[string]interface{}
	if err := json.Unmarshal(text, &doc); err != nil {
		fail(path, "not JSON: %v", err)
		return nil
	}
	return doc
}

func rate(doc map[string]interface{}, path ...string) (float64, bool) {
	var cur interface{} = doc
	for _, key := range path {
		m, ok := cur.(map[string]interface{})
		if !ok {
			return 0, false
		}
		cur, ok = m[key]
		if !ok {
			return 0, false
		}
	}
	v, ok := cur.(float64)
	return v, ok
}

func checkRates(name string, doc map[string]interface{}, block string) {
	for _, k := range scoreKeys {
		v, ok := rate(doc, block, k)
		if !ok {
			fail(name, "no %s.%s", block, k)
			continue
		}
		if v < 0 || v > 1 || math.IsNaN(v) {
			fail(name, "%s.%s is %v, not a rate", block, k, v)
		}
	}
}

// --- the documents --------------------------------------------------------

var cell = regexp.MustCompile(`^\s*\|(.*)\|\s*$`)

// docTables splits a document into markdown tables: each table is the maximal
// run of consecutive table lines, cells trimmed of the bold markers the README
// uses. Tables are kept separate because a label is only unique inside one:
// "currency" is a field in the benchmark table and a difficulty class in the
// by-kind table, with different numbers behind it.
func docTables(text string) [][][]string {
	var tables [][][]string
	var current [][]string
	flush := func() {
		if len(current) > 0 {
			tables = append(tables, current)
			current = nil
		}
	}
	for _, line := range strings.Split(text, "\n") {
		m := cell.FindStringSubmatch(line)
		if m == nil {
			flush()
			continue
		}
		parts := strings.Split(m[1], "|")
		row := make([]string, 0, len(parts))
		for _, p := range parts {
			row = append(row, strings.TrimSpace(strings.ReplaceAll(p, "*", "")))
		}
		current = append(current, row)
	}
	flush()
	return tables
}

// tableWith returns the first table holding a row for every label asked for.
func tableWith(tables [][][]string, checks []docCheck) [][]string {
	for _, t := range tables {
		all := true
		for _, c := range checks {
			found := false
			for _, row := range t {
				if len(row) >= 3 && row[0] == c.label {
					found = true
					break
				}
			}
			if !found {
				all = false
				break
			}
		}
		if all {
			return t
		}
	}
	return nil
}

// percent parses "93.3%", "100%", "+28.9", "-0.7" and the typographic minus
// the README uses in one cell.
func percent(s string) (float64, bool) {
	s = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(s), "%"))
	s = strings.ReplaceAll(s, "−", "-")
	s = strings.TrimPrefix(s, "+")
	if s == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(s, 64)
	return v, err == nil
}

type docCheck struct {
	label string   // the row label as printed
	path  []string // where the rate lives inside the summary documents
}

// checkTable finds a row by label and compares its base, tuned and delta cells
// against the two summary documents.
func checkTable(doc, name string, tables [][][]string, checks []docCheck,
	base, tuned map[string]interface{}, checked *int) {
	rows := tableWith(tables, checks)
	if rows == nil {
		fail(doc, "%s no longer holds a row for every label this checks", name)
		return
	}
	for _, c := range checks {
		found := false
		for _, row := range rows {
			if len(row) < 3 || row[0] != c.label {
				continue
			}
			found = true
			b, okB := rate(base, c.path...)
			t, okT := rate(tuned, c.path...)
			if !okB || !okT {
				fail(doc, "%s: no %s in the summaries", name, strings.Join(c.path, "."))
				break
			}
			printedB, ok1 := percent(row[1])
			printedT, ok2 := percent(row[2])
			if !ok1 || !ok2 {
				fail(doc, "%s row %q: cannot read %q and %q", name, c.label, row[1], row[2])
				break
			}
			if math.Abs(printedB-100*b) > printTol {
				fail(doc, "%s row %q base says %.1f%%, reports/ says %.2f%%",
					name, c.label, printedB, 100*b)
			}
			if math.Abs(printedT-100*t) > printTol {
				fail(doc, "%s row %q fine-tuned says %.1f%%, reports/ says %.2f%%",
					name, c.label, printedT, 100*t)
			}
			*checked += 2
			if len(row) >= 4 {
				if d, ok := percent(row[3]); ok {
					if math.Abs(d-100*(t-b)) > printTol {
						fail(doc, "%s row %q delta says %+.1f, reports/ says %+.2f",
							name, c.label, d, 100*(t-b))
					}
					*checked++
				} else if strings.TrimSpace(row[3]) != "" {
					fail(doc, "%s row %q: cannot read the delta %q", name, c.label, row[3])
				}
			}
			break
		}
		if !found {
			fail(doc, "%s has no row %q any more", name, c.label)
		}
	}
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()
	reports := filepath.Join(*root, "reports")

	csvs, _ := filepath.Glob(filepath.Join(reports, "*.csv"))
	preds, _ := filepath.Glob(filepath.Join(reports, "*.jsonl"))
	jsons, _ := filepath.Glob(filepath.Join(reports, "*.json"))
	sort.Strings(csvs)
	sort.Strings(preds)
	sort.Strings(jsons)
	if len(csvs) == 0 || len(preds) == 0 || len(jsons) == 0 {
		fmt.Fprintf(os.Stderr, "nothing to validate under %s\n", reports)
		os.Exit(2)
	}

	rows := 0
	for _, p := range csvs {
		rows += checkCSV(p)
	}
	counts := map[string]int{}
	for _, p := range preds {
		n := checkPreds(p)
		counts[filepath.Base(p)] = n
		rows += n
	}
	for _, p := range jsons {
		if readJSON(p) == nil {
			continue
		}
		rows++
	}
	// Counts only. What was checked is stated at the end, and only if it held,
	// so a run that found a problem does not print a clean bill above it.
	fmt.Printf("  %d files under reports/, %d rows read, %d score fields per "+
		"prediction\n", len(csvs)+len(preds)+len(jsons), rows, len(scoreKeys))

	base := readJSON(filepath.Join(reports, "summary_base.json"))
	tuned := readJSON(filepath.Join(reports, "summary_tuned.json"))
	fbase := readJSON(filepath.Join(reports, "forgetting_base.json"))
	ftuned := readJSON(filepath.Join(reports, "forgetting_tuned.json"))
	if base == nil || tuned == nil || fbase == nil || ftuned == nil {
		report()
		os.Exit(1)
	}
	checkRates("summary_base.json", base, "benchmark")
	checkRates("summary_base.json", base, "synthetic")
	checkRates("summary_tuned.json", tuned, "benchmark")
	checkRates("summary_tuned.json", tuned, "synthetic")

	// The n each summary claims has to be the number of predictions actually
	// on disk, otherwise every rate in it is divided by the wrong number.
	for _, c := range []struct {
		doc   map[string]interface{}
		name  string
		block string
		file  string
	}{
		{base, "summary_base.json", "benchmark", "preds_base.jsonl"},
		{base, "summary_base.json", "synthetic", "preds_base_synth.jsonl"},
		{tuned, "summary_tuned.json", "benchmark", "preds_tuned.jsonl"},
		{tuned, "summary_tuned.json", "synthetic", "preds_tuned_synth.jsonl"},
	} {
		n, ok := rate(c.doc, c.block, "n")
		if !ok {
			fail(c.name, "no %s.n", c.block)
			continue
		}
		if int(n) != counts[c.file] {
			fail(c.name, "%s.n is %d but %s holds %d predictions",
				c.block, int(n), c.file, counts[c.file])
		}
	}

	// The forgetting files are aggregates over n_mc multiple-choice items, so
	// every rate in them has to be a whole number of items.
	for name, doc := range map[string]map[string]interface{}{
		"forgetting_base.json": fbase, "forgetting_tuned.json": ftuned} {
		nmc, ok := rate(doc, "n_mc")
		if !ok || nmc <= 0 {
			fail(name, "no usable n_mc")
			continue
		}
		for _, k := range []string{"arc_loglikelihood", "arc_generative",
			"arc_parse_rate", "open_ended"} {
			v, ok := rate(doc, k)
			if !ok {
				fail(name, "no %s", k)
				continue
			}
			if v < 0 || v > 1 {
				fail(name, "%s is %v, not a rate", k, v)
			}
			// The published rate is rounded to four places, so the count it
			// implies must be within half a rounding step of an integer.
			k1 := v * nmc
			if math.Abs(k1-math.Round(k1)) > 0.5e-4*nmc+1e-9 {
				fail(name, "%s is %v, which is not %d/%d for any whole number",
					k, v, int(math.Round(k1)), int(nmc))
			}
		}
	}

	checked := 0
	readme, err := os.ReadFile(filepath.Join(*root, "README.md"))
	if err != nil {
		fail("README.md", "unreadable: %v", err)
	} else {
		tables := docTables(string(readme))
		checkTable("README.md", "the target task table", tables, []docCheck{
			{"valid JSON", []string{"benchmark", "json_parsed"}},
			{"every field correct", []string{"benchmark", "all_correct"}},
			{"date", []string{"benchmark", "date"}},
			{"category", []string{"benchmark", "category"}},
		}, base, tuned, &checked)
		checkTable("README.md", "the generalisation gap table", tables, []docCheck{
			{"held-out synthetic (same generator as training)",
				[]string{"synthetic", "all_correct"}},
			{"hand-written benchmark (disjoint vendors, messier)",
				[]string{"benchmark", "all_correct"}},
		}, base, tuned, &checked)
		checkTable("README.md", "the forgetting table", tables, []docCheck{
			{"ARC-Easy, log-likelihood (knowledge)", []string{"arc_loglikelihood"}},
			{"ARC-Easy, generated answer (instruction following)",
				[]string{"arc_generative"}},
			{"answer parseable at all", []string{"arc_parse_rate"}},
			{"open-ended factual probes", []string{"open_ended"}},
		}, fbase, ftuned, &checked)
	}

	results, err := os.ReadFile(filepath.Join(*root, "RESULTS.md"))
	if err != nil {
		fail("RESULTS.md", "unreadable: %v", err)
	} else {
		tables := docTables(string(results))
		var benchmark []docCheck
		for _, k := range scoreKeys {
			benchmark = append(benchmark, docCheck{k, []string{"benchmark", k}})
		}
		checkTable("RESULTS.md", "the benchmark table", tables, benchmark, base, tuned, &checked)

		kinds, _ := base["by_kind"].(map[string]interface{})
		var byKind []docCheck
		names := make([]string, 0, len(kinds))
		for k := range kinds {
			names = append(names, k)
		}
		sort.Strings(names)
		for _, k := range names {
			byKind = append(byKind, docCheck{k, []string{"by_kind", k}})
		}
		checkTable("RESULTS.md", "the by-difficulty table", tables, byKind, base, tuned, &checked)

		checkTable("RESULTS.md", "the forgetting table", tables, []docCheck{
			{"ARC-Easy, log-likelihood (knowledge)", []string{"arc_loglikelihood"}},
			{"ARC-Easy, generated (instruction following)", []string{"arc_generative"}},
			{"answer parseable at all", []string{"arc_parse_rate"}},
			{"open-ended factual probes", []string{"open_ended"}},
		}, fbase, ftuned, &checked)
	}
	fmt.Printf("  %d printed percentages in README.md and RESULTS.md compared "+
		"to reports/ at %.2f points\n", checked, printTol)

	report()
	if len(problems) > 0 {
		os.Exit(1)
	}
	fmt.Println("Go finds no ragged CSV, no duplicate column, no NaN or Inf, " +
		"and every published percentage backed by the data")
}

func report() {
	if len(problems) == 0 {
		return
	}
	fmt.Printf("\n%d structural problems:\n", len(problems))
	for i, p := range problems {
		if i == 40 {
			fmt.Printf("  ... and %d more\n", len(problems)-40)
			break
		}
		fmt.Printf("  %s: %s\n", filepath.Base(p.where), p.what)
	}
}
