# Check the counting claims the README makes in prose, not in a table.
#
# The tables in README.md and RESULTS.md are covered elsewhere: gocheck reads
# every printed percentage back to reports/, and the Java check regenerates the
# whole of RESULTS.md from the predictions. Neither of them looks at a sentence.
# Section 3 in particular carries numbers that appear in no table at all, "32/45
# to 41/45", "fixed 12 cases and broke 3", "three are the same failure", and
# those are exactly the kind of number that gets written once from a notebook
# and then never touched again when the run is repeated.
#
# Each claim is located in the README by pattern rather than hard coded here, so
# rewording the sentence fails loudly instead of leaving a check that quietly
# matches nothing. The recomputation is from reports/preds_*.jsonl and
# reports/forgetting_*.json, never from summary_*.json, so it does not inherit
# the aggregation it is supposed to be checking.
#
# Usage: ruby verify/claims.rb <repository root>

require "json"

root = ARGV[0] || "."
$failures = 0
$checked = 0

def fail(msg)
  puts "FAIL: #{msg}"
  $failures += 1
end

# Ruby 2.6 opens files as US-ASCII, and this README contains a typographic
# minus. Without the explicit encoding, reading it raises rather than failing to
# match, which looks like a broken check instead of a broken claim.
def read_text(path)
  File.read(path, encoding: "UTF-8")
end

def preds(root, name)
  File.readlines(File.join(root, "reports", name))
      .map { |l| JSON.parse(l) }
end

def forgetting(root, name)
  JSON.parse(read_text(File.join(root, "reports", name)))
end

# The README is hard wrapped, so a sentence spans lines; and it bolds and
# code-quotes some of the numbers being claimed. Both are removed before
# matching, otherwise "**+28.9**" reads as the claim having been deleted.
text = read_text(File.join(root, "README.md"))
           .gsub(/\s+/, " ").gsub(/[*`]/, "")

# Find exactly one occurrence of a claim and hand its captures to the block,
# which returns the recomputed values to compare against.
def claim(text, re, what)
  found = text.scan(re)
  if found.length != 1
    fail("the README states #{what} #{found.length} times, expected once")
    return
  end
  got = yield(*found[0])
  $checked += 1
  want = found[0]
  if got.map(&:to_s) != want.map(&:to_s)
    fail("#{what}: the README says #{want.inspect}, the data says #{got.map(&:to_s).inspect}")
  else
    puts format("  ok  %-34s %s", what, want.join(", "))
  end
end

base  = preds(root, "preds_base.jsonl")
tuned = preds(root, "preds_tuned.jsonl")
bs    = preds(root, "preds_base_synth.jsonl")
ts    = preds(root, "preds_tuned_synth.jsonl")

if base.map { |r| r["id"] } != tuned.map { |r| r["id"] }
  fail("the two benchmark prediction files are not the same cases in the same order")
end

fb = forgetting(root, "forgetting_base.json")
ft = forgetting(root, "forgetting_tuned.json")

pct = ->(rows, key) { 100.0 * rows.count { |r| r["score"][key] } / rows.length }

# --- section 3, the category slice --------------------------------------

claim(text, /category still went (\d+)\/(\d+) to (\d+)\/(\d+)/,
      "the category counts") do
  [base.count { |r| r["score"]["category"] }, base.length,
   tuned.count { |r| r["score"]["category"] }, tuned.length]
end

fixed  = base.zip(tuned).count { |b, t| !b["score"]["category"] && t["score"]["category"] }
broke  = base.zip(tuned).count { |b, t| b["score"]["category"] && !t["score"]["category"] }

claim(text, /the fine-tune fixed (\d+) cases and broke (\d+)/,
      "the category cases moved") { [fixed, broke] }

# "I read all four broken cases" is a claim about all_correct, not about the
# category field, and the two counts are different numbers on this data.
broken_all = base.zip(tuned).count { |b, t| b["score"]["all_correct"] && !t["score"]["all_correct"] }
words = { 1 => "one", 2 => "two", 3 => "three", 4 => "four", 5 => "five" }

claim(text, /I read all (\w+) broken cases and (\w+) are the same failure, category falling back to "other"/,
      "the broken cases read by hand") do
  regressed = base.zip(tuned).select { |b, t| b["score"]["all_correct"] && !t["score"]["all_correct"] }
  # The stated failure is the tuned model answering "other" for a category it
  # got right before, so that is what is counted rather than any regression.
  same = regressed.count do |_b, t|
    obj = (JSON.parse(t["raw"]) rescue nil)
    obj.is_a?(Hash) && obj["category"].to_s.downcase == "other" &&
      t["gold"]["category"].to_s.downcase != "other"
  end
  [words.fetch(broken_all, broken_all.to_s), words.fetch(same, same.to_s)]
end

# --- section 3 and the abstract, the two slices that get worse ----------

by_kind = lambda do |rows, kind|
  hit = rows.select { |r| r["kind"] == kind }
  fail("no benchmark case has kind #{kind}") if hit.empty?
  100.0 * hit.count { |r| r["score"]["all_correct"] } / hit.length
end

claim(text, /written_amount falls from ([\d.]+) to ([\d.]+) and currency from ([\d.]+) to ([\d.]+)/,
      "the two slices, abstract") do
  ["%.2f" % (by_kind.(base, "written_amount") / 100),
   "%.2f" % (by_kind.(tuned, "written_amount") / 100),
   "%.2f" % (by_kind.(base, "currency") / 100),
   "%.2f" % (by_kind.(tuned, "currency") / 100)]
end

claim(text, /written-out amounts drop from (\d+)% to (\d+)%, currency from (\d+)% to (\d+)%/,
      "the two slices, section 3") do
  ["%.0f" % by_kind.(base, "written_amount"), "%.0f" % by_kind.(tuned, "written_amount"),
   "%.0f" % by_kind.(base, "currency"), "%.0f" % by_kind.(tuned, "currency")]
end

# --- section 4, the generalisation gap ----------------------------------

claim(text, /The gap between those two rows is ([\d.]+) points/, "the generalisation gap") do
  ["%.1f" % (pct.(ts, "all_correct") - pct.(tuned, "all_correct"))]
end

claim(text, /scoring worse on the synthetic set \(([\d.]+)%\) than on the hand-written one \(([\d.]+)%\)/,
      "the base model on both sets") do
  ["%.1f" % pct.(bs, "all_correct"), "%.1f" % pct.(base, "all_correct")]
end

# --- section 1 and 2, the forgetting check ------------------------------

claim(text, /the two protocols disagree by ([\d.]+) points, ([\d.]+)% ranked against ([\d.]+)% generated/,
      "the protocol disagreement") do
  ["%.1f" % (100 * (fb["arc_generative"] - fb["arc_loglikelihood"])),
   "%.1f" % (100 * fb["arc_loglikelihood"]),
   "%.1f" % (100 * fb["arc_generative"])]
end

claim(text, /movement, .0\.7 points on log-likelihood, is (\w+) question out of (\d+)/,
      "the one question of 150") do
  moved = ((fb["arc_loglikelihood"] - ft["arc_loglikelihood"]) * fb["n_mc"]).round
  [words.fetch(moved, moved.to_s), fb["n_mc"]]
end

claim(text, /it still scores identically on (\d+) multiple-choice science questions/,
      "the size of the ARC sample") { [ft["n_mc"]] }

claim(text, /(\d+) hand-written cases whose vendors never appear in training/,
      "the size of the benchmark") { [tuned.length] }

if $failures > 0
  puts "\n#{$failures} checks failed"
  exit 1
end
puts "Ruby reproduces all #{$checked} counting claims the README makes in prose"
