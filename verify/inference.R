# What the repository never worked out: how much of the measured gain, and of
# the measured forgetting, is sampling noise.
#
# The README reports +28.9 points on 45 hand-written cases and -0.7 points on
# 150 ARC items, and says the second is one question rather than degradation.
# Both are point estimates from small samples and neither came with an interval.
# This recomputes the point estimates from the prediction files, then puts an
# interval on each with base R and its own generator:
#
#   deterministic   the four all_correct rates, which must match the published
#                   summaries exactly
#   stochastic      a paired bootstrap of the benchmark gain, an exact McNemar
#                   test on the same 45 items, and exact binomial intervals on
#                   the three ARC comparisons
#
# No packages, so CI needs nothing beyond the R that is already on the runner.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."
set.seed(20260901)

DRAWS <- 10000
failures <- 0

fail <- function(...) {
    cat("FAIL: ", sprintf(...), "\n", sep = "")
    failures <<- failures + 1
}

# --- reading ------------------------------------------------------------
# Base R has no JSON parser and this file is not allowed to add a package, so
# the two things it needs are pulled out by name with a regular expression and
# the count of matches is checked against the number of lines. A field that
# moved or vanished is a hard error rather than a silent NA.
field_flags <- function(path, key) {
    lines <- readLines(path, warn = FALSE)
    m <- regmatches(lines, regexpr(sprintf('"%s": (true|false)', key), lines))
    if (length(m) != length(lines))
        stop(sprintf("%s: %d of %d lines carry %s", path, length(m),
                     length(lines), key))
    grepl("true", m, fixed = TRUE)
}

record_ids <- function(path) {
    lines <- readLines(path, warn = FALSE)
    m <- regmatches(lines, regexpr('"id": ("[^"]*"|null)', lines))
    if (length(m) != length(lines))
        stop(sprintf("%s: not every line carries an id", path))
    sub('"id": ', "", m, fixed = TRUE)
}

# The summaries are pretty printed with one value per line, so the published
# number is read off the line inside the named block.
summary_value <- function(path, block, key) {
    lines <- readLines(path, warn = FALSE)
    start <- grep(sprintf('^  "%s": \\{', block), lines)
    if (length(start) != 1)
        stop(sprintf("%s: no block %s", path, block))
    stop_at <- grep("^  \\}", lines)
    stop_at <- min(stop_at[stop_at > start])
    hit <- grep(sprintf('^    "%s": ', key), lines[start:stop_at], value = TRUE)
    if (length(hit) != 1)
        stop(sprintf("%s: no %s in block %s", path, key, block))
    as.numeric(sub(",$", "", sub(sprintf('^    "%s": ', key), "", hit)))
}

forgetting_value <- function(path, key) {
    lines <- readLines(path, warn = FALSE)
    hit <- grep(sprintf('^  "%s": ', key), lines, value = TRUE)
    if (length(hit) != 1)
        stop(sprintf("%s: no %s", path, key))
    as.numeric(sub(",$", "", sub(sprintf('^  "%s": ', key), "", hit)))
}

p <- function(...) file.path(root, "reports", ...)

base_bench  <- field_flags(p("preds_base.jsonl"), "all_correct")
tuned_bench <- field_flags(p("preds_tuned.jsonl"), "all_correct")
base_synth  <- field_flags(p("preds_base_synth.jsonl"), "all_correct")
tuned_synth <- field_flags(p("preds_tuned_synth.jsonl"), "all_correct")

if (!identical(record_ids(p("preds_base.jsonl")), record_ids(p("preds_tuned.jsonl"))))
    stop("the two benchmark files are not the same 45 cases in the same order")

# --- deterministic ------------------------------------------------------

cat("point estimates, against reports/summary_*.json\n")
for (row in list(
        list("base benchmark",  base_bench,  "summary_base.json",  "benchmark"),
        list("tuned benchmark", tuned_bench, "summary_tuned.json", "benchmark"),
        list("base synthetic",  base_synth,  "summary_base.json",  "synthetic"),
        list("tuned synthetic", tuned_synth, "summary_tuned.json", "synthetic"))) {
    got <- round(mean(row[[2]]), 4)
    want <- summary_value(p(row[[3]]), row[[4]], "all_correct")
    ok <- identical(got, want)
    if (!ok) failures <- failures + 1
    cat(sprintf("  %-16s all_correct %.4f  published %.4f  %s\n",
                row[[1]], got, want, if (ok) "ok" else "FAIL"))
}

# --- the benchmark gain -------------------------------------------------
# The same 45 cases are scored twice, so the two rates are paired and the
# difference per case is what carries the information.

d <- as.numeric(tuned_bench) - as.numeric(base_bench)
n <- length(d)
gain <- mean(d)
cat(sprintf("\nbenchmark gain %+.1f points on %d paired cases\n", 100 * gain, n))

boot <- numeric(DRAWS)
for (b in seq_len(DRAWS)) {
    idx <- sample.int(n, n, replace = TRUE)
    boot[b] <- mean(d[idx])
}
ci <- quantile(boot, c(0.025, 0.975), names = FALSE)
cat(sprintf("  paired bootstrap, %d draws: 95%% CI [%+.1f, %+.1f] points, se %.1f\n",
            DRAWS, 100 * ci[1], 100 * ci[2], 100 * sd(boot)))
# The bootstrap standard error of a paired mean has a closed form, so the
# resampling can be checked against arithmetic rather than trusted.
analytic_se <- sd(d) / sqrt(n) * sqrt((n - 1) / n)
cat(sprintf("  analytic se %.1f points, bootstrap se %.1f points, ratio %.3f\n",
            100 * analytic_se, 100 * sd(boot), sd(boot) / analytic_se))
if (abs(sd(boot) / analytic_se - 1) > 0.05)
    fail("the bootstrap standard error is %.3f of the analytic one",
         sd(boot) / analytic_se)
if (ci[1] <= 0)
    fail("the 95%% interval for the benchmark gain includes zero")

up <- sum(d > 0)
down <- sum(d < 0)
mcnemar <- binom.test(up, up + down, 0.5)
cat(sprintf("  %d cases fixed, %d broken, exact McNemar p = %.9f\n",
            up, down, mcnemar$p.value))
if (mcnemar$p.value >= 0.05)
    fail("the exact test does not separate the gain from chance")

# --- the forgetting check -----------------------------------------------
# Only the aggregate survives for ARC, so the counts are recovered from the
# published rates and the interval is the one for two independent samples.
# That is the conservative direction here: the same items are scored twice, and
# pairing would narrow the interval rather than widen it.

n_mc <- forgetting_value(p("forgetting_base.json"), "n_mc")
counts <- function(file, key) {
    r <- forgetting_value(p(file), key)
    k <- round(r * n_mc)
    if (abs(k - r * n_mc) > 0.5e-4 * n_mc + 1e-9)
        fail("%s %s = %s is not a whole number of items out of %d", file, key, r, n_mc)
    k
}

diff_ci <- function(k1, k2) {
    p1 <- k1 / n_mc; p2 <- k2 / n_mc
    se <- sqrt(p1 * (1 - p1) / n_mc + p2 * (1 - p2) / n_mc)
    c(p2 - p1 - 1.96 * se, p2 - p1 + 1.96 * se)
}

ll_base  <- counts("forgetting_base.json", "arc_loglikelihood")
ll_tuned <- counts("forgetting_tuned.json", "arc_loglikelihood")
gen_base <- counts("forgetting_base.json", "arc_generative")
gen_tuned <- counts("forgetting_tuned.json", "arc_generative")

cat(sprintf("\nARC on %d items: log-likelihood %d -> %d, generative %d -> %d\n",
            n_mc, ll_base, ll_tuned, gen_base, gen_tuned))

ci_forget <- diff_ci(ll_base, ll_tuned)
cat(sprintf("  fine-tuning effect on knowledge %+.1f points, 95%% CI [%+.1f, %+.1f]\n",
            100 * (ll_tuned - ll_base) / n_mc, 100 * ci_forget[1], 100 * ci_forget[2]))
if (ci_forget[1] > 0 || ci_forget[2] < 0)
    fail("the ARC drop is outside the interval, so the README calling it noise is wrong")

ci_protocol <- diff_ci(ll_base, gen_base)
cat(sprintf("  protocol disagreement on the base model %+.1f points, 95%% CI [%+.1f, %+.1f]\n",
            100 * (gen_base - ll_base) / n_mc, 100 * ci_protocol[1], 100 * ci_protocol[2]))
if (ci_protocol[1] <= 0)
    fail("the two scoring protocols are not separated, so section 2 overstates")

if (failures > 0) {
    cat(sprintf("\n%d checks failed\n", failures))
    quit(status = 1)
}
cat("\nR reproduces the four rates exactly, and separates the gain from noise",
    "and the forgetting from signal\n")
