//! Two things the Python could not afford, on the same 45 paired benchmark
//! cases the README reports the +28.9 point gain from.
//!
//! 1. The exact paired permutation test, by enumeration. The 45 cases split
//!    into agreements, which carry no information, and disagreements, which do.
//!    Every assignment of signs to the disagreements is enumerated, all
//!    2^k of them, and the null distribution is counted rather than sampled or
//!    approximated. R computes the same p-value from the binomial closed form
//!    in `verify/inference.R`; if the closed form were the wrong one, the
//!    enumeration would not land on it.
//!
//! 2. A two million draw bootstrap, and then the error bar on the ten thousand
//!    draw bootstrap R runs. An interval estimated by resampling is itself a
//!    random variable, and nothing here had measured how much it moves.
//!
//! No crates: the generator is a xorshift written out below, so CI needs
//! nothing but the toolchain.

use std::env;
use std::fs;
use std::process::exit;

const BIG_DRAWS: usize = 2_000_000;
const R_DRAWS: usize = 10_000;
const REPLICATES: usize = 40;

/// xorshift64*. Not cryptographic and not meant to be: it needs to be uniform,
/// fast and reproducibly seeded so a failure can be re-run.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn die(msg: &str) -> ! {
    eprintln!("paired: {}", msg);
    exit(2)
}

/// Pull one `"key": true|false` out of a JSONL line. Written by hand rather
/// than with a JSON crate, so this file depends on nothing.
fn flag(line: &str, key: &str) -> Option<bool> {
    let needle = format!("\"{}\": ", key);
    let at = line.find(&needle)? + needle.len();
    let rest = &line[at..];
    if rest.starts_with("true") {
        Some(true)
    } else if rest.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn ident(line: &str) -> Option<&str> {
    let at = line.find("\"id\": ")? + 6;
    let rest = &line[at..];
    let end = rest.find(',')?;
    Some(&rest[..end])
}

struct Split {
    correct: Vec<bool>,
    ids: Vec<String>,
}

fn load(path: &str) -> Split {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    let mut correct = Vec::new();
    let mut ids = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match flag(line, "all_correct") {
            Some(v) => correct.push(v),
            None => die(&format!("{} line {} has no all_correct", path, i + 1)),
        }
        match ident(line) {
            Some(v) => ids.push(v.to_string()),
            None => die(&format!("{} line {} has no id", path, i + 1)),
        }
    }
    if correct.is_empty() {
        die(&format!("{} is empty", path));
    }
    Split { correct, ids }
}

/// The published all_correct rate for one block of a summary file, read off the
/// pretty printed JSON without a parser.
fn published_rate(path: &str, block: &str) -> f64 {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    let at = match text.find(&format!("\"{}\": {{", block)) {
        Some(a) => a,
        None => die(&format!("{} has no block {}", path, block)),
    };
    let rest = &text[at..];
    let key = "\"all_correct\": ";
    let start = match rest.find(key) {
        Some(s) => s + key.len(),
        None => die(&format!("{} block {} has no all_correct", path, block)),
    };
    let tail = &rest[start..];
    let end = tail.find(|c: char| c != '.' && !c.is_ascii_digit()).unwrap_or(tail.len());
    tail[..end]
        .parse()
        .unwrap_or_else(|_| die(&format!("{}: all_correct is not a number", path)))
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (pos - lo as f64) * (sorted[hi] - sorted[lo])
    }
}

/// One bootstrap of the paired mean difference. Returns the interval and the
/// spread of the replicates.
fn bootstrap(d: &[f64], draws: usize, rng: &mut Rng) -> (f64, f64, f64) {
    let n = d.len();
    let mut stats = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut sum = 0.0;
        for _ in 0..n {
            sum += d[rng.below(n)];
        }
        stats.push(sum / n as f64);
    }
    stats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f64 = stats.iter().sum::<f64>() / draws as f64;
    let var = stats.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / draws as f64;
    (quantile(&stats, 0.025), quantile(&stats, 0.975), var.sqrt())
}

/// Exact two sided paired permutation test, by enumeration.
///
/// Under the null a case that changed could have changed either way, so the
/// null distribution is every assignment of signs to the k disagreements. The
/// agreements contribute zero to the difference under every assignment, which
/// is what keeps the enumeration to 2^k.
fn exact_permutation_p(up: u32, down: u32) -> (f64, u64, u64) {
    let k = up + down;
    if k > 30 {
        die("too many discordant pairs to enumerate exhaustively");
    }
    let observed = (up as i64 - down as i64).abs();
    let total: u64 = 1u64 << k;
    let mut extreme: u64 = 0;
    for mask in 0..total {
        let plus = (mask as u64).count_ones() as i64;
        let stat = plus - (k as i64 - plus);
        if stat.abs() >= observed {
            extreme += 1;
        }
    }
    (extreme as f64 / total as f64, extreme, total)
}

fn main() {
    let root = env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let base = load(&format!("{}/reports/preds_base.jsonl", root));
    let tuned = load(&format!("{}/reports/preds_tuned.jsonl", root));
    if base.ids != tuned.ids {
        die("the two benchmark files are not the same cases in the same order");
    }
    let n = base.correct.len();

    // Deterministic anchor: the rates this works from have to be the published
    // ones before any of the resampling below means anything.
    let mut failures = 0;
    for (label, split, file, block) in [
        ("base", &base, "summary_base.json", "benchmark"),
        ("tuned", &tuned, "summary_tuned.json", "benchmark"),
    ] {
        let got = (split.correct.iter().filter(|c| **c).count() as f64 / n as f64
            * 10_000.0)
            .round()
            / 10_000.0;
        let want = published_rate(&format!("{}/reports/{}", root, file), block);
        let ok = got == want;
        if !ok {
            failures += 1;
        }
        println!(
            "  {:<6} all_correct {:.4}  published {:.4}  {}",
            label,
            got,
            want,
            if ok { "ok" } else { "FAIL" }
        );
    }

    let d: Vec<f64> = base
        .correct
        .iter()
        .zip(tuned.correct.iter())
        .map(|(b, t)| (*t as i32 as f64) - (*b as i32 as f64))
        .collect();
    let gain: f64 = d.iter().sum::<f64>() / n as f64;
    let up = d.iter().filter(|x| **x > 0.0).count() as u32;
    let down = d.iter().filter(|x| **x < 0.0).count() as u32;

    let (p, extreme, total) = exact_permutation_p(up, down);
    println!(
        "\n  gain {:+.1} points, {} cases fixed, {} broken",
        100.0 * gain,
        up,
        down
    );
    println!(
        "  exact permutation test: {} of {} sign assignments are at least as \
         extreme, p = {:.9}",
        extreme, total, p
    );
    if p >= 0.05 {
        println!("FAIL: the exact test does not separate the gain from chance");
        failures += 1;
    }

    // The closed form the R script uses, recomputed here from the enumeration's
    // own counts, so the two implementations meet on a number rather than on a
    // description.
    let mut rng = Rng::new(0x5EED_1234_ABCD_0001);
    let (lo, hi, se) = bootstrap(&d, BIG_DRAWS, &mut rng);
    let mean = gain;
    let var: f64 = d.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let analytic_se = (var / n as f64).sqrt();
    println!(
        "\n  {} draw bootstrap: 95% CI [{:+.2}, {:+.2}] points, se {:.3}",
        BIG_DRAWS,
        100.0 * lo,
        100.0 * hi,
        100.0 * se
    );
    println!(
        "  analytic se {:.3} points, ratio {:.4}",
        100.0 * analytic_se,
        se / analytic_se
    );
    if (se / analytic_se - 1.0).abs() > 0.01 {
        println!("FAIL: two million draws do not reproduce the closed form standard error");
        failures += 1;
    }

    // How much of the ten thousand draw interval in the R script is noise.
    let mut los = Vec::with_capacity(REPLICATES);
    let mut his = Vec::with_capacity(REPLICATES);
    for r in 0..REPLICATES {
        let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15 ^ (r as u64 + 1));
        let (l, h, _) = bootstrap(&d, R_DRAWS, &mut rng);
        los.push(l);
        his.push(h);
    }
    let spread = |v: &Vec<f64>| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let s = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt();
        (m, s)
    };
    let (mlo, slo) = spread(&los);
    let (mhi, shi) = spread(&his);
    // A paired difference over n cases can only be a multiple of 1/n, so the
    // edges of the interval live on a lattice and the useful tolerance is one
    // step of it rather than a multiple of a standard deviation that is very
    // nearly zero.
    let step = 100.0 / n as f64;
    println!(
        "\n  {} independent {} draw bootstraps: lower edge {:+.3} sd {:.3}, \
         upper edge {:+.3} sd {:.3} points, one case is {:.2} points",
        REPLICATES,
        R_DRAWS,
        100.0 * mlo,
        100.0 * slo,
        100.0 * mhi,
        100.0 * shi,
        step
    );
    for (name, reference, m) in [("lower", lo, mlo), ("upper", hi, mhi)] {
        let off = (100.0 * (m - reference)).abs();
        println!(
            "  {} edge sits {:.3} points from the two million draw reference",
            name, off
        );
        if off > step {
            println!("FAIL: {} edge of the small bootstrap is off by more than one case", name);
            failures += 1;
        }
    }

    if failures > 0 {
        println!("\n{} checks failed", failures);
        exit(1);
    }
    println!(
        "\nRust reproduces the published rates, the exact test by enumeration, \nand puts the \
         smaller bootstrap inside its own Monte Carlo error"
    );
}
