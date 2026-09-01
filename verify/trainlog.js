// Check section 6 of the README against reports/train_log.csv.
//
// Section 6 makes five claims about a training run that took 74 minutes and is
// never going to be repeated: how long it took, where the loss was at step 140,
// where it first touched 0.0001, where it ended, and how many steps there were.
// Those numbers were typed into the README by hand from a run that has since
// been thrown away, and the CSV is the only surviving record of it. Nothing
// compared the two.
//
// The claims are pulled out of the README by pattern rather than hard coded
// here, so rewording the sentence is a loud failure rather than a silent one:
// a check that goes on passing after the text it checks has changed is not
// checking the text.
//
// Usage: node verify/trainlog.js <repository root>

const fs = require("fs");
const path = require("path");

const root = process.argv[2] || ".";
let failures = 0;

function fail(msg) {
    console.log("FAIL: " + msg);
    failures++;
}

function claim(text, re, what) {
    const all = [...text.matchAll(re)];
    if (all.length !== 1) {
        fail(`the README makes the claim about ${what} ${all.length} times, expected once`);
        return null;
    }
    return all[0];
}

// --- the log ------------------------------------------------------------

// The log is written with CRLF line endings, so the carriage return is
// stripped before anything is parsed. Left in, it turns the last column of
// every row into a string that reads as a number in some places and not in
// others, which is the worst way for this to go wrong.
const csv = fs.readFileSync(path.join(root, "reports", "train_log.csv"), "utf8")
    .trim().split("\n").map((l) => l.replace(/\r$/, ""));
const header = csv[0].split(",");
const want = ["step", "epoch", "loss", "lr", "seconds"];
for (const c of want) {
    if (!header.includes(c)) {
        fail(`train_log.csv has no ${c} column`);
    }
}
const at = Object.fromEntries(want.map((c) => [c, header.indexOf(c)]));

const rows = csv.slice(1).map((line, i) => {
    const cells = line.split(",");
    if (cells.length !== header.length) {
        fail(`train_log.csv row ${i + 2} has ${cells.length} cells, header has ${header.length}`);
    }
    const row = {};
    for (const c of want) {
        const v = Number(cells[at[c]]);
        if (!Number.isFinite(v)) {
            fail(`train_log.csv row ${i + 2} column ${c} is ${cells[at[c]]}`);
        }
        row[c] = v;
    }
    return row;
});

// Ordering, because a log written out of order would make every claim below
// mean something different.
for (let i = 1; i < rows.length; i++) {
    if (rows[i].step <= rows[i - 1].step) {
        fail(`step goes ${rows[i - 1].step} then ${rows[i].step} at row ${i + 2}`);
    }
    if (rows[i].seconds <= rows[i - 1].seconds) {
        fail(`the clock goes backwards at row ${i + 2}`);
    }
    if (rows[i].epoch < rows[i - 1].epoch) {
        fail(`epoch goes backwards at row ${i + 2}`);
    }
    if (rows[i].loss < 0) {
        fail(`negative loss at row ${i + 2}`);
    }
}

const last = rows[rows.length - 1];
const epochs = [...new Set(rows.map((r) => r.epoch))].sort((a, b) => a - b);
const minutes = last.seconds / 60;

console.log(`  ${rows.length} logged steps, ${epochs.length} epochs, last step ` +
    `${last.step}, ${minutes.toFixed(1)} minutes of wall clock`);

// --- the claims ---------------------------------------------------------

// The claims are sentences and the README is hard wrapped, so a claim can be
// split across two lines. Runs of whitespace collapse to one space before any
// pattern is applied, otherwise the checks pass by never matching.
const readme = fs.readFileSync(path.join(root, "README.md"), "utf8")
    .replace(/\s+/g, " ");
let checked = 0;

const wall = claim(readme, /real run on variable-length ones took (\d+) minutes/g, "wall clock");
if (wall) {
    const claimed = Number(wall[1]);
    const got = Math.round(minutes);
    checked++;
    if (claimed !== got) {
        fail(`the README says ${claimed} minutes, the log says ${got}`);
    }
}

const at140 = claim(readme,
    /Loss was already down to ([\d.]+) by step (\d+) of (\d+)/g, "the loss at a step");
if (at140) {
    const [, claimedLoss, claimedStep, claimedTotal] = at140;
    const row = rows.find((r) => r.step === Number(claimedStep));
    checked++;
    if (!row) {
        fail(`the log has no step ${claimedStep}`);
    } else if (!(row.loss <= Number(claimedLoss))) {
        fail(`at step ${claimedStep} the loss is ${row.loss}, not down to ${claimedLoss}`);
    }

    // The total number of steps is not in the log, which is written every ten
    // steps. What the log does fix is a bracket: the epoch column changes
    // somewhere inside the ten steps before each change is first seen, and the
    // run cannot be shorter than its last logged step.
    const total = Number(claimedTotal);
    checked++;
    if (total % epochs.length !== 0) {
        fail(`${total} steps is not a whole number of ${epochs.length} epochs`);
    } else {
        const perEpoch = total / epochs.length;
        let lo = 1, hi = Infinity;
        for (let e = 1; e < epochs.length; e++) {
            const firstSeen = rows.find((r) => r.epoch === epochs[e]).step;
            const prev = rows.filter((r) => r.epoch === epochs[e - 1]).pop().step;
            // epoch e starts after the last step logged under epoch e-1 and no
            // later than the first step logged under epoch e.
            lo = Math.max(lo, Math.ceil((prev + 1) / e));
            hi = Math.min(hi, Math.floor(firstSeen / e));
        }
        lo = Math.max(lo, Math.ceil(last.step / epochs.length));
        if (perEpoch < lo || perEpoch > hi) {
            fail(`${total} steps means ${perEpoch} per epoch, but the log brackets ` +
                `that to [${lo}, ${hi}]`);
        }
    }
}

const touched = claim(readme, /first touched ([\d.]+) at step (\d+)/g, "the first low loss");
if (touched) {
    const [, level, claimedStep] = touched;
    const first = rows.find((r) => r.loss <= Number(level));
    checked++;
    if (!first) {
        fail(`the loss never reaches ${level}`);
    } else if (first.step !== Number(claimedStep)) {
        fail(`the loss first reaches ${level} at step ${first.step}, not ${claimedStep}`);
    }
}

const final = claim(readme, /final loss ([\d.]+)/g, "the final loss");
if (final) {
    const claimed = final[1];
    const got = last.loss.toFixed(claimed.split(".")[1].length);
    checked++;
    if (got !== claimed) {
        fail(`the README says the final loss is ${claimed}, the log ends at ${got}`);
    }
}

if (failures > 0) {
    console.log(`\n${failures} checks failed`);
    process.exit(1);
}
console.log(`  ${checked} claims in section 6 of the README hold against the log`);
console.log("JavaScript reproduces every training claim the README makes");
