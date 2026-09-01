/* Rescore every prediction in reports/preds_*.jsonl from the raw model output.
 *
 * Everything else under verify/ starts from the per-field `score` object that
 * src/loraft/task.py already wrote. This one does not: it reads the raw string
 * the model produced and the gold record, reimplements the JSON extraction, the
 * field normalisation and the comparison in C, and requires its own verdict to
 * match the published one on every prediction and every field. An error in the
 * Python scorer would have to be repeated here to survive, and the two were
 * written from opposite ends: Python from regular expressions, C from character
 * scans.
 *
 * It then averages its own verdicts into the published all_correct rate, so a
 * single program goes from raw text to the number in the README.
 *
 * Usage: score <repository root>
 */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 65536
#define ARENA_BYTES (1 << 20)

static const char *FIELDS[5] = {"vendor", "amount", "currency", "date", "category"};
static const char *CATEGORIES[6] = {"travel", "meals", "software", "hardware",
                                    "office", "other"};

static void die(const char *what, const char *detail)
{
    fprintf(stderr, "score: %s: %s\n", what, detail);
    exit(2);
}

/* --- arena ------------------------------------------------------------- */
/* One line at a time is parsed, so the whole tree is thrown away by resetting
 * a pointer. No ownership to get wrong and nothing to free. */
static char arena[ARENA_BYTES];
static size_t arena_used;

static void *arena_alloc(size_t n)
{
    size_t aligned = (n + 15u) & ~(size_t)15u;
    if (arena_used + aligned > sizeof arena)
        die("arena exhausted", "a line is larger than this program expects");
    void *p = &arena[arena_used];
    arena_used += aligned;
    return p;
}

/* --- JSON ------------------------------------------------------------- */

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_OBJ, J_ARR } JType;

typedef struct JVal JVal;
struct JVal {
    JType type;
    int boolean;
    double number;
    char *string;      /* decoded, NUL terminated */
    char **keys;       /* J_OBJ */
    JVal **vals;
    JVal **items;      /* J_ARR */
    int n;
};

/* Recursive descent. Returns NULL on any malformed input, which is the same
 * answer json.loads() gives by raising, and the caller treats both as "this
 * response did not parse". */
static JVal *parse_value(const char **p);

static void skip_ws(const char **p)
{
    while (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r')
        (*p)++;
}

static int hex_nibble(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static char *parse_string(const char **p)
{
    if (**p != '"')
        return NULL;
    (*p)++;
    /* A decoded string is never longer than the text still to be read, so
     * this is a safe upper bound and keeps the arena small. */
    size_t cap = strlen(*p) + 8;
    char *out = arena_alloc(cap);
    size_t k = 0;
    while (**p && **p != '"') {
        if (k + 4 >= cap)
            return NULL;
        if (**p == '\\') {
            (*p)++;
            char c = **p;
            (*p)++;
            switch (c) {
            case '"': out[k++] = '"'; break;
            case '\\': out[k++] = '\\'; break;
            case '/': out[k++] = '/'; break;
            case 'b': out[k++] = '\b'; break;
            case 'f': out[k++] = '\f'; break;
            case 'n': out[k++] = '\n'; break;
            case 'r': out[k++] = '\r'; break;
            case 't': out[k++] = '\t'; break;
            case 'u': {
                int v = 0, i;
                for (i = 0; i < 4; i++) {
                    int d = hex_nibble((*p)[i]);
                    if (d < 0)
                        return NULL;
                    v = v * 16 + d;
                }
                *p += 4;
                /* Stored as UTF-8. Surrogate pairs are refused rather than
                 * half handled; nothing in these files produces one. */
                if (v >= 0xd800 && v <= 0xdfff)
                    die("surrogate pair in input",
                        "this checker does not decode astral characters");
                if (v < 0x80) {
                    out[k++] = (char)v;
                } else if (v < 0x800) {
                    out[k++] = (char)(0xc0 | (v >> 6));
                    out[k++] = (char)(0x80 | (v & 0x3f));
                } else {
                    out[k++] = (char)(0xe0 | (v >> 12));
                    out[k++] = (char)(0x80 | ((v >> 6) & 0x3f));
                    out[k++] = (char)(0x80 | (v & 0x3f));
                }
                break;
            }
            default:
                return NULL;
            }
        } else {
            out[k++] = **p;
            (*p)++;
        }
    }
    if (**p != '"')
        return NULL;
    (*p)++;
    out[k] = '\0';
    return out;
}

static JVal *new_val(JType t)
{
    JVal *v = arena_alloc(sizeof *v);
    memset(v, 0, sizeof *v);
    v->type = t;
    return v;
}

#define MAX_MEMBERS 64

static JVal *parse_object(const char **p)
{
    JVal *v = new_val(J_OBJ);
    v->keys = arena_alloc(MAX_MEMBERS * sizeof(char *));
    v->vals = arena_alloc(MAX_MEMBERS * sizeof(JVal *));
    (*p)++;
    skip_ws(p);
    if (**p == '}') { (*p)++; return v; }
    for (;;) {
        skip_ws(p);
        char *key = parse_string(p);
        if (!key)
            return NULL;
        skip_ws(p);
        if (**p != ':')
            return NULL;
        (*p)++;
        skip_ws(p);
        JVal *child = parse_value(p);
        if (!child)
            return NULL;
        if (v->n >= MAX_MEMBERS)
            return NULL;
        v->keys[v->n] = key;
        v->vals[v->n] = child;
        v->n++;
        skip_ws(p);
        if (**p == ',') { (*p)++; continue; }
        if (**p == '}') { (*p)++; return v; }
        return NULL;
    }
}

static JVal *parse_array(const char **p)
{
    JVal *v = new_val(J_ARR);
    v->items = arena_alloc(MAX_MEMBERS * sizeof(JVal *));
    (*p)++;
    skip_ws(p);
    if (**p == ']') { (*p)++; return v; }
    for (;;) {
        skip_ws(p);
        JVal *child = parse_value(p);
        if (!child || v->n >= MAX_MEMBERS)
            return NULL;
        v->items[v->n++] = child;
        skip_ws(p);
        if (**p == ',') { (*p)++; continue; }
        if (**p == ']') { (*p)++; return v; }
        return NULL;
    }
}

static JVal *parse_value(const char **p)
{
    skip_ws(p);
    switch (**p) {
    case '{': return parse_object(p);
    case '[': return parse_array(p);
    case '"': {
        char *s = parse_string(p);
        if (!s)
            return NULL;
        JVal *v = new_val(J_STR);
        v->string = s;
        return v;
    }
    case 't':
        if (strncmp(*p, "true", 4) != 0) return NULL;
        *p += 4;
        { JVal *v = new_val(J_BOOL); v->boolean = 1; return v; }
    case 'f':
        if (strncmp(*p, "false", 5) != 0) return NULL;
        *p += 5;
        { JVal *v = new_val(J_BOOL); v->boolean = 0; return v; }
    case 'n':
        if (strncmp(*p, "null", 4) != 0) return NULL;
        *p += 4;
        return new_val(J_NULL);
    default: {
        char *end;
        double d = strtod(*p, &end);
        if (end == *p) return NULL;
        /* strtod accepts "nan", "inf" and hexadecimal, json.loads does not
         * accept the last of those and none belong in this data. */
        if (!(**p == '-' || (**p >= '0' && **p <= '9'))) return NULL;
        *p = end;
        JVal *v = new_val(J_NUM);
        v->number = d;
        return v;
    }
    }
}

/* Whole-text parse: trailing content is an error, as it is for json.loads. */
static JVal *json_parse(const char *text)
{
    const char *p = text;
    JVal *v = parse_value(&p);
    if (!v)
        return NULL;
    skip_ws(&p);
    return *p == '\0' ? v : NULL;
}

/* Last occurrence wins, which is what a Python dict does with a repeated key. */
static JVal *obj_get(const JVal *o, const char *key)
{
    int i;
    if (!o || o->type != J_OBJ)
        return NULL;
    for (i = o->n - 1; i >= 0; i--)
        if (strcmp(o->keys[i], key) == 0)
            return o->vals[i];
    return NULL;
}

/* --- the scorer ------------------------------------------------------- */

/* extract_json(): markdown fence, then the first brace balanced run.
 *
 * The brace counting is deliberately not string aware, because the Python is
 * not either. Copying the tolerant behaviour matters more than improving on
 * it: this is meant to reproduce the published score, not to disagree with it
 * on a technicality. */
static JVal *extract_json(const char *raw)
{
    static char slice[MAX_LINE];
    const char *candidate = raw;
    size_t clen;
    const char *open, *close;
    size_t i, start;
    int depth;

    if (!raw || !*raw)
        return NULL;

    open = strstr(raw, "```");
    if (open) {
        const char *body = open + 3;
        if (strncmp(body, "json", 4) == 0)
            body += 4;
        while (*body && isspace((unsigned char)*body))
            body++;
        close = strstr(body, "```");
        if (close) {
            candidate = body;
            clen = (size_t)(close - body);
        } else {
            candidate = raw;
            clen = strlen(raw);
        }
    } else {
        clen = strlen(raw);
    }

    for (start = 0; start < clen && candidate[start] != '{'; start++)
        ;
    if (start == clen)
        return NULL;

    depth = 0;
    for (i = start; i < clen; i++) {
        if (candidate[i] == '{')
            depth++;
        else if (candidate[i] == '}') {
            depth--;
            if (depth == 0) {
                size_t n = i - start + 1;
                if (n >= sizeof slice)
                    die("object too long", raw);
                memcpy(slice, candidate + start, n);
                slice[n] = '\0';
                JVal *v = json_parse(slice);
                return (v && v->type == J_OBJ) ? v : NULL;
            }
        }
    }
    return NULL;
}

static int schema_ok(const JVal *o)
{
    int f, i, distinct = 0;
    if (!o || o->type != J_OBJ)
        return 0;
    for (i = 0; i < o->n; i++) {
        int dup = 0, j;
        for (j = 0; j < i; j++)
            if (strcmp(o->keys[i], o->keys[j]) == 0)
                dup = 1;
        if (!dup)
            distinct++;
    }
    if (distinct != 5)
        return 0;
    for (f = 0; f < 5; f++)
        if (!obj_get(o, FIELDS[f]))
            return 0;
    return 1;
}

/* Normalised values. A NULL string or absent number is Python's None. */
typedef struct {
    int is_null;
    double number;
    char text[512];
} Norm;

static Norm null_norm(void)
{
    Norm n;
    n.is_null = 1;
    n.number = 0.0;
    n.text[0] = '\0';
    return n;
}

static const char *as_string(const JVal *v, const char *field)
{
    if (!v || v->type == J_NULL)
        return NULL;
    if (v->type == J_STR) {
        /* Python normalises with NFKD before comparing. On ASCII that is the
         * identity, which is why this reimplementation is allowed to skip it.
         * Anything outside ASCII would need the real thing, so it stops here
         * rather than quietly comparing the wrong strings. */
        const unsigned char *b;
        for (b = (const unsigned char *)v->string; *b; b++)
            if (*b > 0x7f)
                die("non-ASCII text in a scored field", field);
        return v->string;
    }
    /* str(v) on a number would need Python's repr rules. Nothing in this
     * repository produces one, so it is refused loudly rather than guessed. */
    die("unsupported value type for field", field);
    return NULL;
}

/* [^\w\s.&'-] removed, whitespace collapsed, then " ." stripped from both
 * ends. \w is ASCII here, which the parser has already enforced. */
static Norm norm_text(const JVal *v, const char *field)
{
    const char *s = as_string(v, field);
    char buf[512];
    size_t k = 0, i, len;
    Norm out = null_norm();
    if (!s)
        return out;

    while (*s && isspace((unsigned char)*s))
        s++;
    len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1]))
        len--;

    for (i = 0; i < len && k + 1 < sizeof buf; i++) {
        unsigned char c = (unsigned char)tolower((unsigned char)s[i]);
        int keep = isalnum(c) || c == '_' || isspace(c) ||
                   c == '.' || c == '&' || c == '\'' || c == '-';
        if (keep)
            buf[k++] = (char)(isspace(c) ? ' ' : c);
    }
    buf[k] = '\0';

    /* collapse runs of spaces */
    {
        size_t r = 0, w = 0;
        while (r < k) {
            if (buf[r] == ' ') {
                buf[w++] = ' ';
                while (r < k && buf[r] == ' ')
                    r++;
            } else {
                buf[w++] = buf[r++];
            }
        }
        buf[w] = '\0';
        k = w;
    }
    {
        size_t lo = 0, hi = k;
        while (lo < hi && (buf[lo] == ' ' || buf[lo] == '.'))
            lo++;
        while (hi > lo && (buf[hi - 1] == ' ' || buf[hi - 1] == '.'))
            hi--;
        if (hi == lo)
            return null_norm();
        out.is_null = 0;
        memcpy(out.text, buf + lo, hi - lo);
        out.text[hi - lo] = '\0';
    }
    return out;
}

static double round2(double x)
{
    return nearbyint(x * 100.0) / 100.0;
}

static Norm norm_amount(const JVal *v)
{
    Norm out = null_norm();
    char buf[512];
    size_t k = 0, i;
    const char *s;
    char *end;
    double d;

    if (!v || v->type == J_NULL)
        return out;
    if (v->type == J_NUM) {
        out.is_null = 0;
        out.number = round2(v->number);
        return out;
    }
    if (v->type != J_STR)
        die("unsupported value type for field", "amount");

    s = v->string;
    for (i = 0; s[i] && k + 1 < sizeof buf; i++)
        if (isdigit((unsigned char)s[i]) || s[i] == '.' || s[i] == ',' || s[i] == '-')
            buf[k++] = s[i];
    buf[k] = '\0';
    if (k == 0)
        return out;

    {
        char *last_dot = strrchr(buf, '.');
        char *last_comma = strrchr(buf, ',');
        char tmp[512];
        size_t w = 0, j;
        if (last_dot && last_comma) {
            if (last_dot > last_comma) {          /* 1,204.55 */
                for (j = 0; j < k; j++)
                    if (buf[j] != ',')
                        tmp[w++] = buf[j];
            } else {                              /* 1.204,55 */
                for (j = 0; j < k; j++) {
                    if (buf[j] == '.')
                        continue;
                    tmp[w++] = buf[j] == ',' ? '.' : buf[j];
                }
            }
        } else if (last_comma) {
            size_t tail = k - (size_t)(last_comma - buf) - 1;
            for (j = 0; j < k; j++) {
                if (buf[j] == ',') {
                    if (tail == 2)
                        tmp[w++] = '.';
                } else {
                    tmp[w++] = buf[j];
                }
            }
        } else {
            for (j = 0; j < k; j++)
                tmp[w++] = buf[j];
        }
        tmp[w] = '\0';
        memcpy(buf, tmp, w + 1);
        k = w;
    }

    if (k == 0)
        return out;
    d = strtod(buf, &end);
    if (end != buf + k)   /* float() consumes the whole string or raises */
        return out;
    out.is_null = 0;
    out.number = round2(d);
    return out;
}

static Norm norm_currency(const JVal *v)
{
    const char *s = as_string(v, "currency");
    Norm out = null_norm();
    size_t len, i, lo = 0;
    if (!s)
        return out;
    while (*s && isspace((unsigned char)*s))
        s++;
    len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1]))
        len--;
    if (len != 3)
        return out;
    for (i = 0; i < 3; i++) {
        char c = (char)toupper((unsigned char)s[lo + i]);
        if (c < 'A' || c > 'Z')
            return out;
        out.text[i] = c;
    }
    out.text[3] = '\0';
    out.is_null = 0;
    return out;
}

static Norm norm_date(const JVal *v)
{
    const char *s = as_string(v, "date");
    Norm out = null_norm();
    size_t len, i;
    if (!s)
        return out;
    while (*s && isspace((unsigned char)*s))
        s++;
    len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1]))
        len--;
    if (len != 10)
        return out;
    for (i = 0; i < 10; i++) {
        int wantdash = (i == 4 || i == 7);
        if (wantdash ? s[i] != '-' : !isdigit((unsigned char)s[i]))
            return out;
    }
    memcpy(out.text, s, 10);
    out.text[10] = '\0';
    out.is_null = 0;
    return out;
}

static Norm norm_category(const JVal *v)
{
    const char *s = as_string(v, "category");
    Norm out = null_norm();
    size_t len, i;
    int c;
    if (!s)
        return out;
    while (*s && isspace((unsigned char)*s))
        s++;
    len = strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1]))
        len--;
    if (len >= sizeof out.text)
        return out;
    for (i = 0; i < len; i++)
        out.text[i] = (char)tolower((unsigned char)s[i]);
    out.text[len] = '\0';
    for (c = 0; c < 6; c++)
        if (strcmp(out.text, CATEGORIES[c]) == 0) {
            out.is_null = 0;
            return out;
        }
    return null_norm();
}

static Norm normalise(int field, const JVal *v)
{
    switch (field) {
    case 0: return norm_text(v, "vendor");
    case 1: return norm_amount(v);
    case 2: return norm_currency(v);
    case 3: return norm_date(v);
    default: return norm_category(v);
    }
}

static int norm_equal(int field, Norm a, Norm b)
{
    if (a.is_null || b.is_null)
        return a.is_null && b.is_null;
    if (field == 1)
        return a.number == b.number;
    return strcmp(a.text, b.text) == 0;
}

/* Order matches task.score_one(): json_parsed, schema_ok, the five fields,
 * all_correct. */
static void score_one(const char *raw, const JVal *gold, int out[8])
{
    JVal *obj = extract_json(raw);
    int ok = schema_ok(obj), f, all = ok;
    out[0] = obj != NULL;
    out[1] = ok;
    for (f = 0; f < 5; f++) {
        Norm pred = ok ? normalise(f, obj_get(obj, FIELDS[f])) : null_norm();
        Norm want = normalise(f, obj_get(gold, FIELDS[f]));
        if (!ok)
            pred.is_null = 1;
        out[2 + f] = ok && norm_equal(f, pred, want);
        all = all && out[2 + f];
    }
    out[7] = all;
}

/* --- driver ----------------------------------------------------------- */

static const char *SCORE_KEYS[8] = {"json_parsed", "schema_ok", "vendor",
                                    "amount", "currency", "date", "category",
                                    "all_correct"};

struct Split {
    const char *preds;
    const char *summary;
    const char *block;
};

static double published_rate(const char *root, const char *summary,
                             const char *block)
{
    /* Read the summary file whole and pull block.all_correct out of it. */
    char path[1024];
    static char text[1 << 20];
    FILE *f;
    size_t n;
    JVal *doc, *b, *v;

    snprintf(path, sizeof path, "%s/reports/%s", root, summary);
    f = fopen(path, "rb");
    if (!f)
        die("cannot open", path);
    n = fread(text, 1, sizeof text - 1, f);
    fclose(f);
    text[n] = '\0';
    arena_used = 0;
    doc = json_parse(text);
    b = obj_get(doc, block);
    v = obj_get(b, "all_correct");
    if (!v || v->type != J_NUM)
        die("no all_correct in", path);
    return v->number;
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    struct Split splits[4] = {
        {"preds_base.jsonl", "summary_base.json", "benchmark"},
        {"preds_tuned.jsonl", "summary_tuned.json", "benchmark"},
        {"preds_base_synth.jsonl", "summary_base.json", "synthetic"},
        {"preds_tuned_synth.jsonl", "summary_tuned.json", "synthetic"},
    };
    long total_rows = 0, total_fields = 0, disagreements = 0;
    int s;

    for (s = 0; s < 4; s++) {
        char path[1024];
        static char line[MAX_LINE];
        FILE *f;
        long rows = 0, correct = 0, bad = 0;
        double got, want;

        snprintf(path, sizeof path, "%s/reports/%s", root, splits[s].preds);
        f = fopen(path, "rb");
        if (!f)
            die("cannot open", path);

        while (fgets(line, sizeof line, f)) {
            JVal *rec, *raw, *gold, *published;
            int mine[8], k;
            size_t len = strlen(line);

            if (len == 0 || line[len - 1] != '\n') {
                if (!feof(f))
                    die("line too long in", path);
            }
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
                line[--len] = '\0';
            if (len == 0)
                continue;

            arena_used = 0;
            rec = json_parse(line);
            if (!rec)
                die("unparseable record in", path);
            raw = obj_get(rec, "raw");
            gold = obj_get(rec, "gold");
            published = obj_get(rec, "score");
            if (!raw || raw->type != J_STR || !gold || !published)
                die("record is missing raw, gold or score in", path);

            score_one(raw->string, gold, mine);
            for (k = 0; k < 8; k++) {
                JVal *p = obj_get(published, SCORE_KEYS[k]);
                int theirs;
                if (!p || p->type != J_BOOL)
                    die("score field is not a boolean in", path);
                theirs = p->boolean;
                total_fields++;
                if (theirs != mine[k]) {
                    bad++;
                    if (bad <= 5)
                        printf("  %s row %ld field %s: C says %d, published %d\n",
                               splits[s].preds, rows + 1, SCORE_KEYS[k],
                               mine[k], theirs);
                }
            }
            correct += mine[7];
            rows++;
        }
        fclose(f);
        if (rows == 0)
            die("no rows in", path);

        got = nearbyint((double)correct / (double)rows * 10000.0) / 10000.0;
        want = published_rate(root, splits[s].summary, splits[s].block);
        printf("  %-24s %4ld predictions rescored, %ld field disagreements, "
               "all_correct %.4f against published %.4f\n",
               splits[s].preds, rows, bad, got, want);
        if (got != want) {
            printf("  %s: rate disagrees with %s\n", splits[s].preds,
                   splits[s].summary);
            disagreements++;
        }
        disagreements += bad;
        total_rows += rows;
    }

    printf("  %ld predictions, %ld field verdicts recomputed from the raw model "
           "output\n", total_rows, total_fields);
    if (disagreements > 0) {
        printf("C disagrees with the published scoring in %ld places\n",
               disagreements);
        return 1;
    }
    printf("C reproduces every published field verdict and every all_correct "
           "rate exactly\n");
    return 0;
}
