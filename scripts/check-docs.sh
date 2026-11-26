#!/usr/bin/env bash
# better-docs universal deterministic documentation check gate.
# Specification: docs/check-spec.md (diagnostics BD001–BD070 hard-fail, BD901–BD903 advisory).
# Prerequisites: bash (>= 4.4), POSIX awk/utilities, git (unless --no-git), sha256sum or shasum.

set -uo pipefail

KIT_VERSION_BUILT="0.1.0"

ROOT="."
CONFIG_PATH=""
CONFIG_SET=0
CI_MODE=0
BASE_REF=""
BASE_REF_SET=0
SEAL_MODE=0
PROBES_MODE=0
LIST_BUDGETS=0
NO_GIT=0

usage() {
  cat <<'EOF'
Usage: check-docs.sh [options]
  --root <dir>         Target repository root (default: .)
  --config <path>      Config file path (default: <root>/docs-check.conf)
  --ci                 CI mode: requires git worktree and --base-ref (or BETTER_DOCS_BASE_REF) != HEAD; incompatible with --seal and --no-git
  --base-ref <ref>     Git baseline ref for append-only archive seal checks
  --seal               Verify existing seals, then rewrite path-sorted manifest.sha256; incompatible with --ci
  --probes             Emit advisory EN/ZH lexical recall diagnostics (BD901; never changes exit code)
  --list-budgets       Also print token usage for every configured budget= entry (enforces BD010 only when feature_budgets=true)
  --no-git             Run without git (disables tracked-scratch BD001, baseline BD041, and .gitmodules BD903; incompatible with --ci)
  --version            Print gate kit version and exit 0
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      [[ $# -ge 2 ]] || { echo "check-docs.sh: --root requires an argument" >&2; exit 2; }
      ROOT="$2"; shift 2 ;;
    --config)
      [[ $# -ge 2 ]] || { echo "check-docs.sh: --config requires an argument" >&2; exit 2; }
      CONFIG_PATH="$2"; CONFIG_SET=1; shift 2 ;;
    --ci) CI_MODE=1; shift ;;
    --base-ref)
      [[ $# -ge 2 ]] || { echo "check-docs.sh: --base-ref requires an argument" >&2; exit 2; }
      BASE_REF="$2"; BASE_REF_SET=1; shift 2 ;;
    --seal) SEAL_MODE=1; shift ;;
    --probes) PROBES_MODE=1; shift ;;
    --list-budgets) LIST_BUDGETS=1; shift ;;
    --no-git) NO_GIT=1; shift ;;
    --version) printf '%s\n' "$KIT_VERSION_BUILT"; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "check-docs.sh: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$CI_MODE" -eq 1 && "$SEAL_MODE" -eq 1 ]]; then
  echo "check-docs.sh: --ci cannot be used with --seal" >&2
  exit 2
fi
if [[ "$CI_MODE" -eq 1 && "$NO_GIT" -eq 1 ]]; then
  echo "check-docs.sh: --ci cannot be used with --no-git" >&2
  exit 2
fi
if [[ "$BASE_REF_SET" -eq 1 && -z "$BASE_REF" ]]; then
  echo "check-docs.sh: --base-ref requires a non-empty ref" >&2
  exit 2
fi

if [[ -z "$ROOT" || ! -d "$ROOT" ]]; then
  echo "check-docs.sh: root directory not found: $ROOT" >&2
  exit 2
fi
ROOT="$(cd "$ROOT" && pwd -P)"

if [[ "$CONFIG_SET" -eq 1 ]]; then
  if [[ -z "$CONFIG_PATH" ]]; then
    echo "check-docs.sh: --config requires a non-empty path" >&2
    exit 2
  fi
  CONFIG_DISPLAY="$CONFIG_PATH"
else
  CONFIG_PATH="$ROOT/docs-check.conf"
  CONFIG_DISPLAY="docs-check.conf"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "check-docs.sh: config file not found: $CONFIG_PATH" >&2
  exit 2
fi

# Prerequisite checks
if ! command -v awk >/dev/null 2>&1; then
  echo "check-docs.sh: missing prerequisite 'awk'" >&2
  exit 2
fi
SHA_CMD=""
if command -v sha256sum >/dev/null 2>&1; then
  SHA_CMD="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
  SHA_CMD="shasum -a 256"
else
  echo "check-docs.sh: missing prerequisite 'sha256sum' or 'shasum'" >&2
  exit 2
fi

hash_file() {
  local f="$1"
  if [[ "$SHA_CMD" == "sha256sum" ]]; then
    sha256sum < "$f" | awk '{print $1}'
  else
    shasum -a 256 < "$f" | awk '{print $1}'
  fi
}

IN_GIT=0
if [[ "$NO_GIT" -eq 0 ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "check-docs.sh: missing prerequisite 'git' (pass --no-git to run without git)" >&2
    exit 2
  fi
  if [[ "$(git -C "$ROOT" rev-parse --is-inside-work-tree 2>/dev/null || true)" == "true" ]]; then
    IN_GIT=1
  fi
fi

if [[ "$CI_MODE" -eq 1 && "$IN_GIT" -eq 0 ]]; then
  echo "check-docs.sh: --ci requires a git worktree" >&2
  exit 2
fi

FAILURES=0
fail() {
  local id="$1"
  local loc="$2"
  local msg="$3"
  printf '%s %s %s\n' "$id" "$loc" "$msg" >&2
  FAILURES=$((FAILURES + 1))
}

advise() {
  local id="$1"
  local loc="$2"
  local msg="$3"
  printf '%s %s %s\n' "$id" "$loc" "$msg"
}

is_safe_rel_path() {
  local p="$1"
  [[ -n "$p" ]] || return 1
  [[ "$p" != /* && "$p" != */ && "$p" != *//* ]] || return 1
  [[ ! "$p" =~ (^|/)\.\.?(/|$) ]] || return 1
  return 0
}

# ---------------------------------------------------------------------------
# 1. Tracked .agents/scratch/ rejection (BD001) — runs FIRST before exclusions
# ---------------------------------------------------------------------------
if [[ "$IN_GIT" -eq 1 ]]; then
  while IFS= read -r tracked_scratch; do
    [[ -n "$tracked_scratch" ]] && fail "BD001" "$tracked_scratch" "tracked file under .agents/scratch/ is forbidden"
  done < <(git -C "$ROOT" -c core.quotePath=false ls-files -- .agents/scratch)
fi

# ---------------------------------------------------------------------------
# 2. Parse and validate config (BD002, BD003)
# ---------------------------------------------------------------------------
KIT_VERSION=""
LANGUAGE_MODE="single-en"
LANGUAGE_PRIMARY="en"
FEATURE_NOTES="false"
FEATURE_EMPIRICAL_RECORDS="false"
FEATURE_ARCHIVE_SEAL="false"
FEATURE_BUDGETS="false"
NOTE_CLASSES="feature,bug-fix,architecture,simplification,process,testing"

declare -a SURFACE_RULES=()
declare -a GRANDFATHERED_NOTES=()
declare -a SUBTREES=()
declare -a SUBMODULE_OWNED=()
declare -a SUBMODULE_SHARED=()
declare -a SUBMODULE_EXTERNAL=()
declare -a BUDGETS=()
declare -a HOOKS_REQUIRED=()
declare -a HOOKS_ADVISORY=()

line_no=0
while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line_no=$((line_no + 1))
  line="${raw_line%$'\r'}"
  if [[ "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*# ]]; then
    continue
  fi
  line="$(printf '%s' "$line" | sed -E 's/[[:space:]]+#.*$//')"
  line="$(printf '%s' "$line" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
  if [[ "$line" != *=* ]]; then
    fail "BD002" "$CONFIG_DISPLAY:$line_no" "malformed config line (expected key=value)"
    continue
  fi
  key="${line%%=*}"
  val="${line#*=}"
  key="$(printf '%s' "$key" | sed -E 's/[[:space:]]+$//')"
  val="$(printf '%s' "$val" | sed -E 's/^[[:space:]]+//')"

  case "$key" in
    kit_version) KIT_VERSION="$val" ;;
    language_mode) LANGUAGE_MODE="$val" ;;
    language_primary) LANGUAGE_PRIMARY="$val" ;;
    surface_rule)
      if [[ -z "$val" ]]; then fail "BD002" "$CONFIG_DISPLAY:$line_no" "surface_rule must not be empty"; else SURFACE_RULES+=("$val"); fi ;;
    feature_notes) FEATURE_NOTES="$val" ;;
    feature_empirical_records) FEATURE_EMPIRICAL_RECORDS="$val" ;;
    feature_archive_seal) FEATURE_ARCHIVE_SEAL="$val" ;;
    feature_budgets) FEATURE_BUDGETS="$val" ;;
    note_classes) NOTE_CLASSES="$val" ;;
    grandfathered_note)
      if [[ -z "$val" ]]; then fail "BD002" "$CONFIG_DISPLAY:$line_no" "grandfathered_note must not be empty"; else GRANDFATHERED_NOTES+=("$val"); fi ;;
    subtree)
      if [[ -z "$val" ]]; then fail "BD002" "$CONFIG_DISPLAY:$line_no" "subtree must not be empty"; else SUBTREES+=("$val"); fi ;;
    submodule_owned)
      if [[ -z "$val" ]]; then fail "BD002" "$CONFIG_DISPLAY:$line_no" "submodule_owned must not be empty"; else SUBMODULE_OWNED+=("$val"); fi ;;
    submodule_shared)
      if [[ -z "$val" ]]; then fail "BD002" "$CONFIG_DISPLAY:$line_no" "submodule_shared must not be empty"; else SUBMODULE_SHARED+=("$val"); fi ;;
    submodule_external)
      if [[ -z "$val" ]]; then fail "BD002" "$CONFIG_DISPLAY:$line_no" "submodule_external must not be empty"; else SUBMODULE_EXTERNAL+=("$val"); fi ;;
    budget)
      if [[ -z "$val" ]]; then fail "BD002" "$CONFIG_DISPLAY:$line_no" "budget must not be empty"; else BUDGETS+=("$val"); fi ;;
    hook_required)
      if [[ -z "$val" ]]; then fail "BD002" "$CONFIG_DISPLAY:$line_no" "hook_required must not be empty"; else HOOKS_REQUIRED+=("$val"); fi ;;
    hook_advisory)
      if [[ -z "$val" ]]; then fail "BD002" "$CONFIG_DISPLAY:$line_no" "hook_advisory must not be empty"; else HOOKS_ADVISORY+=("$val"); fi ;;
    *) fail "BD002" "$CONFIG_DISPLAY:$line_no" "unknown config key '$key'" ;;
  esac
done < "$CONFIG_PATH"

is_bool() { [[ "$1" == "true" || "$1" == "false" ]]; }

if [[ -z "$KIT_VERSION" ]]; then
  fail "BD002" "$CONFIG_DISPLAY" "missing required key 'kit_version'"
elif [[ ! "$KIT_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
  fail "BD002" "$CONFIG_DISPLAY" "invalid kit_version '$KIT_VERSION' (expected semver e.g. 0.1.0)"
fi
case "$LANGUAGE_MODE" in
  single-en|single-zh|split-by-surface|i18n-en-zh) ;;
  *) fail "BD002" "$CONFIG_DISPLAY" "invalid language_mode '$LANGUAGE_MODE'" ;;
esac
case "$LANGUAGE_PRIMARY" in
  en|zh) ;;
  *) fail "BD002" "$CONFIG_DISPLAY" "invalid language_primary '$LANGUAGE_PRIMARY'" ;;
esac
is_bool "$FEATURE_NOTES" || fail "BD002" "$CONFIG_DISPLAY" "feature_notes must be true or false"
is_bool "$FEATURE_EMPIRICAL_RECORDS" || fail "BD002" "$CONFIG_DISPLAY" "feature_empirical_records must be true or false"
is_bool "$FEATURE_ARCHIVE_SEAL" || fail "BD002" "$CONFIG_DISPLAY" "feature_archive_seal must be true or false"
is_bool "$FEATURE_BUDGETS" || fail "BD002" "$CONFIG_DISPLAY" "feature_budgets must be true or false"

if [[ "$FEATURE_EMPIRICAL_RECORDS" == "true" && "$FEATURE_NOTES" != "true" ]]; then
  fail "BD002" "$CONFIG_DISPLAY" "feature_empirical_records=true requires feature_notes=true"
fi
if [[ "$FEATURE_ARCHIVE_SEAL" == "true" && "$FEATURE_NOTES" != "true" ]]; then
  fail "BD002" "$CONFIG_DISPLAY" "feature_archive_seal=true requires feature_notes=true"
fi

# Validate note_classes unconditionally (reject empty, leading/trailing comma, non-kebab-case)
declare -A ALLOWED_CLASSES=()
if [[ -z "$NOTE_CLASSES" || "$NOTE_CLASSES" == ,* || "$NOTE_CLASSES" == *, ]]; then
  fail "BD002" "$CONFIG_DISPLAY" "invalid note_classes '$NOTE_CLASSES' (expected non-empty comma-separated lowercase kebab-case)"
else
  IFS=',' read -ra _cls_arr <<< "$NOTE_CLASSES"
  for c in "${_cls_arr[@]}"; do
    c="$(printf '%s' "$c" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
    if [[ -z "$c" || ! "$c" =~ ^[a-z0-9-]+$ ]]; then
      fail "BD002" "$CONFIG_DISPLAY" "invalid note class '$c' in note_classes (expected lowercase kebab-case)"
    else
      ALLOWED_CLASSES["$c"]=1
    fi
  done
fi

for sr in "${SURFACE_RULES[@]}"; do
  if [[ ! "$sr" =~ ^[^|]+\|(en|zh)\|(owner|derived)$ ]]; then
    fail "BD002" "$CONFIG_DISPLAY" "invalid surface_rule '$sr' (expected <glob>|<en|zh>|<owner|derived>)"
  fi
done

for gf in "${GRANDFATHERED_NOTES[@]}"; do
  if ! is_safe_rel_path "$gf"; then
    fail "BD002" "$CONFIG_DISPLAY" "invalid grandfathered_note path '$gf' (expected normalized repo-relative path)"
  fi
done

for entry in "${SUBMODULE_OWNED[@]}"; do
  if [[ ! "$entry" =~ ^[^|]+\|[^|]+\|[^|]+\|[^|]+$ ]]; then
    fail "BD002" "$CONFIG_DISPLAY" "invalid submodule_owned '$entry' (expected <path>|<repo_id>|<starting_sha>|<scope>)"
  else
    sub_p="${entry%%|*}"
    if ! is_safe_rel_path "$sub_p"; then
      fail "BD002" "$CONFIG_DISPLAY" "invalid submodule_owned path '$sub_p' (expected normalized repo-relative path without trailing '/')"
    fi
  fi
done

for entry in "${SUBMODULE_SHARED[@]}"; do
  if ! is_safe_rel_path "$entry"; then
    fail "BD002" "$CONFIG_DISPLAY" "invalid submodule_shared path '$entry' (expected normalized repo-relative path without trailing '/')"
  fi
done

for entry in "${SUBMODULE_EXTERNAL[@]}"; do
  if ! is_safe_rel_path "$entry"; then
    fail "BD002" "$CONFIG_DISPLAY" "invalid submodule_external path '$entry' (expected normalized repo-relative path without trailing '/')"
  fi
done

for b in "${BUDGETS[@]}"; do
  if [[ ! "$b" =~ ^[^:]+:[1-9][0-9]*$ ]]; then
    fail "BD002" "$CONFIG_DISPLAY" "invalid budget entry '$b' (expected <path>:<positive-int>)"
  else
    b_p="${b%:*}"
    if ! is_safe_rel_path "$b_p"; then
      fail "BD002" "$CONFIG_DISPLAY" "invalid budget path '$b_p' (expected normalized repo-relative path)"
    fi
  fi
done

for sub_dir in "${SUBTREES[@]}"; do
  if ! is_safe_rel_path "$sub_dir"; then
    fail "BD002" "$CONFIG_DISPLAY" "invalid subtree path '$sub_dir' (expected normalized repo-relative path)"
    continue
  fi
  if [[ ! -f "$ROOT/$sub_dir/AGENTS.md" ]]; then
    fail "BD003" "$sub_dir/AGENTS.md" "configured subtree is missing AGENTS.md"
  fi
done

# ---------------------------------------------------------------------------
# 2b. Resolve CI / baseline git ref (BD040)
# ---------------------------------------------------------------------------
EXPLICIT_BASE_REF=0
EFFECTIVE_BASE_REF="$BASE_REF"
if [[ -n "$BASE_REF" ]]; then
  EXPLICIT_BASE_REF=1
elif [[ -n "${BETTER_DOCS_BASE_REF:-}" ]]; then
  EFFECTIVE_BASE_REF="$BETTER_DOCS_BASE_REF"
  EXPLICIT_BASE_REF=1
fi

BASE_COMMIT=""
BASE_REF_FAILED=0
MANIFEST_REL=".agents/notes/archived/manifest.sha256"
if [[ "$CI_MODE" -eq 1 ]]; then
  if [[ -z "$EFFECTIVE_BASE_REF" ]]; then
    fail "BD040" "$MANIFEST_REL" "--ci requires --base-ref or BETTER_DOCS_BASE_REF"
    BASE_REF_FAILED=1
  else
    head_sha="$(git -C "$ROOT" rev-parse --verify HEAD 2>/dev/null || true)"
    base_sha="$(git -C "$ROOT" rev-parse --verify "${EFFECTIVE_BASE_REF}^{commit}" 2>/dev/null || true)"
    if [[ -z "$base_sha" ]]; then
      fail "BD040" "$MANIFEST_REL" "cannot resolve --base-ref '$EFFECTIVE_BASE_REF' to a commit"
      BASE_REF_FAILED=1
    elif [[ "$base_sha" == "$head_sha" ]]; then
      fail "BD040" "$MANIFEST_REL" "--ci --base-ref must be distinct from HEAD"
      BASE_REF_FAILED=1
    else
      BASE_COMMIT="$base_sha"
    fi
  fi
elif [[ "$IN_GIT" -eq 1 ]]; then
  if [[ "$EXPLICIT_BASE_REF" -eq 1 ]]; then
    base_sha="$(git -C "$ROOT" rev-parse --verify "${EFFECTIVE_BASE_REF}^{commit}" 2>/dev/null || true)"
    if [[ -z "$base_sha" ]]; then
      fail "BD040" "$MANIFEST_REL" "cannot resolve --base-ref '$EFFECTIVE_BASE_REF' to a commit"
      BASE_REF_FAILED=1
    else
      BASE_COMMIT="$base_sha"
    fi
  else
    BASE_COMMIT="$(git -C "$ROOT" rev-parse --verify "HEAD^{commit}" 2>/dev/null || true)"
  fi
fi

# ---------------------------------------------------------------------------
# 3. Submodule discovery and default-closed gitlink tracking (BD903)
# ---------------------------------------------------------------------------
declare -A GITLINK_PATHS=()
declare -A DECLARED_SUBMODULES=()

for entry in "${SUBMODULE_OWNED[@]}"; do
  sub_p="${entry%%|*}"
  is_safe_rel_path "$sub_p" && DECLARED_SUBMODULES["$sub_p"]="owned"
done
for entry in "${SUBMODULE_SHARED[@]}"; do
  is_safe_rel_path "$entry" && DECLARED_SUBMODULES["$entry"]="shared"
done
for entry in "${SUBMODULE_EXTERNAL[@]}"; do
  is_safe_rel_path "$entry" && DECLARED_SUBMODULES["$entry"]="external"
done

if [[ "$IN_GIT" -eq 1 && -f "$ROOT/.gitmodules" ]]; then
  while IFS= read -r gm_line; do
    sub_p="${gm_line#*.path }"
    [[ -z "$sub_p" ]] && continue
    GITLINK_PATHS["$sub_p"]=1
    if [[ -z "${DECLARED_SUBMODULES[$sub_p]:-}" ]]; then
      advise "BD903" "$sub_p" "SKIPPED (unlisted gitlink; default-closed)"
    fi
    if [[ ! -e "$ROOT/$sub_p/.git" ]]; then
      advise "BD903" "$sub_p" "SKIPPED (uninitialized)"
    fi
  done < <(git config --file "$ROOT/.gitmodules" --get-regexp '^submodule\..*\.path$' 2>/dev/null || true)
fi

is_excluded_path() {
  local p="$1"
  case "$p" in
    vendor/*|node_modules/*|.git/*|.agents/scratch/*|.agents/notes/archived/*) return 0 ;;
  esac
  for gp in "${!GITLINK_PATHS[@]}"; do
    if [[ "$p" == "$gp" || "$p" == "$gp/"* ]]; then
      return 0
    fi
  done
  for gp in "${!DECLARED_SUBMODULES[@]}"; do
    if [[ "$p" == "$gp" || "$p" == "$gp/"* ]]; then
      return 0
    fi
  done
  return 1
}

# Collect active Markdown files (sorted; skips symlinks; handles UTF-8 filenames via core.quotePath=false)
declare -a ACTIVE_MD_FILES=()
if [[ "$IN_GIT" -eq 1 ]]; then
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    [[ ! -f "$ROOT/$f" || -L "$ROOT/$f" ]] && continue
    is_excluded_path "$f" && continue
    ACTIVE_MD_FILES+=("$f")
  done < <(git -C "$ROOT" -c core.quotePath=false ls-files --cached --others --exclude-standard -- '*.md' | LC_ALL=C sort -u)
else
  while IFS= read -r f; do
    f="${f#./}"
    [[ -z "$f" ]] && continue
    [[ -L "$ROOT/$f" ]] && continue
    is_excluded_path "$f" && continue
    ACTIVE_MD_FILES+=("$f")
  done < <(cd "$ROOT" && find . -type f -name '*.md' | LC_ALL=C sort)
fi

# Detect concluded empirical study directories (used by archive seal and BD901 exclusion)
declare -A CONCLUDED_STUDY_DIRS=()
if [[ "$FEATURE_EMPIRICAL_RECORDS" == "true" && -d "$ROOT/.agents/records" && ! -L "$ROOT/.agents/records" ]]; then
  for study_dir in "$ROOT/.agents/records"/*; do
    [[ ! -d "$study_dir" || -L "$study_dir" ]] && continue
    study_readme=""
    [[ -f "$study_dir/README.md" && ! -L "$study_dir/README.md" ]] && study_readme="$study_dir/README.md"
    [[ -z "$study_readme" && -f "$study_dir/index.md" && ! -L "$study_dir/index.md" ]] && study_readme="$study_dir/index.md"
    [[ -z "$study_readme" ]] && continue
    line3="$(sed -n '3p' "$study_readme" | tr -d '\r')"
    if [[ "$line3" == "Status: concluded" ]]; then
      CONCLUDED_STUDY_DIRS["${study_dir#$ROOT/}"]=1
    fi
  done
fi

# ---------------------------------------------------------------------------
# 4. Token counter (EN whitespace tokens + 1 per CJK Unified Ideograph)
# ---------------------------------------------------------------------------
count_tokens() {
  local file="$1"
  LC_ALL=C awk '
    BEGIN {
      for (i = 0; i < 256; i++) ord[sprintf("%c", i)] = i
      tokens = 0
    }
    {
      s = $0
      n = length(s)
      i = 1
      in_word = 0
      while (i <= n) {
        c = substr(s, i, 1)
        b1 = ord[c]
        if (b1 >= 224 && b1 <= 239 && i + 2 <= n) {
          b2 = ord[substr(s, i + 1, 1)]
          b3 = ord[substr(s, i + 2, 1)]
          cp = (b1 - 224) * 4096 + (b2 - 128) * 64 + (b3 - 128)
          if ((cp >= 19968 && cp <= 40959) || (cp >= 13312 && cp <= 19903) || (cp >= 63744 && cp <= 64255)) {
            tokens++
            in_word = 0
            i += 3
            continue
          }
          if (!in_word) { tokens++; in_word = 1 }
          i += 3
          continue
        }
        if (c == " " || c == "\t" || c == "\r") {
          in_word = 0
        } else {
          if (!in_word) { tokens++; in_word = 1 }
        }
        i++
      }
    }
    END { print tokens + 0 }
  ' "$file"
}

# ---------------------------------------------------------------------------
# 5. Wordcount budgets (BD010)
# ---------------------------------------------------------------------------
if [[ "$FEATURE_BUDGETS" == "true" || "$LIST_BUDGETS" -eq 1 ]]; then
  for b in "${BUDGETS[@]}"; do
    [[ ! "$b" =~ ^[^:]+:[1-9][0-9]*$ ]] && continue
    b_path="${b%:*}"
    b_max="${b##*:}"
    is_safe_rel_path "$b_path" || continue
    if [[ ! -f "$ROOT/$b_path" ]]; then
      [[ "$LIST_BUDGETS" -eq 1 ]] && printf 'MISS  %6s / %-6s %s\n' "-" "$b_max" "$b_path"
      [[ "$FEATURE_BUDGETS" == "true" ]] && fail "BD010" "$b_path" "budgeted file does not exist"
      continue
    fi
    b_actual="$(count_tokens "$ROOT/$b_path")"
    if [[ "$LIST_BUDGETS" -eq 1 ]]; then
      status_lbl="ok  "
      (( b_actual > b_max )) && status_lbl="OVER"
      printf '%s  %6d / %-6d %s\n' "$status_lbl" "$b_actual" "$b_max" "$b_path"
    fi
    if [[ "$FEATURE_BUDGETS" == "true" ]] && (( b_actual > b_max )); then
      fail "BD010" "$b_path" "$b_actual tokens exceeds ceiling of $b_max (relocate, condense, or justify raise)"
    fi
  done
fi

# ---------------------------------------------------------------------------
# Shared CommonMark awk helpers (fence toggle + exact backtick-run code spans)
# ---------------------------------------------------------------------------
read -r -d '' AWK_CM_HELPERS <<'AWK_EOF' || true
function check_fence(line,    m, fc, fl, rest) {
  if (match(line, /^[ ]{0,3}(`{3,}|~{3,})/)) {
    m = substr(line, RSTART, RLENGTH)
    sub(/^[ ]+/, "", m)
    fc = substr(m, 1, 1)
    fl = length(m)
    rest = substr(line, RSTART + RLENGTH)
    if (!in_fence) {
      if (fc == "~" || index(rest, "`") == 0) {
        in_fence = 1; fence_char = fc; fence_len = fl
        return 1
      }
    } else if (fc == fence_char && fl >= fence_len && rest ~ /^[[:space:]]*$/) {
      in_fence = 0
      return 1
    }
  }
  return 0
}
function strip_code_spans(s,    out, n, i, k, j, ck, found) {
  out = ""
  n = length(s)
  i = 1
  while (i <= n) {
    if (substr(s, i, 1) == "`") {
      k = 0
      while (i + k <= n && substr(s, i + k, 1) == "`") k++
      j = i + k
      found = 0
      while (j <= n) {
        if (substr(s, j, 1) == "`") {
          ck = 0
          while (j + ck <= n && substr(s, j + ck, 1) == "`") ck++
          if (ck == k) {
            out = out "CODE"
            i = j + ck
            found = 1
            break
          }
          j += ck
        } else {
          j++
        }
      }
      if (!found) {
        out = out substr(s, i, k)
        i += k
      }
    } else {
      out = out substr(s, i, 1)
      i++
    }
  }
  return out
}
AWK_EOF

# ---------------------------------------------------------------------------
# 6. Heading anchor index & relative Markdown link checks (BD020, BD021)
# ---------------------------------------------------------------------------
extract_anchors() {
  local file="$1"
  LC_ALL=C awk "$AWK_CM_HELPERS"'
    BEGIN {
      for (i = 0; i < 256; i++) ord[sprintf("%c", i)] = i
      in_fence = 0; fence_char = ""; fence_len = 0
    }
    {
      line = $0
      sub(/\r$/, "", line)
      if (check_fence(line)) next
      if (in_fence) next
      if (line ~ /^#{1,6}[ \t]+/) {
        h = line
        sub(/^#{1,6}[ \t]+/, "", h)
        sub(/[ \t]+#+[ \t]*$/, "", h)
        while (match(h, /\[[^\]]*\]\([^)]*\)/)) {
          seg = substr(h, RSTART, RLENGTH)
          sub(/^\[/, "", seg)
          sub(/\]\([^)]*\)$/, "", seg)
          h = substr(h, 1, RSTART - 1) seg substr(h, RSTART + RLENGTH)
        }
        gsub(/[`*_]/, "", h)
        h = tolower(h)
        out = ""
        n = length(h)
        i = 1
        while (i <= n) {
          ch = substr(h, i, 1)
          b1 = ord[ch]
          if (b1 < 128) {
            if (ch ~ /[a-z0-9 -]/) out = out ch
            i++
          } else if (b1 >= 224 && b1 <= 239 && i + 2 <= n) {
            b2 = ord[substr(h, i + 1, 1)]
            b3 = ord[substr(h, i + 2, 1)]
            cp = (b1 - 224) * 4096 + (b2 - 128) * 64 + (b3 - 128)
            if ((cp >= 19968 && cp <= 40959) || (cp >= 13312 && cp <= 19903) || (cp >= 63744 && cp <= 64255)) {
              out = out substr(h, i, 3)
            }
            i += 3
          } else if (b1 >= 192 && b1 <= 223 && i + 1 <= n) {
            i += 2
          } else if (b1 >= 240 && i + 3 <= n) {
            i += 4
          } else {
            i++
          }
        }
        gsub(/^[ ]+|[ ]+$/, "", out)
        gsub(/ /, "-", out)
        if (out == "") next
        cnt = seen[out] + 0
        seen[out] = cnt + 1
        if (cnt == 0) print out
        else print out "-" cnt
      }
    }
  ' "$file"
}

declare -A ANCHOR_CACHE=()
has_anchor() {
  local target_file="$1"
  local want="$2"
  if [[ -z "${ANCHOR_CACHE[$target_file]+x}" ]]; then
    ANCHOR_CACHE["$target_file"]=$'\n'"$(extract_anchors "$target_file")"$'\n'
  fi
  [[ "${ANCHOR_CACHE[$target_file]}" == *$'\n'"$want"$'\n'* ]]
}

normalize_rel_path() {
  local base_dir="$1"
  local rel="$2"
  local combined
  if [[ "$rel" == /* ]]; then
    combined="${rel#/}"
  elif [[ -n "$base_dir" && "$base_dir" != "." ]]; then
    combined="$base_dir/$rel"
  else
    combined="$rel"
  fi
  P="$combined" awk '
    BEGIN {
      p = ENVIRON["P"]
      n = split(p, parts, "/")
      depth = 0
      for (i = 1; i <= n; i++) {
        seg = parts[i]
        if (seg == "" || seg == ".") continue
        if (seg == "..") {
          if (depth == 0) { print "__ESCAPE__"; exit }
          depth--
        } else {
          stack[++depth] = seg
        }
      }
      out = ""
      for (i = 1; i <= depth; i++) {
        out = (i == 1 ? stack[i] : out "/" stack[i])
      }
      print out
    }
  '
}

for rel_file in "${ACTIVE_MD_FILES[@]}"; do
  abs_file="$ROOT/$rel_file"
  file_dir="$(dirname "$rel_file")"
  while IFS=$'\t' read -r lno raw_target; do
    [[ -z "$raw_target" ]] && continue
    target="$(printf '%s' "$raw_target" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+("[^"]*"|'\''[^'\'']*'\'')[[:space:]]*$//; s/[[:space:]]+$//')"
    case "$target" in
      ""|"#"|http://*|https://*|mailto:*|tel:*|data:*|file://*|"<"*) continue ;;
    esac
    if [[ "$target" == *"{{"* ]]; then continue; fi

    link_path="${target%%#*}"
    link_anchor=""
    if [[ "$target" == *#* ]]; then
      link_anchor="${target#*#}"
    fi

    resolved=""
    if [[ -z "$link_path" ]]; then
      resolved="$rel_file"
    else
      resolved="$(normalize_rel_path "$file_dir" "$link_path")"
      if [[ "$resolved" == "__ESCAPE__" ]]; then
        fail "BD020" "$rel_file:$lno" "link '$target' escapes repository root"
        continue
      fi
    fi

    if [[ ! -e "$ROOT/$resolved" ]]; then
      fail "BD020" "$rel_file:$lno" "broken relative link '$target' (missing '$resolved')"
      continue
    fi

    if [[ -n "$link_anchor" && -f "$ROOT/$resolved" && "$resolved" == *.md ]]; then
      norm_anchor="$(printf '%s' "$link_anchor" | LC_ALL=C tr '[:upper:]' '[:lower:]')"
      if ! has_anchor "$ROOT/$resolved" "$norm_anchor"; then
        fail "BD021" "$rel_file:$lno" "missing anchor '#$link_anchor' in '$resolved'"
      fi
    fi
  done < <(LC_ALL=C awk "$AWK_CM_HELPERS"'
    function scan_links(s, nr,    n, i, j, k, depth, close_b, pdepth, tend, ch, inner, target) {
      n = length(s)
      i = 1
      while (i <= n) {
        if (substr(s, i, 1) == "\\") { i += 2; continue }
        if (substr(s, i, 1) == "[") {
          depth = 1; close_b = 0
          for (j = i + 1; j <= n; j++) {
            ch = substr(s, j, 1)
            if (ch == "\\") { j++; continue }
            if (ch == "[") depth++
            else if (ch == "]") {
              depth--
              if (depth == 0) { close_b = j; break }
            }
          }
          if (close_b > 0) {
            inner = substr(s, i + 1, close_b - i - 1)
            if (close_b + 1 <= n && substr(s, close_b + 1, 1) == "(") {
              pdepth = 1; tend = 0
              for (k = close_b + 2; k <= n; k++) {
                ch = substr(s, k, 1)
                if (ch == "\\") { k++; continue }
                if (ch == "(") pdepth++
                else if (ch == ")") {
                  pdepth--
                  if (pdepth == 0) { tend = k; break }
                }
              }
              if (tend > 0) {
                if (index(inner, "[") > 0) scan_links(inner, nr)
                target = substr(s, close_b + 2, tend - close_b - 2)
                printf "%d\t%s\n", nr, target
                i = tend + 1
                continue
              }
            }
          }
        }
        i++
      }
    }
    BEGIN { in_fence = 0; fence_char = ""; fence_len = 0 }
    {
      line = $0
      sub(/\r$/, "", line)
      if (check_fence(line)) next
      if (in_fence) next
      scan_links(strip_code_spans(line), NR)
    }
  ' "$abs_file")
done

# ---------------------------------------------------------------------------
# 7. Agent Note tree and format checks (BD030–BD033)
# ---------------------------------------------------------------------------
if [[ "$FEATURE_NOTES" == "true" ]]; then
  declare -A GRANDFATHER_SET=()
  for gf in "${GRANDFATHERED_NOTES[@]}"; do
    is_safe_rel_path "$gf" && GRANDFATHER_SET["$gf"]=1
  done

  for rel_file in "${ACTIVE_MD_FILES[@]}"; do
    [[ "$rel_file" != .agents/notes/* ]] && continue
    case "$rel_file" in
      .agents/notes/README.md|.agents/notes/README.zh.md|.agents/notes/AGENTS.md|.agents/notes/*/AGENTS.md) continue ;;
    esac
    if [[ "$LANGUAGE_MODE" == "i18n-en-zh" && "$rel_file" == *.zh.md ]]; then
      continue
    fi

    sub="${rel_file#.agents/notes/}"
    if [[ ! "$sub" =~ ^(proposed|implemented|concluded|rejected)/([a-z0-9-]+)/([0-9]{4}-[0-9]{2}-[0-9]{2}-[a-z0-9-]+)\.md$ ]]; then
      fail "BD030" "$rel_file" "note path must match .agents/notes/{proposed,implemented,concluded,rejected}/{class}/YYYY-MM-DD-slug.md"
      continue
    fi
    lifecycle="${BASH_REMATCH[1]}"
    nclass="${BASH_REMATCH[2]}"
    if [[ "$lifecycle" == "concluded" && "$FEATURE_EMPIRICAL_RECORDS" != "true" ]]; then
      fail "BD030" "$rel_file" "concluded/ lifecycle requires feature_empirical_records=true"
    fi
    if [[ -z "${ALLOWED_CLASSES[$nclass]:-}" ]]; then
      fail "BD030" "$rel_file" "unknown note class '$nclass' (allowed: $NOTE_CLASSES)"
    fi

    is_gf=0
    [[ -n "${GRANDFATHER_SET[$rel_file]:-}" ]] && is_gf=1

    while IFS=$'\t' read -r diag_id diag_msg; do
      [[ -n "$diag_id" ]] && fail "$diag_id" "$rel_file" "$diag_msg"
    done < <(LC_ALL=C awk -v lifecycle="$lifecycle" -v is_gf="$is_gf" "$AWK_CM_HELPERS"'
      BEGIN { in_fence = 0; fence_char = ""; fence_len = 0; status_hits = 0; h2_count = 0 }
      {
        line = $0
        sub(/\r$/, "", line)
        lines[NR] = line
        if (check_fence(line)) next
        if (in_fence) next
        if (line ~ /^Status:/) status_hits++
        if (line ~ /^## /) {
          h = line
          sub(/[[:space:]]+$/, "", h)
          h2[++h2_count] = h
          has_h2[h] = 1
        }
      }
      END {
        if (lines[1] !~ /^# Agent Note: [^[:space:]]/) {
          printf "BD031\tline 1 must be `# Agent Note: <title>`\n"
        }
        if (NR < 2 || lines[2] != "") {
          printf "BD031\tline 2 must be blank\n"
        }
        status_ok = 0
        if (lifecycle == "proposed" && lines[3] == "Status: proposed") status_ok = 1
        if (lifecycle == "implemented" && lines[3] == "Status: implemented") status_ok = 1
        if (lifecycle == "concluded" && lines[3] == "Status: concluded") status_ok = 1
        if (lifecycle == "rejected" && lines[3] ~ /^Status: rejected — [^[:space:]]/) status_ok = 1
        if (!status_ok) {
          printf "BD031\tline 3 does not match %s status grammar\n", lifecycle
        }
        if (NR < 4 || lines[4] != "") {
          printf "BD031\tline 4 must be blank\n"
        }
        if (status_hits > 1) {
          printf "BD031\tmultiple Status: lines found outside fenced code\n"
        }

        if (lifecycle == "concluded") {
          req_n = split("## Hypothesis / Question|## Evidence|## Conclusion|## Direction impact", req, "|")
        } else if (lifecycle == "proposed") {
          req_n = split("## Problem|## Proposal|## Acceptance criteria|## Risks", req, "|")
        } else if (lifecycle == "implemented") {
          req_n = split("## Problem|## Decision|## Consequences", req, "|")
        } else {
          req_n = split("## Problem|## Proposal", req, "|")
        }

        for (i = 1; i <= req_n; i++) {
          if (!has_h2[req[i]]) {
            printf "BD032\tmissing required section `%s`\n", req[i]
          }
        }
        if (!is_gf && !has_h2["## Alternatives considered"]) {
          printf "BD032\tmissing required section `## Alternatives considered`\n"
        }

        if (lifecycle == "implemented") {
          for (i = 1; i <= h2_count; i++) {
            low = tolower(h2[i])
            if (low ~ /^## (proposal|plan|migration plan|acceptance criteria)($| )/) {
              printf "BD033\tproposal-era heading `%s` is forbidden in implemented/\n", h2[i]
            }
          }
        }
      }
    ' "$ROOT/$rel_file")
  done
fi

# ---------------------------------------------------------------------------
# 8. Append-only frozen archive seal (BD040–BD043)
# ---------------------------------------------------------------------------
if [[ "$FEATURE_ARCHIVE_SEAL" == "true" ]]; then
  seal_pre_failures="$FAILURES"
  ARCHIVE_DIR="$ROOT/.agents/notes/archived"
  MANIFEST_ABS="$ROOT/$MANIFEST_REL"

  # Reject symlinked root archive/records directories (BD043)
  declare -A DISK_SEALED=()
  if [[ -L "$ARCHIVE_DIR" ]]; then
    fail "BD043" ".agents/notes/archived" "symlinks are forbidden for sealed archive directory"
  elif [[ -d "$ARCHIVE_DIR" ]]; then
    while IFS= read -r entry; do
      [[ -z "$entry" ]] && continue
      rel="${entry#$ROOT/}"
      if [[ -L "$entry" ]]; then
        fail "BD043" "$rel" "symlinks are forbidden inside sealed archive scope"
        continue
      fi
      case "$rel" in
        .agents/notes/archived/AGENTS.md|"$MANIFEST_REL") continue ;;
        .agents/notes/archived/.manifest.sha256.tmp.*)
          sub_tmp="${rel#.agents/notes/archived/.manifest.sha256.tmp.}"
          [[ "$sub_tmp" != */* ]] && continue
          ;;
      esac
      [[ -f "$entry" ]] && DISK_SEALED["$rel"]="$(hash_file "$entry")"
    done < <(find "$ARCHIVE_DIR" -mindepth 1 \( -type f -o -type l \) | LC_ALL=C sort)
  fi

  if [[ "$FEATURE_EMPIRICAL_RECORDS" == "true" ]]; then
    if [[ -L "$ROOT/.agents/records" ]]; then
      fail "BD043" ".agents/records" "symlinks are forbidden for records directory"
    elif [[ -d "$ROOT/.agents/records" ]]; then
      # Reject every symlink anywhere under .agents/records/
      while IFS= read -r lentry; do
        [[ -n "$lentry" ]] && fail "BD043" "${lentry#$ROOT/}" "symlinks are forbidden inside .agents/records/ scope"
      done < <(find "$ROOT/.agents/records" -mindepth 1 -type l | LC_ALL=C sort)

      for study_rel in "${!CONCLUDED_STUDY_DIRS[@]}"; do
        while IFS= read -r entry; do
          [[ -z "$entry" ]] && continue
          rel="${entry#$ROOT/}"
          [[ -f "$entry" && ! -L "$entry" ]] && DISK_SEALED["$rel"]="$(hash_file "$entry")"
        done < <(find "$ROOT/$study_rel" -mindepth 1 -type f | LC_ALL=C sort)
      done
    fi
  fi

  # Parse current manifest.sha256 (only if regular non-symlink file)
  declare -A CUR_MANIFEST=()
  if [[ -L "$MANIFEST_ABS" ]]; then
    : # Already failed BD043 above
  elif [[ -e "$MANIFEST_ABS" && ! -f "$MANIFEST_ABS" ]]; then
    fail "BD043" "$MANIFEST_REL" "archive manifest must be a regular file"
  elif [[ -f "$MANIFEST_ABS" ]]; then
    mlno=0
    while IFS= read -r mline || [[ -n "$mline" ]]; do
      mlno=$((mlno + 1))
      mline="${mline%$'\r'}"
      [[ -z "$mline" ]] && continue
      if [[ ! "$mline" =~ ^([0-9a-f]{64})\ \ (.+)$ ]]; then
        fail "BD042" "$MANIFEST_REL:$mlno" "malformed manifest line (expected '<64-hex>  <path>')"
        continue
      fi
      mhash="${BASH_REMATCH[1]}"
      mpath="${BASH_REMATCH[2]}"
      if ! is_safe_rel_path "$mpath"; then
        fail "BD043" "$MANIFEST_REL:$mlno" "non-normalized path '$mpath'"
        continue
      fi
      if [[ -n "${CUR_MANIFEST[$mpath]+x}" ]]; then
        fail "BD042" "$MANIFEST_REL:$mlno" "duplicate manifest entry '$mpath'"
        continue
      fi
      CUR_MANIFEST["$mpath"]="$mhash"
    done < "$MANIFEST_ABS"
  elif [[ "$SEAL_MODE" -eq 0 ]]; then
    fail "BD040" "$MANIFEST_REL" "missing archive manifest (run --seal to initialize)"
  fi

  # Verify append-only extension relative to baseline commit (BD041)
  if [[ -n "$BASE_COMMIT" ]]; then
    if git -C "$ROOT" cat-file -e "${BASE_COMMIT}:${MANIFEST_REL}" 2>/dev/null; then
      while IFS= read -r bline || [[ -n "$bline" ]]; do
        bline="${bline%$'\r'}"
        [[ -z "$bline" ]] && continue
        if [[ "$bline" =~ ^([0-9a-f]{64})\ \ (.+)$ ]]; then
          bhash="${BASH_REMATCH[1]}"
          bpath="${BASH_REMATCH[2]}"
          if [[ -z "${CUR_MANIFEST[$bpath]:-}" ]]; then
            fail "BD041" "$bpath" "sealed entry from baseline ($BASE_COMMIT) was removed"
          elif [[ "${CUR_MANIFEST[$bpath]}" != "$bhash" ]]; then
            fail "BD041" "$bpath" "sealed digest from baseline ($BASE_COMMIT) was modified"
          fi
        fi
      done < <(git -C "$ROOT" show "${BASE_COMMIT}:${MANIFEST_REL}")
    fi
  fi

  # Verify existing manifest entries match disk
  for mpath in "${!CUR_MANIFEST[@]}"; do
    if [[ -z "${DISK_SEALED[$mpath]:-}" ]]; then
      fail "BD042" "$mpath" "manifest entry has no corresponding sealed file on disk"
    elif [[ "${DISK_SEALED[$mpath]}" != "${CUR_MANIFEST[$mpath]}" ]]; then
      fail "BD042" "$mpath" "sealed file digest mismatch (expected ${CUR_MANIFEST[$mpath]}, got ${DISK_SEALED[$mpath]})"
    fi
  done

  if [[ "$SEAL_MODE" -eq 1 ]]; then
    if [[ "$BASE_REF_FAILED" -eq 0 && "$FAILURES" -eq "$seal_pre_failures" ]]; then
      if [[ -L "$MANIFEST_ABS" || ( -e "$MANIFEST_ABS" && ! -f "$MANIFEST_ABS" ) ]]; then
        fail "BD043" "$MANIFEST_REL" "archive manifest destination must be a regular file"
      elif ! mkdir -p "$ARCHIVE_DIR" 2>/dev/null; then
        fail "BD040" "$MANIFEST_REL" "failed to create archive directory '.agents/notes/archived'"
      else
        tmp_manifest="$(mktemp "$ARCHIVE_DIR/.manifest.sha256.tmp.XXXXXX" 2>/dev/null || true)"
        seal_replaced=0
        if [[ -n "$tmp_manifest" ]] && {
          for sp in "${!DISK_SEALED[@]}"; do
            printf '%s  %s\n' "${DISK_SEALED[$sp]}" "$sp"
          done
        } | LC_ALL=C sort -k1.67 > "$tmp_manifest" 2>/dev/null; then
          if mv --help 2>&1 | grep -Fq -- '--no-target-directory'; then
            mv -T -f "$tmp_manifest" "$MANIFEST_ABS" 2>/dev/null && [[ -f "$MANIFEST_ABS" && ! -L "$MANIFEST_ABS" ]] && seal_replaced=1
          else
            [[ ! -d "$MANIFEST_ABS" && ! -L "$MANIFEST_ABS" ]] && mv -f "$tmp_manifest" "$MANIFEST_ABS" 2>/dev/null && [[ -f "$MANIFEST_ABS" && ! -L "$MANIFEST_ABS" ]] && seal_replaced=1
          fi
        fi
        if [[ "$seal_replaced" -ne 1 ]]; then
          [[ -n "$tmp_manifest" ]] && rm -f "$tmp_manifest" 2>/dev/null || true
          fail "BD040" "$MANIFEST_REL" "failed to write archive manifest atomically"
        fi
      fi
    fi
  else
    for sp in "${!DISK_SEALED[@]}"; do
      if [[ -z "${CUR_MANIFEST[$sp]:-}" ]]; then
        fail "BD042" "$sp" "sealed file is missing from $MANIFEST_REL (run --seal)"
      fi
    done
  fi
fi

# ---------------------------------------------------------------------------
# 9. Mechanical prose hygiene (BD050) and draft markers (BD051)
# ---------------------------------------------------------------------------
for rel_file in "${ACTIVE_MD_FILES[@]}"; do
  abs_file="$ROOT/$rel_file"
  if [[ -s "$abs_file" ]]; then
    last_byte="$(tail -c 1 "$abs_file" | od -An -tx1 | tr -d ' \n')"
    if [[ "$last_byte" != "0a" ]]; then
      fail "BD050" "$rel_file" "missing trailing newline at EOF"
    fi
  fi

  while IFS=$'\t' read -r diag_id lno diag_msg; do
    [[ -n "$diag_id" ]] && fail "$diag_id" "$rel_file:$lno" "$diag_msg"
  done < <(LC_ALL=C awk "$AWK_CM_HELPERS"'
    BEGIN { in_fence = 0; fence_char = ""; fence_len = 0 }
    {
      raw = $0
      sub(/\r$/, "", raw)
      if (raw ~ /[ \t]+$/) {
        printf "BD050\t%d\ttrailing whitespace\n", NR
      }
      if (check_fence(raw)) next
      if (in_fence) next
      s = strip_code_spans(raw)
      if (s ~ /\(decision [0-9]+\)|\(audit [A-Z][0-9]+\)/) {
        printf "BD051\t%d\tunresolved internal draft marker in prose\n", NR
      }
    }
  ' "$abs_file")
done

# ---------------------------------------------------------------------------
# 10. i18n-en-zh pairing check (BD060)
# ---------------------------------------------------------------------------
if [[ "$LANGUAGE_MODE" == "i18n-en-zh" ]]; then
  declare -A ACTIVE_MD_SET=()
  for rel_file in "${ACTIVE_MD_FILES[@]}"; do
    ACTIVE_MD_SET["$rel_file"]=1
  done

  for rel_file in "${ACTIVE_MD_FILES[@]}"; do
    case "$rel_file" in
      *.zh.md|AGENTS.md|CLAUDE.md|*/AGENTS.md|*/CLAUDE.md|ROADMAP.md|.agents/skills/*) continue ;;
    esac
    stem="${rel_file%.md}"
    zh_rel="${stem}.zh.md"
    sidecar_rel="${stem}.i18n.yaml"
    if [[ -L "$ROOT/$zh_rel" ]]; then
      fail "BD060" "$rel_file" "Chinese counterpart '$zh_rel' must be a regular non-symlink file"
    elif [[ ! -f "$ROOT/$zh_rel" ]]; then
      fail "BD060" "$rel_file" "missing Chinese counterpart '$zh_rel'"
    elif [[ -z "${ACTIVE_MD_SET[$zh_rel]:-}" ]]; then
      fail "BD060" "$rel_file" "Chinese counterpart '$zh_rel' is excluded or gitignored"
    fi
    if [[ -L "$ROOT/$sidecar_rel" ]]; then
      fail "BD060" "$rel_file" "i18n sidecar '$sidecar_rel' must be a regular non-symlink file"
      continue
    elif [[ ! -f "$ROOT/$sidecar_rel" ]]; then
      fail "BD060" "$rel_file" "missing i18n sidecar '$sidecar_rel'"
      continue
    elif [[ "$IN_GIT" -eq 1 && "$(git -C "$ROOT" -c core.quotePath=false ls-files --cached --others --exclude-standard -- ":(literal)$sidecar_rel")" != "$sidecar_rel" ]]; then
      fail "BD060" "$rel_file" "i18n sidecar '$sidecar_rel' is gitignored"
      continue
    fi
    recorded_sha="$(LC_ALL=C awk '
      /^[[:space:]]*source_sha256[[:space:]]*:/ {
        v = $0
        sub(/^[[:space:]]*source_sha256[[:space:]]*:[[:space:]]*/, "", v)
        sub(/\r$/, "", v)
        sub(/[[:space:]]+#.*$/, "", v)
        sub(/[[:space:]]+$/, "", v)
        print v
        exit
      }
    ' "$ROOT/$sidecar_rel")"
    actual_sha="$(hash_file "$ROOT/$rel_file")"
    if [[ -z "$recorded_sha" ]]; then
      fail "BD060" "$sidecar_rel" "missing source_sha256 field"
    elif [[ "$recorded_sha" != "$actual_sha" ]]; then
      fail "BD060" "$sidecar_rel" "stale translation sidecar (source_sha256=$recorded_sha, actual=$actual_sha)"
    fi
  done
fi

# ---------------------------------------------------------------------------
# 11. Advisory EN/ZH lexical CoT/slop probes (--probes -> BD901)
# ---------------------------------------------------------------------------
is_concluded_record_file() {
  local p="$1"
  for sd in "${!CONCLUDED_STUDY_DIRS[@]}"; do
    if [[ "$p" == "$sd/"* ]]; then
      return 0
    fi
  done
  return 1
}

if [[ "$PROBES_MODE" -eq 1 ]]; then
  for rel_file in "${ACTIVE_MD_FILES[@]}"; do
    case "$rel_file" in
      skills/prose-standard/references/*|.agents/skills/prose-standard/references/*) continue ;;
    esac
    is_concluded_record_file "$rel_file" && continue
    while IFS=$'\t' read -r lno hit; do
      [[ -n "$lno" ]] && advise "BD901" "$rel_file:$lno" "advisory probe match: $hit"
    done < <(LC_ALL=C awk "$AWK_CM_HELPERS"'
      BEGIN { in_fence = 0; fence_char = ""; fence_len = 0 }
      {
        raw = $0
        sub(/\r$/, "", raw)
        if (check_fence(raw)) next
        if (in_fence) next
        s = strip_code_spans(raw)
        low = tolower(s)
        if (match(low, /(^|[^a-z])(used to|this pr|previously|no longer|in conclusion|it is worth noting|it is important to note|seamless|seamlessly|effortless)([^a-z]|$)/)) {
          m = substr(low, RSTART, RLENGTH)
          sub(/^[^a-z]+/, "", m)
          sub(/[^a-z]+$/, "", m)
          printf "%d\t%s\n", NR, m
        }
        if (match(s, /(以前|不再|旧版|上一轮|评审|需要注意的是|值得一提的是|不仅如此|总而言之|综上所述)/)) {
          printf "%d\t%s\n", NR, substr(s, RSTART, RLENGTH)
        }
      }
    ' "$ROOT/$rel_file")
  done
fi

# ---------------------------------------------------------------------------
# 12. External hooks: required (BD070) vs advisory (BD902)
# ---------------------------------------------------------------------------
find_hook_config() {
  local root="$1"
  shift
  local rel
  for rel in "$@"; do
    if [[ -f "$root/$rel" ]]; then
      printf '%s\n' "$root/$rel"
      return 0
    fi
  done
  return 1
}

escape_lychee_glob() {
  local s="$1"
  if [[ "$s" != *[*?\[\]]* ]]; then
    printf '%s' "$s"
    return
  fi
  local out="" i ch
  for ((i = 0; i < ${#s}; i++)); do
    ch="${s:i:1}"
    case "$ch" in
      '*') out+='[*]' ;;
      '?') out+='[?]' ;;
      '[') out+='[[]' ;;
      ']') out+='[]]' ;;
      *)   out+="$ch" ;;
    esac
  done
  printf '%s' "$out"
}

LYCHEE_FILES=()
VALE_FILES=()
for rel_file in "${ACTIVE_MD_FILES[@]}"; do
  LYCHEE_FILES+=("$(escape_lychee_glob "$ROOT/$rel_file")")
  case "$rel_file" in
    skills/prose-standard/references/*|.agents/skills/prose-standard/references/*) continue ;;
  esac
  is_concluded_record_file "$rel_file" && continue
  VALE_FILES+=("$ROOT/$rel_file")
done

invoke_hook() {
  local mode="$1"
  local tool="$2"
  local cfg=""
  local hook_out=""
  local rc=0

  if ! command -v "$tool" >/dev/null 2>&1; then
    if [[ "$mode" == "required" ]]; then
      fail "BD070" "$tool" "required hook '$tool' is not installed on PATH"
    else
      advise "BD902" "$tool" "SKIPPED (not installed)"
    fi
    return
  fi

  case "$tool" in
    lychee)
      [[ "${#LYCHEE_FILES[@]}" -eq 0 ]] && return
      cfg="$(find_hook_config "$ROOT" ".lychee.toml" "lychee.toml" "gates/optional/lychee.toml" || true)"
      if [[ -n "$cfg" ]]; then
        (cd "$ROOT" && lychee --no-progress --offline --root-dir "$ROOT" --config "$cfg" "${LYCHEE_FILES[@]}") >/dev/null 2>&1 || rc=$?
      else
        (cd "$ROOT" && lychee --no-progress --offline --root-dir "$ROOT" "${LYCHEE_FILES[@]}") >/dev/null 2>&1 || rc=$?
      fi
      if [[ "$rc" -ne 0 ]]; then
        if [[ "$mode" == "required" ]]; then
          fail "BD070" "lychee" "required hook 'lychee' failed"
        else
          advise "BD902" "lychee" "advisory hook 'lychee' reported warnings"
        fi
      fi
      ;;
    vale)
      cfg="$(find_hook_config "$ROOT" ".vale.ini" "vale.ini" "gates/optional/vale.ini" || true)"
      if [[ -z "$cfg" ]]; then
        if [[ "$mode" == "required" ]]; then
          fail "BD070" "vale" "required hook 'vale' has no config (.vale.ini, vale.ini, or gates/optional/vale.ini)"
        else
          advise "BD902" "vale" "SKIPPED (no vale config found)"
        fi
        return
      fi
      [[ "${#VALE_FILES[@]}" -eq 0 ]] && return
      hook_out="$((cd "$ROOT" && vale --config "$cfg" --output=line "${VALE_FILES[@]}") 2>&1)" || rc=$?
      if [[ -n "$hook_out" ]]; then
        while IFS= read -r hline; do
          [[ -z "$hline" ]] && continue
          hline="${hline#"$ROOT/"}"
          advise "BD902" "vale" "$hline"
        done <<< "$hook_out"
      fi
      if [[ "$rc" -ne 0 ]]; then
        if [[ "$mode" == "required" ]]; then
          fail "BD070" "vale" "required hook 'vale' failed"
        else
          advise "BD902" "vale" "advisory hook 'vale' reported warnings"
        fi
      fi
      ;;
    *)
      (cd "$ROOT" && "$tool" "$ROOT") >/dev/null 2>&1 || rc=$?
      if [[ "$rc" -ne 0 ]]; then
        if [[ "$mode" == "required" ]]; then
          fail "BD070" "$tool" "required hook '$tool' failed"
        else
          advise "BD902" "$tool" "advisory hook '$tool' reported warnings"
        fi
      fi
      ;;
  esac
}

for h in "${HOOKS_REQUIRED[@]}"; do
  invoke_hook "required" "$h"
done

for h in "${HOOKS_ADVISORY[@]}"; do
  invoke_hook "advisory" "$h"
done

if [[ "$FAILURES" -gt 0 ]]; then
  printf 'check-docs: FAILED (%d violation(s))\n' "$FAILURES" >&2
  exit 1
fi

printf 'check-docs: OK (%d active markdown file(s) checked)\n' "${#ACTIVE_MD_FILES[@]}"
exit 0
