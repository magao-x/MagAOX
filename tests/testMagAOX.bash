#!/bin/bash
# Run all prebuilt tests listed in tests.list, then the generated logger suites.
# Invoke from any directory; build the tests first with make -C tests.
# Save both output streams and per-command exit codes under tests/results/.
# Keep previous runs and point results/latest at the current run.
# Continue after test failures and return nonzero if any test or log write fails.

set +e
set -uo pipefail

tests_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
cd "$tests_dir" || exit 1

mkdir -p results || exit 1
run_dir=$(mktemp -d "$tests_dir/results/$(date +%Y%m%d-%H%M%S).XXXXXX") || exit 1
ln -sfnT -- "${run_dir##*/}" results/latest || exit 1
run_log="$run_dir/run.log"
summary="$run_dir/summary.tsv"
printf 'result\texit_code\ttest\n' > "$summary" || exit 1
printf 'Running MagAO-X tests\nResults: %s\n\n' "$run_dir" | tee "$run_log" || exit 1

passed=0
failed=0

# Run a named command, tee both streams, and record its exit code before continuing.
# Arguments: summary label, executable, then any executable arguments.
run_test()
{
    local label=$1
    shift
    printf '\nrunning %s\n' "$label" | tee -a "$run_log" || exit 1

    { "$@"; } 2>&1 | tee -a "$run_log"
    local statuses=("${PIPESTATUS[@]}")
    local status=${statuses[0]}
    local result=PASS
    if (( status != 0 )); then
        result=FAIL
        failed=$((failed + 1))
    else
        passed=$((passed + 1))
    fi
    printf '%s\t%d\t%s\n' "$result" "$status" "$label" >> "$summary" || exit 1
    if (( statuses[1] != 0 )); then
        printf 'Could not save complete test output to %s\n' "$run_log" >&2
        exit 1
    fi
    printf '%s (exit %d): %s\n' "$result" "$status" "$label" | tee -a "$run_log" || exit 1
}

if [[ ! -r tests.list ]]; then
    printf 'Cannot read %s/tests.list\n' "$tests_dir" | tee -a "$run_log"
    exit 1
fi

# Accept CRLF lists without modifying the tracked input file.
while IFS= read -r test <&3 || [[ -n "$test" ]]; do
    test=${test%$'\r'}
    [[ -z "$test" || "$test" == \#* ]] && continue
    run_test "$test" "$test"
done 3< tests.list

# Run these independently so a generated-suite failure does not skip entropy tests.
run_test logger/catch2 make -C ../libMagAOX/logger/tests do_catch2_test COVERAGE=1
run_test logger/entropy make -C ../libMagAOX/logger/tests do_entropy_test COVERAGE=1

printf '\nSummary: %d passed, %d failed\n' "$passed" "$failed" | tee -a "$run_log" || exit 1
if (( failed != 0 )); then
    awk -F '\t' '$1 == "FAIL"' "$summary" | tee -a "$run_log" || exit 1
fi
printf 'Full output: %s\nSummary: %s\n' "$run_log" "$summary" | tee -a "$run_log" || exit 1
(( failed == 0 ))
