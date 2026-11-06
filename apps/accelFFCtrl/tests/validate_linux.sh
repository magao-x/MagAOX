#!/usr/bin/env bash
# \file validate_linux.sh
# \brief Build-only Linux qualification with private synthetic streams; never install or launch applications.
set -euo pipefail

if [[ $(uname -s) != Linux ]]; then
    echo 'Native Linux is required; no validation was performed.' >&2
    exit 2
fi
case $(hostname -s) in
    exao1|exao2|exao3|exao5)
        echo 'Use a configured non-instrument Linux development host for this validation phase.' >&2
        exit 2
        ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
accff_report=${1:-$(mktemp -d /tmp/accelFFCtrl-linux-report-XXXXXX)}
mkdir -p -- "$accff_report"
accff_report=$(cd -- "$accff_report" && pwd)
accff_cxx=${CXX:-g++}
accff_python=${PYTHON:-python3}
accff_seconds=${VALIDATION_SECONDS:-6}

# Record each exact command and stop on any compiler, test or pipeline failure.
run() {
    local accff_name=$1
    shift
    printf '%q ' "$@" > "$accff_report/$accff_name.command"
    printf '\n' >> "$accff_report/$accff_name.command"
    "$@" 2>&1 | tee "$accff_report/$accff_name.log"
}

run platform uname -a
run compiler "$accff_cxx" --version
run glibc getconf GNU_LIBC_VERSION
run versions pkg-config --modversion mxlib eigen3 ImageStreamIO
run isio-flags pkg-config --cflags --libs ImageStreamIO
run branch git branch --show-current
run revision git rev-parse HEAD
run worktree git status --short
git diff --binary > "$accff_report/source.patch"
git ls-files --others --exclude-standard > "$accff_report/untracked-files.txt"
git ls-files -co --exclude-standard -z apps/accelFFCtrl apps/accelSim \
    agents/plans/2026-11/accelSim-28hz-feedforward-controller.md | xargs -0 sha256sum > "$accff_report/source-sha256.txt"
# Force current headers into the translation units, and record selected dependency headers.
run observer-build make -C apps/accelFFCtrl "CXX=$accff_cxx -H" -W accelFFCtrl.hpp
run simulator-build make -C apps/accelSim "CXX=$accff_cxx -H" -W accelSim.hpp
run app-tests-build make -C tests -f Makefile.one "CXX=$accff_cxx" \
    t=../apps/accelFFCtrl/tests/accelFFCtrl_test.cpp -W ../apps/accelFFCtrl/tests/accelFFCtrl_test.cpp
run app-tests apps/accelFFCtrl/tests/accelFFCtrl_test '[accelFFCtrl][app]'
run controller-tests make -C apps/accelFFCtrl/tests "CXX=$accff_cxx" \
    "BUILD_DIR=$accff_report/bin" test benchmark
run native-tests make -C apps/accelFFCtrl/tests "CXX=$accff_cxx -H" \
    "BUILD_DIR=$accff_report/bin" native-test
run native-rate-tests make -C apps/accelFFCtrl/tests "CXX=$accff_cxx" \
    "BUILD_DIR=$accff_report/bin" native-rate-test
run observer-libraries ldd apps/accelFFCtrl/accelFFCtrl
run simulator-libraries ldd apps/accelSim/accelSim
run native-libraries ldd "$accff_report/bin/accelFFCtrlStream_test"
run replay env ACCELFFCTRL_TIMING_TRACE=1 make -C apps/accelFFCtrl/tests "CXX=$accff_cxx" "PYTHON=$accff_python" \
    "BUILD_DIR=$accff_report/bin" "VALIDATION_DIR=$accff_report/replay" \
    "VALIDATION_SECONDS=$accff_seconds" gap-validation
for accff_rate in 4000 8000; do
    for accff_policy in reset optional; do
        for accff_scenario in loss1_1s stamp_jitter2pct_loss1 stamp_jitter10pct_loss1; do
            accff_trace="$accff_report/replay/$accff_rate/$accff_policy/single28_${accff_scenario}_1000000"
            run "timing-$accff_rate-$accff_policy-$accff_scenario" env "MPLCONFIGDIR=$accff_report/.matplotlib" \
                "$accff_python" apps/accelFFCtrl/tests/analyze_timestamps.py "${accff_trace}_frames.csv" \
                --queries "${accff_trace}_queries.csv" --sample-rate "$accff_rate" \
                --output "$accff_report/timing/$accff_rate/$accff_policy/$accff_scenario" --plots
        done
    done
done
run simulator-python "$accff_python" -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py -v

# Only numerical simulator tests/mock DM sinks: never run the simulator's native DM tests.
accff_sources=(
    apps/accelSim/tests/accelSim_test.cpp apps/accelSim/tests/accelSimDisturbance_test.cpp
    apps/accelSim/tests/accelSimDM_test.cpp
    apps/accelFFCtrl/tests/accelFFCtrlModel_test.cpp apps/accelFFCtrl/tests/accelFFCtrlReplay_test.cpp
    apps/accelFFCtrl/tests/accelFFCtrlInput_test.cpp apps/accelFFCtrl/tests/accelFFCtrlValidation_test.cpp
    apps/accelFFCtrl/tests/accelFFCtrlGap_test.cpp apps/accelFFCtrl/tests/accelFFCtrlJitter_test.cpp
    apps/accelFFCtrl/tests/accelFFCtrlCancellation_test.cpp tests/testMain.cpp
)
read -r -a accff_eigen_flags <<< "$(pkg-config --cflags eigen3)"
run sanitizer-build "$accff_cxx" -std=c++20 -O1 -g -fno-fast-math -Wall -Wextra -pthread \
    -fsanitize=address,undefined -fno-omit-frame-pointer -DACCELSIM_MODEL_ONLY \
    "${accff_eigen_flags[@]}" "${accff_sources[@]}" -o "$accff_report/combined-asan"
run sanitizer env ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 "$accff_report/combined-asan"
run native-sanitizer env ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    make -C apps/accelFFCtrl/tests "CXX=$accff_cxx" "BUILD_DIR=$accff_report/asan" \
    'CXXFLAGS=-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' native-test
printf 'Build/private validation complete. Preserve report: %s\n' "$accff_report"
