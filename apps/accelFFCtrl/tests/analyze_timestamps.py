r"""\file analyze_timestamps.py
\brief Offline integer-nanosecond cadence, clock, gap-guard and query audit.
\ingroup accelFFCtrl_unit_test

No stream access. CSV schemas and capture limitations are in ../README.md.
Guard eligibility is necessary, not sufficient: this does not reconstruct the
estimator's maturity, last query, numerical conditioning or complete reset state.
"""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import statistics


FRAME_FIELDS = (
    "generation", "counter", "acquired_ns", "written_ns", "receipt_mono_ns",
    "clock_real_ns", "clock_mono_ns", "clock_uncertainty_ns", "depth", "accepted",
)
QUERY_FIELDS = ("query_ns", "target_ns", "sample_ns", "ready")


def read_csv(path, required):
    """Parse decimal integers directly; reject float epochs, missing cells and reordered receipts."""
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: missing columns {set(required)-set(reader.fieldnames or ())}")
        rows = []
        for line, raw in enumerate(reader, 2):
            try:
                row = {key: int(value) for key, value in raw.items()}
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line}: expected decimal integer fields") from error
            if any(row[key] < 0 for key in required if key in ("generation", "counter", "depth")):
                raise ValueError(f"{path}:{line}: negative source identity")
            if "accepted" in row and row["accepted"] not in (0, 1):
                raise ValueError(f"{path}:{line}: accepted must be 0 or 1")
            if "ready" in row and row["ready"] not in (0, 1):
                raise ValueError(f"{path}:{line}: ready must be 0 or 1")
            time_key = "receipt_mono_ns" if "receipt_mono_ns" in required else "query_ns"
            if rows and row[time_key] < rows[-1][time_key]:
                raise ValueError(f"{path}:{line}: decreasing {time_key}; split host/boot recordings")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty recording")
    return rows


def describe(values):
    """Summarize centered nanosecond values, without rounding epoch timestamps through float."""
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {"count": len(values), "min": ordered[0], "max": ordered[-1],
            "mean": statistics.fmean(values), "rms": math.sqrt(statistics.fmean(x*x for x in values)),
            "stddev": statistics.pstdev(values),
            "p50": statistics.median(values), "p99": ordered[math.ceil(.99*len(values))-1]}


def frame_analysis(rows, rate, max_age=2_000_000, max_interval=2_000_000, clock_tolerance=1_000_000):
    """Audit observed frames and necessary single-gap timing prerequisites, never infer actual admissions."""
    if not math.isfinite(rate) or rate <= 0 or min(max_age, max_interval, clock_tolerance) <= 0:
        raise ValueError("positive finite rate and timing limits required")
    period = 1e9/rate
    intervals, result, segments = [], [], []
    last, accepted, anchor, segment = None, None, None, []
    last_valid = False
    receipt_intervals = []
    missing = rollbacks = generations = duplicates = timing_faults = gaps = eligible = 0
    for index, row in enumerate(rows):
        new_generation = last is None or row["generation"] != last["generation"]
        delta = row["counter"]-last["counter"] if last and not new_generation else 0
        if new_generation or delta < 0:
            if segment:
                segments.append(segment)
            segment, accepted, anchor = [], None, None
            generations += int(last is not None and new_generation)
            rollbacks += int(not new_generation and delta < 0)
        elif last:
            missing += max(0, delta-1)
            duplicates += int(delta == 0)
        clock_ok = (0 < row["clock_real_ns"] <= 2**63-1 and 0 <= row["clock_mono_ns"] <= 2**63-1 and
                    0 <= row["clock_uncertainty_ns"] <= clock_tolerance)
        if anchor is not None:
            clock_ok &= (row["clock_mono_ns"] >= anchor["clock_mono_ns"] and
                         abs((row["clock_real_ns"]-anchor["clock_real_ns"])-
                             (row["clock_mono_ns"]-anchor["clock_mono_ns"])) <= clock_tolerance)
        if anchor is None and clock_ok:
            anchor = row
        mapped = (anchor["clock_mono_ns"] + row["acquired_ns"]-anchor["clock_real_ns"]
                  if anchor else -1)
        age = row["receipt_mono_ns"]-mapped
        valid = (clock_ok and 0 < row["acquired_ns"] <= 2**63-1 and row["depth"] > 0 and
                 0 < row["counter"] <= 2**64-1 and row["generation"] > 0 and
                 row["acquired_ns"] <= row["written_ns"] <= row["clock_real_ns"] and
                 row["clock_mono_ns"] == row["receipt_mono_ns"] and mapped >= 0 and
                 0 <= age <= max_age)
        if accepted is not None:
            valid &= mapped > accepted[1]
        if "adc_x" in row or "adc_y" in row:
            valid &= all(0 <= row.get(key, -1) <= 4095 for key in ("adc_x", "adc_y"))
        for prefix in ("acquired", "written"):
            if f"{prefix}_sec" in row or f"{prefix}_nsec" in row:
                sec, nsec = row.get(f"{prefix}_sec", -1), row.get(f"{prefix}_nsec", -1)
                valid &= sec >= 0 and 0 <= nsec < 1_000_000_000 and sec*1_000_000_000+nsec == row[f"{prefix}_ns"]
        timing_faults += int(not valid)
        if last and last_valid and valid and delta > 0:
            dt = row["acquired_ns"]-last["acquired_ns"]
            if dt > 0:
                intervals.append(dt/delta-period)
                receipt_intervals.append((row["receipt_mono_ns"]-last["receipt_mono_ns"])/delta-period)
        gap = accepted is not None and row["counter"]-accepted[0]["counter"] > 1
        gap_error = mapped-accepted[1]-2*period if gap else 0
        candidate = bool(gap and valid and row["depth"] == 1 and
                         row["counter"]-accepted[0]["counter"] == 2 and
                         0 < mapped-accepted[1] <= min(1_000_000, max_interval) and
                         row["receipt_mono_ns"]-accepted[1] <= max_age and
                         abs(gap_error) <= .05*period)
        gaps += int(gap)
        eligible += int(candidate)
        result.append({"row": index, "generation": row["generation"], "counter": row["counter"],
                       "mapped_ns": mapped, "sample_age_ns": age,
                       "write_delay_ns": row["written_ns"]-row["acquired_ns"],
                       "publication_to_receipt_ns": age-(row["written_ns"]-row["acquired_ns"]),
                       "clock_offset_ns": row["clock_real_ns"]-row["clock_mono_ns"],
                       "timing_valid": int(valid), "gap_candidate": int(gap),
                       "gap_error_ns": gap_error, "timing_eligible": int(candidate)})
        if valid:
            segment.append((row["counter"], row["acquired_ns"]))
        if row["accepted"] and valid:
            accepted = (row, mapped)
        elif not row["accepted"] or not valid:
            accepted = None  # Conservative: an omitted rejection/reset history cannot authorize retention.
        if not clock_ok:
            anchor = None
        last = row
        last_valid = valid
    if segment:
        segments.append(segment)
    drift = []
    for segment in segments:
        x = [counter-segment[0][0] for counter, _ in segment]
        y = [stamp-segment[0][1] for _, stamp in segment]
        xmean, ymean = statistics.fmean(x), statistics.fmean(y)
        xx = sum((value-xmean)**2 for value in x)
        if xx:
            slope = sum((a-xmean)*(b-ymean) for a, b in zip(x, y))/xx
            drift.append((slope/period-1)*1e6)
    summary = {"records": len(rows), "observed_missing_lower_bound": missing,
               "generation_changes": generations, "counter_rollbacks": rollbacks, "duplicate_observations": duplicates,
               "timing_fault_records": timing_faults, "observed_gap_candidates": gaps,
               "timing_eligible_gaps": eligible, "timing_eligibility_fraction": eligible/gaps if gaps else None,
               "cadence_error_per_publication_ns": describe(intervals), "segment_period_drift_ppm": drift,
               "receipt_cadence_error_per_publication_ns": describe(receipt_intervals),
               "sample_age_ns": describe([r["sample_age_ns"] for r in result if r["timing_valid"]]),
               "write_delay_ns": describe([r["write_delay_ns"] for r in result if r["timing_valid"]]),
               "publication_to_receipt_ns": describe([r["publication_to_receipt_ns"] for r in result if r["timing_valid"]]),
               "qualification": "timing prerequisites only; fit maturity, gap spacing and conditioning not reconstructed"}
    return summary, result


def query_analysis(rows, horizon=1_000_000, max_age=2_000_000, max_prediction=50_000_000):
    """Check logged readiness against actual query/sample/target times and lifetime diagnostics."""
    if not 0 <= horizon <= max_prediction or max_age <= 0:
        raise ValueError("invalid prediction bounds")
    ready = [row for row in rows if row["ready"]]
    bad = [row for row in ready if not (0 < row["sample_ns"] <= row["query_ns"] <= row["target_ns"] <= 2**63-1 and
           row["target_ns"] == row["query_ns"]+horizon and
           row["query_ns"]-row["sample_ns"] <= max_age and
           row["target_ns"]-row["sample_ns"] <= max_prediction)]
    counters = {}
    for key in ("processed", "dropped", "resets", "admitted_gaps", "rejected_gaps"):
        if all(key in row for row in rows):
            values = [row[key] for row in rows]
            if any(b < a for a, b in zip(values, values[1:])):
                raise ValueError(f"decreasing lifetime {key}; split controller process runs")
            counters[key] = {"first": values[0], "last": values[-1], "delta": values[-1]-values[0]}
    return {"queries": len(rows), "ready": len(ready), "availability_pct": 100*len(ready)/len(rows),
            "invalid_ready_queries": len(bad), "sample_age_ns": describe([r["query_ns"]-r["sample_ns"] for r in ready]),
            "effective_horizon_ns": describe([r["target_ns"]-r["sample_ns"] for r in ready]),
            "service_ns": describe([r["service_ns"] for r in rows if "service_ns" in r]),
            "counters": counters}


def wfs_alignment(frames, wfs, max_interval=10_000_000):
    """Associate acquisitions with the latest valid observed WFS reference; this is not simulator wake timing."""
    references = [(raw["receipt_mono_ns"], audited["mapped_ns"]) for raw, audited in wfs
                  if audited["timing_valid"] and raw["accepted"]]
    receipts = [r[0] for r in references]
    acquired, observed = [], []
    for frame in frames:
        if not frame["timing_valid"]:
            continue
        index = bisect.bisect_right(receipts, frame["mapped_ns"])-1
        if index >= 0:
            receipt, sample = references[index]
            if 0 <= frame["mapped_ns"]-receipt <= max_interval:
                observed.append(frame["mapped_ns"]-receipt)
                acquired.append(frame["mapped_ns"]-sample)
    return {"acc_since_observed_wfs_receipt_ns": describe(observed),
            "acc_since_that_wfs_acquisition_ns": describe(acquired),
            "qualification": "observer-relative alignment; actual accelSim reference arrival/deadline must be logged separately"}


def analyze(frames_path, output, rate, queries=None, wfs=None, plots=False,
            max_age=2_000_000, max_interval=2_000_000, clock_tolerance=1_000_000,
            horizon=1_000_000, max_prediction=50_000_000, wfs_rate=2000):
    """Write a reproducible offline report; input files must come from one host boot and known realtime clocks."""
    rows = read_csv(frames_path, FRAME_FIELDS)
    report, audited = frame_analysis(rows, rate, max_age, max_interval, clock_tolerance)
    if queries:
        report["prediction"] = query_analysis(read_csv(queries, QUERY_FIELDS), horizon, max_age, max_prediction)
    if wfs:
        wfs_rows = read_csv(wfs, FRAME_FIELDS)
        wfs_report, wfs_audited = frame_analysis(wfs_rows, wfs_rate, max(10_000_000, max_age), max_interval, clock_tolerance)
        report["wfs_timing"] = {key: wfs_report[key] for key in
            ("records", "observed_missing_lower_bound", "generation_changes", "counter_rollbacks",
             "timing_fault_records", "cadence_error_per_publication_ns", "segment_period_drift_ppm")}
        report["wfs_alignment"] = wfs_alignment(audited, list(zip(wfs_rows, wfs_audited)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/"frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audited[0]))
        writer.writeheader()
        writer.writerows(audited)
    with (output/"summary.json").open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        time = [(r["receipt_mono_ns"]-rows[0]["receipt_mono_ns"])*1e-9 for r in rows]
        figure, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
        axes[0].plot(time, [r["sample_age_ns"]*1e-3 for r in audited], linewidth=.5)
        axes[0].set_ylabel("Sample age [us]")
        axes[1].plot(time, [(r["clock_offset_ns"]-audited[0]["clock_offset_ns"])*1e-3 for r in audited])
        axes[1].set_ylabel("Clock offset change [us]")
        gaps = [(t, r) for t, r in zip(time, audited) if r["gap_candidate"]]
        axes[2].scatter([t for t, _ in gaps], [r["gap_error_ns"]*1e-3 for _, r in gaps], s=8)
        for sign in (-1, 1):
            axes[2].axhline(sign*.05*1e6/rate, color="red", linewidth=.6)
        axes[2].set_ylabel("Gap cadence error [us]")
        axes[2].set_xlabel("Receipt time since recording start [s]")
        figure.tight_layout()
        figure.savefig(output/"timing.png")
        plt.close(figure)
    return report


def main():
    """Analyze existing local CSV recordings only; never attach shared-memory streams."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=float, required=True)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--wfs", type=Path)
    parser.add_argument("--wfs-rate", type=float, default=2000)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--max-age-ns", type=int, default=2_000_000)
    parser.add_argument("--max-interval-ns", type=int, default=2_000_000)
    parser.add_argument("--clock-tolerance-ns", type=int, default=1_000_000)
    parser.add_argument("--horizon-ns", type=int, default=1_000_000)
    parser.add_argument("--max-prediction-ns", type=int, default=50_000_000)
    args = parser.parse_args()
    report = analyze(args.frames, args.output, args.sample_rate, args.queries, args.wfs, args.plots,
                     args.max_age_ns, args.max_interval_ns, args.clock_tolerance_ns,
                     args.horizon_ns, args.max_prediction_ns, args.wfs_rate)
    print(json.dumps(report, indent=2, allow_nan=False))
    if report.get("prediction", {}).get("invalid_ready_queries", 0):
        raise SystemExit("Recorded ready predictions violate timing bounds; see summary.json")


if __name__ == "__main__":
    main()
