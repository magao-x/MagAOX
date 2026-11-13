r"""\file verify_native_observation.py
\brief Inspect private completed observation artifacts; never attach a native stream or qualify training.
"""
import argparse
import csv
import json
from pathlib import Path


def verify(directory, minimum_seconds=59.5):
    """Return transport failures and provenance limitations from exact integer records, without changing files."""
    summary = json.loads((directory / "summary.json").read_text())
    failures, limitations, rates = [], [], {}
    if summary.get("schema") != 1 or summary.get("observation_only") is not True:
        failures.append("Unsupported/non-observation artifact")
    for key in ("training_enabled", "exposure_qualified", "physical_application_qualified", "io_error"):
        if summary.get(key) is not False:
            failures.append(f"{key} must be explicitly false")
    duration = (summary["stopped_ns"] - summary["started_ns"]) / 1e9
    if duration < minimum_seconds:
        failures.append(f"Recording too short: {duration:.6f} s")
    for key in ("queue_drops", "adc_missing"):
        if summary[key]:
            failures.append(f"{key}={summary[key]}")
    for key in ("dropped", "invalid", "timing_gaps", "restarts"):
        if summary["adc_capture"][key]:
            failures.append(f"adc_capture.{key}={summary['adc_capture'][key]}")
    sources = summary["sources"]
    if len(sources) != 6 or [s["role"] for s in sources] != ["adc", "modal", "camera", "dm", "completion", "sdk"]:
        raise ValueError("Unexpected source-role schema")
    counts, first, last, identities = [0]*6, [None]*6, [None]*6, [None]*6
    rows = 0
    with (directory / "observations.csv").open(newline="") as file:
        for record in csv.DictReader(file):
            rows += 1
            slot, event = int(record["source"]), int(record["event"])
            if not 0 <= slot < 6:
                raise ValueError("Unknown source role")
            if event != 0:
                continue
            counts[slot] += 1
            identity = tuple(int(record[key]) for key in ("device", "file_inode", "generation"))
            counter = int(record["counter"])
            if identities[slot] == identity and last[slot] and counter <= last[slot][0]:
                failures.append(f"Nonincreasing journal counter in {sources[slot]['role']}")
            stamp = int(record["acquired_raw_ns"])
            if first[slot] is None:
                first[slot] = (counter, stamp)
            last[slot], identities[slot] = (counter, stamp), identity
            if int(record["received_mono_ns"]) > int(record["available_mono_ns"]):
                failures.append("Receipt later than observer availability")
    if rows != summary["written"]:
        failures.append(f"Journal rows {rows} != written {summary['written']}")
    for i, source in enumerate(sources):
        role = source["role"]
        if counts[i] != source["samples"]:
            failures.append(f"{role}: sample/journal mismatch")
        if i < 4 and counts[i] < 2:
            failures.append(f"{role}: insufficient observations")
        for key in ("invalid", "clock_mismatches", "incompatible", "protocol_invalid", "restarts", "protocol_restarts"):
            if source[key]:
                failures.append(f"{role}.{key}={source[key]}")
        for key in ("counter_gaps", "sequence_gaps", "protocol_incomplete", "association_failures", "clock_unknown"):
            if source[key]:
                limitations.append(f"{role}.{key}={source[key]}")
        if i >= 4 and not counts[i]:
            limitations.append(f"{role}: software endpoint unavailable/unobserved")
        if i < 4 and first[i] and last[i] and last[i][1] > first[i][1]:
            rates[role] = (counts[i]-1)*1e9 / (last[i][1]-first[i][1])
    adc_rate = rates.get("adc", 0)
    if not 7600 <= adc_rate <= 8400:
        failures.append(f"ADC accepted rate outside 8 kHz +/-5%: {adc_rate:.3f} Hz")
    if first[0] and last[0] and (last[0][1]-first[0][1])/1e9 < max(0, minimum_seconds-.5):
        failures.append("ADC timestamp coverage shorter than requested observation interval")
    limitations.append("Exposure boundaries, modal-camera origin and physical DM application remain unqualified")
    return {"transport_pass": not failures, "duration_sec": duration, "observed_rates_hz": rates,
            "failures": failures, "limitations": limitations, "training_allowed": False}


def main():
    """Read one completed private run and return a failing exit status for transport/recording defects."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.directory)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Incomplete/invalid observation artifacts: {error}") from error
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["transport_pass"] else 1)


if __name__ == "__main__":
    main()
