r"""\file verify_native_observation.py
\brief Inspect private completed observation artifacts; never attach a native stream or qualify training.
"""
import argparse
import csv
import json
from pathlib import Path

from inspect_native_commands import inspect


def verify(directory, minimum_seconds=59.5, profile="full"):
    """Verify exact records without changing files; full preserves the original four-source contract."""
    if profile not in ("full", "adc-wfs", "live"):
        raise ValueError("Unknown observation profile")
    required = {"adc", "modal", "camera"} | ({"dm"} if profile != "adc-wfs" else set())
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
    first_receipt, last_receipt = [None]*6, [None]*6
    journal_gaps = [0]*6
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
            receipt = int(record["available_mono_ns"])
            if first_receipt[slot] is None:
                first_receipt[slot] = receipt
            last_receipt[slot] = receipt
            identity = tuple(int(record[key]) for key in ("device", "file_inode", "generation"))
            counter = int(record["counter"])
            if identities[slot] == identity and last[slot] and counter <= last[slot][0]:
                failures.append(f"Nonincreasing journal counter in {sources[slot]['role']}")
            if identities[slot] == identity and last[slot] and counter > last[slot][0]+1:
                journal_gaps[slot] += counter-last[slot][0]-1
            stamp = int(record["acquired_raw_ns"])
            if first[slot] is None:
                first[slot] = (counter, stamp)
            last[slot], identities[slot] = (counter, stamp), identity
            if int(record["received_mono_ns"]) > int(record["available_mono_ns"]):
                failures.append("Receipt later than observer availability")
    if rows != summary["written"]:
        failures.append(f"Journal rows {rows} != written {summary['written']}")
    commands = None
    if profile == "live" and summary.get("dm_commands_enabled") is not True:
        failures.append("Full DM command recording must be enabled for the live-input profile")
    if summary.get("dm_commands_enabled"):
        for key in ("dm_command_drops", "dm_command_oversize"):
            if summary.get(key) != 0:
                failures.append(f"{key}={summary.get(key)}")
        commands = inspect(directory / "dm_commands.bin", values=0)
        if commands["records"] != counts[3] or commands["records"] != summary.get("dm_commands_written"):
            failures.append("Full DM command/metadata record counts differ")
        if commands["nonfinite_values"]:
            failures.append("Nonfinite full DM command values")
    for i, source in enumerate(sources):
        role = source["role"]
        if counts[i] != source["samples"]:
            failures.append(f"{role}: sample/journal mismatch")
        if role in required and counts[i] < 2:
            failures.append(f"{role}: insufficient observations")
        if profile in ("adc-wfs", "live") and role in ("modal", "camera") and counts[i] >= 2:
            span = last_receipt[i]-first_receipt[i]
            if span/1e9 < max(0, minimum_seconds-.5):
                failures.append(f"{role}: insufficient observer-time coverage")
            rate = (counts[i]-1)*1e9/span if span > 0 else 0
            if not 1900 <= rate <= 2100:
                failures.append(f"{role}: observed cadence outside 2 kHz +/-5%: {rate:.3f} Hz")
        for key in ("invalid", "clock_mismatches", "incompatible", "protocol_invalid", "restarts", "protocol_restarts"):
            if source[key]:
                target = failures if profile != "adc-wfs" or role in required else limitations
                target.append(f"{role}.{key}={source[key]}")
        if profile in ("adc-wfs", "live") and role in required and source["counter_gaps"]:
            failures.append(f"{role}.counter_gaps={source['counter_gaps']}")
        if profile in ("adc-wfs", "live") and role in required and journal_gaps[i]:
            failures.append(f"{role}.journal_counter_gaps={journal_gaps[i]}")
        for key in ("counter_gaps", "sequence_gaps", "protocol_incomplete", "association_failures", "clock_unknown"):
            if source[key]:
                limitations.append(f"{role}.{key}={source[key]}")
        if i >= 4 and not counts[i]:
            limitations.append(f"{role}: software endpoint unavailable/unobserved")
        if i < 4 and first[i] and last[i] and last[i][1] > first[i][1]:
            rates[role] = (counts[i]-1)*1e9 / (last[i][1]-first[i][1])
        if profile == "live" and role == "dm" and counts[i] >= 2:
            if (last_receipt[i]-first_receipt[i])/1e9 < max(0, minimum_seconds-.5):
                failures.append("dm: insufficient observer-time coverage")
    adc_rate = rates.get("adc", 0)
    if not 7600 <= adc_rate <= 8400:
        failures.append(f"ADC accepted rate outside 8 kHz +/-5%: {adc_rate:.3f} Hz")
    if first[0] and last[0] and (last[0][1]-first[0][1])/1e9 < max(0, minimum_seconds-.5):
        failures.append("ADC timestamp coverage shorter than requested observation interval")
    limitations.append("Exposure boundaries, modal-camera origin and physical DM application remain unqualified")
    return {"validation_profile": profile, "required_sources": sorted(required),
            "journal_counter_gaps": {s["role"]: journal_gaps[i] for i, s in enumerate(sources)},
            "dm_transport": "deferred" if profile == "adc-wfs" else "included",
            "exposure_qualified": False, "physical_application_qualified": False,
            "full_dm_commands": commands,
            "transport_pass": not failures, "duration_sec": duration, "observed_rates_hz": rates,
            "failures": failures, "limitations": limitations, "training_allowed": False}


def main():
    """Read one completed private run and return a failing exit status for transport/recording defects."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--profile", choices=("full", "adc-wfs", "live"), default="full")
    args = parser.parse_args()
    try:
        result = verify(args.directory, profile=args.profile)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Incomplete/invalid observation artifacts: {error}") from error
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["transport_pass"] else 1)


if __name__ == "__main__":
    main()
