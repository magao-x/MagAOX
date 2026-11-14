r"""\file native_timing_audit.py
\brief Read-only native journal clock, four-to-one phase and causal-availability analysis; no runtime transport.
\ingroup loPredCtrlAcc_unit_test
"""
import argparse
from array import array
from bisect import bisect_left, bisect_right
from collections import deque
import csv
import json
import math
from pathlib import Path
import statistics

from verify_native_observation import verify

ROLES = ("adc", "modal", "camera", "dm", "completion", "sdk")
CLOCKS = ("unknown", "realtime", "tai", "monotonic")


def distribution(values):
    """Summarize relative nanoseconds in microseconds; empty populations stay explicitly unavailable."""
    values = sorted(values)
    if not values:
        return {"count": 0}
    mean = math.fsum(values)/len(values)
    absolute = sorted(abs(v) for v in values)
    return {"count": len(values), "min_us": values[0]/1000, "median_us": statistics.median(values)/1000,
            "p99_us": values[math.ceil(.99*len(values))-1]/1000, "max_us": values[-1]/1000,
            "p99_abs_us": absolute[math.ceil(.99*len(values))-1]/1000,
            "stddev_us": math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values))/1000,
            "rms_us": math.sqrt(math.fsum(v*v for v in values)/len(values))/1000}


def wrap(value, period):
    """Return a signed phase residual, without changing an original acquisition timestamp."""
    return (value + period/2) % period - period/2


class Series:
    """Compact integer columns for one source; identity indexes never cross source/segment incarnations."""

    def __init__(self):
        """Allocate empty owned arrays without opening streams or files."""
        self.counter, self.counter1 = array("Q"), array("Q")
        self.acquired, self.published = array("q"), array("q")
        self.received, self.available = array("q"), array("q")
        self.epoch, self.has_counter1 = array("I"), array("B")
        self.identities = {}

    def append(self, record):
        """Retain only integer timing/identity fields from one sample row."""
        identity = tuple(int(record[key]) for key in ("device", "file_inode", "image_inode", "generation", "segment"))
        epoch = self.identities.setdefault(identity, len(self.identities))
        for name, key in (("counter", "counter"), ("counter1", "counter1"), ("acquired", "acquired_raw_ns"),
                          ("published", "published_raw_ns"), ("received", "received_mono_ns"),
                          ("available", "available_mono_ns"), ("has_counter1", "has_counter1")):
            getattr(self, name).append(int(record.get(key, 0)))
        self.epoch.append(epoch)


def read_journal(directory, summary):
    """Read ordinary artifacts once; retain decimated local clock beacons independently of source-clock claims."""
    series = {role: Series() for role in ROLES}
    beacons = {clock: [] for clock in CLOCKS[1:]}
    with (directory / "observations.csv").open(newline="") as file:
        for record in csv.DictReader(file):
            if int(record["event"]) != 0:
                continue
            slot = int(record["source"])
            if not 0 <= slot < len(ROLES):
                raise ValueError("Unknown source role")
            series[ROLES[slot]].append(record)
            domain = summary["sources"][slot]["publication_clock"]
            now, width, after = (int(record[key]) for key in
                                 ("clock_now_ns", "clock_bracket_ns", "available_mono_ns"))
            if domain in beacons and now > 0 and 0 <= width <= 1000000 and after >= width:
                entries = beacons[domain]
                if not entries or after-entries[-1][0] >= 1000000:
                    entries.append((after, now, width))
    return series, beacons


def clock_spec(manifest, summary, role, field):
    """Keep an observer declaration distinct from a same-host clock attested with source evidence."""
    declared = summary["sources"][ROLES.index(role)][field + "_clock"]
    spec = dict(manifest.get("sources", {}).get(role, {}).get(field, {}))
    spec.setdefault("clock", declared)
    spec.setdefault("verified", False)
    spec.setdefault("same_host", False)
    spec.setdefault("evidence", "")
    spec.setdefault("event", "unknown")
    if spec["clock"] not in CLOCKS or type(spec["verified"]) is not bool or type(spec["same_host"]) is not bool:
        raise ValueError("Invalid source clock contract")
    if spec["verified"] and (spec["clock"] == "unknown" or not spec["same_host"] or
                             not spec["evidence"].strip() or spec["event"] == "unknown" or not spec["event"].strip()):
        raise ValueError("Verified clocks require a known same-host domain, event and source evidence")
    return spec


class ClockMap:
    """Bracketed local mapping with explicit age/drift allowance and discontinuity segmentation."""

    def __init__(self, beacons, uncertainty_ns=25000, max_anchor_age_ns=2000000, drift_ppm=1000):
        """Prepare immutable local beacons; a source's clock identity is established separately."""
        self.beacons, self.epochs, self.times = {}, {}, {}
        self.uncertainty_ns, self.max_anchor_age_ns, self.drift_ppm = uncertainty_ns, max_anchor_age_ns, drift_ppm
        self.steps = []
        for domain, entries in beacons.items():
            entries = sorted(entries)
            epochs, epoch, previous = [], 0, None
            for after, clock, width in entries:
                low, high = after-width-clock, after-clock
                if previous is not None:
                    old_after, old_low, old_high = previous
                    distance = max(0, low-old_high, old_low-high)
                    allowance = math.ceil((after-old_after)*drift_ppm/1e6)
                    if distance > allowance:
                        epoch += 1
                        self.steps.append({"clock": domain, "after_mono_ns": after,
                                           "offset_change_us": ((low+high)-(old_low+old_high))/2000})
                epochs.append(epoch)
                previous = after, low, high
            self.beacons[domain], self.epochs[domain] = entries, epochs
            self.times[domain] = [entry[0] for entry in entries]

    def map(self, raw, available, domain):
        """Return midpoint, conservative mapping half-width and epoch, or None; never bridge a detected step."""
        if raw <= 0 or available < 0:
            return None
        if domain == "monotonic":
            return raw, 0, 0
        entries = self.beacons.get(domain, [])
        if not entries:
            return None
        position = bisect_left(self.times[domain], available)
        candidates = [i for i in (position-1, position) if 0 <= i < len(entries)]
        index = min(candidates, key=lambda i: abs(entries[i][0]-available))
        after, clock, width = entries[index]
        if abs(after-available) > self.max_anchor_age_ns:
            return None
        bounds = []
        for candidate in candidates:
            endpoint, clock, width = entries[candidate]
            if abs(endpoint-available) <= self.max_anchor_age_ns and self.epochs[domain][candidate] == self.epochs[domain][index]:
                extra = math.ceil(abs(raw-clock)*self.drift_ppm/1e6)
                bounds.append((raw+endpoint-width-clock-extra, raw+endpoint-clock+extra))
        low, high = min(v[0] for v in bounds), max(v[1] for v in bounds)
        uncertainty = (high-low+1)//2
        if uncertainty > self.uncertainty_ns:
            return None
        # Samples acquired before an intervening clock step cannot inherit a post-step offset.
        first = max(0, bisect_right(self.times[domain], min(low, available))-1)
        last = min(len(entries)-1, bisect_left(self.times[domain], max(high, available)))
        if self.epochs[domain][first] != self.epochs[domain][last]:
            return None
        return low+(high-low)//2, uncertainty, self.epochs[domain][index]


def mapped_series(series, spec, mapper, field):
    """Map a declared timestamp hypothesis; qualification is not implied by successful arithmetic."""
    times, uncertainty, epochs = array("q"), array("q"), array("q")
    for raw, available, incarnation in zip(getattr(series, field), series.available, series.epoch):
        result = mapper.map(raw, available, spec["clock"]) if spec["same_host"] else None
        time, error, epoch = result if result is not None else (-1, -1, -1)
        times.append(time)
        uncertainty.append(error)
        epochs.append(incarnation*1000000+epoch if epoch >= 0 else -1)
    return times, uncertainty, epochs


def cadence(series):
    """Report original counter gaps and raw acquisition cadence without claiming a timestamp clock or exposure event."""
    periods, jumps = [], 0
    for i in range(1, len(series.counter)):
        if series.epoch[i] != series.epoch[i-1]:
            continue
        delta = series.counter[i]-series.counter[i-1]
        elapsed = series.acquired[i]-series.acquired[i-1]
        if delta > 0 and elapsed > 0:
            periods.append(elapsed/delta)
        else:
            jumps += 1
    median = statistics.median(periods) if periods else None
    return {"samples": len(series.counter), "incarnations": len(series.identities),
            "raw_period": distribution(periods), "raw_rate_hz": 1e9/median if median else None,
            "nonincreasing_intervals": jumps,
            "observer_time_regressions": sum(b < a for a, b in zip(series.available, series.available[1:]))}


def associations(modal, camera, contract):
    """Count exact candidate joins; only an evidenced pixel/origin contract can qualify them."""
    method = contract.get("method", "none")
    if method not in ("none", "counter1", "acquisition_timestamp"):
        raise ValueError("Unsupported modal-camera association method")
    verified = contract.get("verified", False)
    if type(verified) is not bool or (verified and (method == "none" or not contract.get("evidence", "").strip())):
        raise ValueError("A verified association requires an explicit evidenced producer contract")
    result = {"method": method, "qualified": False, "candidates": {}}
    for candidate in ("counter1", "acquisition_timestamp"):
        keys = camera.counter if candidate == "counter1" else camera.acquired
        lookup = {}
        for i, key in enumerate(keys):
            if key > 0:
                lookup[key] = i if key not in lookup else -1
        matched, missing, ambiguous, reused, used = 0, 0, 0, 0, set()
        values = modal.counter1 if candidate == "counter1" else modal.acquired
        for i, key in enumerate(values):
            key_available = candidate != "counter1" or modal.has_counter1[i]
            found = lookup.get(key) if key_available and key > 0 else None
            if found is None:
                missing += 1
            elif found == -1:
                ambiguous += 1
            elif found in used:
                reused += 1
            else:
                used.add(found)
                matched += 1
        result["candidates"][candidate] = {"matched": matched, "missing": missing,
                                             "ambiguous": ambiguous, "reused": reused}
    if method != "none":
        v = result["candidates"][method]
        result["qualified"] = bool(verified and v["matched"] and not (v["missing"] or v["ambiguous"] or v["reused"]))
    return result


def phase_population(pairs, period):
    """Use counter-anchored unwrapped residuals so full-cycle slips remain visible; fit relative time."""
    if not pairs:
        return {"pairs": 0}
    unwrapped, times, previous, group, anchor = [], [], None, None, None
    counter_errors, jumps, events, adjustments, changes = 0, 0, [], 0, []
    for t, residual, counter, frame, incarnation, error in pairs:
        if incarnation != group:
            previous, anchor, group = None, t, incarnation
        phase = residual
        if previous is not None:
            counter_errors += counter-previous[1] != 4*(frame-previous[2])
            changes.append(phase-previous[0])
            adjustments += abs(phase-previous[0]) > max(period/80, 2*(error+previous[3]))
            changed = abs(phase-previous[0]) > max(period/8, 2*(error+previous[3]))
            jumps += changed
            if changed and len(events) < 32:
                events.append({"camera_publication_mono_ns": t, "adc_counter": counter,
                               "camera_counter": frame, "phase_change_us": (phase-previous[0])/1000})
        unwrapped.append(phase)
        times.append((t-anchor)/1e9)
        previous = phase, counter, frame, error
    # Fit each population only when it contains a single incarnation; never hide resets in a global drift fit.
    groups = {p[4] for p in pairs}
    drift = None
    detrended = []
    if len(groups) == 1 and len(times) > 1:
        mean_x, mean_y = statistics.mean(times), statistics.mean(unwrapped)
        square = sum((x-mean_x)**2 for x in times)
        slope = sum((x-mean_x)*(y-mean_y) for x, y in zip(times, unwrapped))/square if square else 0
        drift = slope/1000  # nanoseconds per second -> ppm
        detrended = [y-(mean_y+slope*(x-mean_x)) for x, y in zip(times, unwrapped)]
    center = statistics.median(unwrapped)
    return {"pairs": len(pairs), "segments": len(groups), "offset_us": wrap(center, period)/1000,
            "counter_unwrapped_offset_us": center/1000,
            "phase_error": distribution([wrap(y-center, period) for y in unwrapped]),
            "detrended_jitter": distribution(detrended), "drift_ppm": drift,
            "counter_stride_failures": counter_errors, "candidate_phase_discontinuities": jumps,
            "candidate_phase_adjustments_or_jitter": adjustments, "consecutive_phase_change": distribution(changes),
            "candidate_discontinuity_examples": events}


def phase_audit(adc, camera, adc_map, camera_map, expected=None, residue=None):
    """Evaluate all four publication-counter residues, not array positions; pairing is offline characterization only."""
    adc_time, adc_error, adc_epoch = adc_map
    camera_time, camera_error, camera_epoch = camera_map
    intervals = [camera_time[i]-camera_time[i-1] for i in range(1, len(camera_time))
                 if camera_time[i-1] >= 0 and camera_time[i] > camera_time[i-1] and camera_epoch[i] == camera_epoch[i-1]]
    if not intervals:
        return {"available": False, "reason": "camera publication clock unmapped"}
    period = statistics.median(intervals)
    groups = [[] for _ in range(4)]
    for i, stamp in enumerate(adc_time):
        if stamp >= 0:
            groups[adc.counter[i] % 4].append(i)
    results, calibration = [], []
    for indexes in groups:
        indexes.sort(key=lambda i: adc_time[i])
        stamps = [adc_time[i] for i in indexes]
        lookup = {(adc_epoch[i], adc.counter[i]): i for i in indexes}
        pairs, anchors = [], {}
        for k, stamp in enumerate(camera_time):
            if stamp < 0 or not stamps:
                continue
            target = stamp + (expected or 0)
            position = bisect_left(stamps, target)
            candidates = [indexes[j] for j in (position-1, position) if 0 <= j < len(indexes)]
            i = min(candidates, key=lambda j: abs(adc_time[j]-target))
            incarnation = adc_epoch[i], camera_epoch[k]
            if incarnation not in anchors:
                if abs(adc_time[i]-target) > period/2-adc_error[i]-camera_error[k]:
                    continue
                anchors[incarnation] = adc.counter[i], camera.counter[k]
            base_adc, base_camera = anchors[incarnation]
            i = lookup.get((incarnation[0], base_adc+4*(camera.counter[k]-base_camera)))
            if i is None:
                continue
            error = adc_error[i]+camera_error[k]
            pairs.append((stamp, adc_time[i]-stamp, adc.counter[i], camera.counter[k],
                          (adc_epoch[i], camera_epoch[k]), error))
        population = phase_population(pairs, period)
        population["unpaired_camera_frames"] = sum(t >= 0 for t in camera_time)-len(pairs)
        if expected is not None:
            population["absolute_expected_phase_error"] = distribution(
                [abs(p[1]-expected)+p[5] for p in pairs])
        results.append(population)
        initial = [p for p in pairs if p[0] < pairs[0][0]+1000000000] if pairs else []
        calibration.append(phase_population(initial, period))
    if residue is None:
        choices = [i for i, v in enumerate(calibration) if v["pairs"]]
        residue = min(choices, key=lambda i: abs(wrap(calibration[i]["offset_us"]*1000-(expected or 0), period))) if choices else None
    if residue is not None and (type(residue) is not int or not 0 <= residue <= 3):
        raise ValueError("ADC residue must be an integer in [0,3]")
    return {"available": residue is not None, "reference": "camera_publication_not_exposure",
            "period_us": period/1000, "residue_selection": "first-second diagnostic calibration unless supplied",
            "selected_residue": residue, "residues": results,
            "selected": results[residue] if residue is not None else None,
            "expected_phase_us": expected/1000 if expected is not None else None,
            "hardware_phase_lock_qualified": False,
            "note": "Nearest pairing may include later ADC acquisitions; it is not causal feature admission."}


def causal_audit(adc, modal, adc_map, publication_map, window_ns=500000, deadlines=None):
    """Count ADC records actually available by each observed modal deadline; no proposals or model calls."""
    acquired, error, epoch = adc_map
    published, pub_error, _ = publication_map
    at, last, no_input, stale, incomplete, mapped, future = 0, None, 0, 0, 0, 0, 0
    ages, window_counts, available_delays = [], [], []
    eligible, pending, examples = [], deque(), []
    deadlines = modal.available if deadlines is None else deadlines
    for deadline in deadlines:
        while at < len(adc.available) and adc.available[at] <= deadline:
            if acquired[at] >= 0 and published[at] >= 0 and adc.received[at] <= adc.available[at]:
                pending.append(at)
            at += 1
        while pending:
            i = pending[0]
            if acquired[i]+error[i] > deadline or published[i]+pub_error[i] > deadline:
                break
            pending.popleft()
            eligible.append(i)
            last = i
            mapped += 1
        if last is None:
            no_input += 1
            continue
        ages.append(deadline-acquired[last])
        future += (adc.available[last] > deadline or acquired[last]+error[last] > deadline or
                   published[last]+pub_error[last] > deadline)
        if len(examples) < 16:
            examples.append({"deadline_ns": deadline, "last_counter": adc.counter[last],
                             "adc_available_ns": adc.available[last],
                             "acquired_upper_ns": acquired[last]+error[last],
                             "published_upper_ns": published[last]+pub_error[last]})
        available_delays.append(adc.available[last]-acquired[last])
        stale += deadline-(acquired[last]-error[last]) > window_ns
        start = deadline-window_ns
        tail = len(eligible)-1
        count, anchor = 0, None
        while tail >= 0:
            i = eligible[tail]
            if epoch[i] != epoch[last]:
                break
            if acquired[i]+error[i] <= start:
                anchor = i
                break
            count += 1
            tail -= 1
        window_counts.append(count)
        # Coverage needs a known left anchor, chronological consecutive publications, and bounded holds.
        selected = eligible[tail:] if anchor is not None else []
        complete = bool(selected)
        for a, b in zip(selected, selected[1:]):
            complete = complete and adc.counter[b] == adc.counter[a]+1 and acquired[b]-acquired[a] <= 250000
        if complete:
            complete = deadline-(acquired[last]-error[last]) <= 250000
        incomplete += not complete
    return {"deadline_reference": "modal_observer_availability_not_exposure_or_synco_cutoff",
            "deadlines": len(deadlines), "mapped_delivered_adc": mapped,
            "no_eligible_adc": no_input, "older_than_one_wfs_period": stale,
            "incomplete_500us_diagnostic_windows": incomplete, "latest_adc_age": distribution(ages),
            "adc_frontend_delivery_delay": distribution(available_delays),
            "window_sample_counts": {str(n): window_counts.count(n) for n in sorted(set(window_counts))},
            "future_samples_admitted": future, "deadline_examples": examples, "training_allowed": False}


def synchronization_model(period_ns, contract):
    """Report the inspected delay equation's assumptions relative to semaphore receipt, never exposure."""
    terms = contract.get("latency_terms_ns", [3000, 51500, 10000, 276100])
    post = contract.get("post_delay_us", 0)
    if len(terms) != 4 or any(not math.isfinite(v) or v < 0 for v in terms) or not math.isfinite(post):
        raise ValueError("Invalid delay model assumptions")
    delay = (period_ns/2-sum(terms)+post*1000) % period_ns
    return {"reference": "synchronization_semaphore_receipt", "latency_terms_ns": terms,
            "post_delay_us": post, "modeled_delay_us": delay/1000,
            "adc_lattice_phase_us": (delay % (period_ns/4))/1000,
            "parameters": "manifest assumptions" if contract else "source defaults, not measured configuration",
            "period_basis": "mapped camera publication median or nominal 500us, not measured semaphore receipt period",
            "hardware_revision_note": "AD7689 0ae4dcb8 performs one scan per wake; continuous 4:1 hardware contract unverified"}


def synco_constraints(period_ns, contract):
    """Report nominal future scheduling assumptions without importing any scheduler/producer transport."""
    service = contract.get("dm_service_us", 110)
    lead = contract.get("ready_lead_us", 70)
    if not math.isfinite(service) or not math.isfinite(lead) or service <= 0 or lead <= 0:
        raise ValueError("Invalid future synco budget assumptions")
    interval = period_ns/4000
    return {"reference": "genuine_main_dm_command_publication, not camera timestamps",
            "protocol": "production version 2, same-host CLOCK_MONOTONIC",
            "steps_per_main": 4, "slot_spacing_us": interval, "assumed_dm_service_us": service,
            "nominal_slack_us": interval-service, "assumed_input_ready_lead_us": lead,
            "main_cycle_offsets_frames": [0, .25, .5, .75],
            "following_cycle_display_phases": [1, 1.25, 1.5, 1.75],
            "advertised_step0": "next genuine main target; steps 1..3 are current cycle quarters",
            "ready_cutoff_rule": "target - ready_lead; frozen epoch/generation/cycle/step identity",
            "acknowledgement": "transport snapshot/discard only, not physical application",
            "scheduling_qualified": False, "dm_output_allowed": False}


def audit(directory, manifest=None, max_mapping_uncertainty_us=25, minimum_seconds=59.5):
    """Analyze immutable artifacts; all learning/output permissions remain false irrespective of diagnostic success."""
    manifest = manifest or {}
    if not isinstance(manifest, dict):
        raise ValueError("Provenance must be a JSON object")
    if manifest and manifest.get("schema") != 1:
        raise ValueError("Unsupported provenance schema")
    if not math.isfinite(max_mapping_uncertainty_us) or not 0 < max_mapping_uncertainty_us <= 500:
        raise ValueError("Mapping uncertainty must be in (0,500] us")
    kind = manifest.get("acquisition_kind", "unknown")
    if kind not in ("unknown", "simulator", "synchronized_hardware"):
        raise ValueError("Unknown acquisition kind")
    transport = verify(directory, minimum_seconds=minimum_seconds, profile="adc-wfs")
    summary = json.loads((directory / "summary.json").read_text())
    series, beacons = read_journal(directory, summary)
    mapper = ClockMap(beacons, round(max_mapping_uncertainty_us*1000))
    specs = {role: {field: clock_spec(manifest, summary, role, field) for field in ("acquisition", "publication")}
             for role in ("adc", "camera", "modal")}
    adc = mapped_series(series["adc"], specs["adc"]["acquisition"], mapper, "acquired")
    adc_pub = mapped_series(series["adc"], specs["adc"]["publication"], mapper, "published")
    camera = mapped_series(series["camera"], specs["camera"]["publication"], mapper, "published")
    modal_pub = mapped_series(series["modal"], specs["modal"]["publication"], mapper, "published")
    sync = manifest.get("synchronization", {})
    expected = sync.get("expected_publication_phase_us")
    if expected is not None and (not math.isfinite(expected) or sync.get("expected_reference") != "camera_publication"):
        raise ValueError("Expected publication phase requires an explicit camera_publication reference")
    phase = phase_audit(series["adc"], series["camera"], adc, camera,
                        None if expected is None else expected*1000, sync.get("counter_residue"))
    phase["clock_provenance_verified"] = specs["adc"]["acquisition"]["verified"] and specs["camera"]["publication"]["verified"]
    receipt_phase = phase_audit(
        series["adc"], series["camera"],
        (series["adc"].available, [0]*len(series["adc"].available), series["adc"].epoch),
        (series["camera"].available, [0]*len(series["camera"].available), series["camera"].epoch))
    receipt_phase["reference"] = "observer_availability_only; polling and batching bias, not sensor synchronization"
    association = associations(series["modal"], series["camera"], manifest.get("association", {}))
    cadence_stats = {role: cadence(series[role]) for role in ("adc", "camera", "modal")}
    period = phase.get("period_us", 500)*1000
    blockers = []
    for role, field in (("adc", "acquisition"), ("adc", "publication"), ("camera", "publication"),
                        ("modal", "publication")):
        if not specs[role][field]["verified"]:
            blockers.append(f"{role}.{field} clock provenance unverified")
    if not transport["transport_pass"]:
        blockers.append("ADC/WFS transport failed")
    if mapper.steps:
        blockers.append("Clock offset discontinuities detected")
    if not phase["available"]:
        blockers.append("Acquisition/publication phase unavailable")
    if expected is None:
        blockers.append("Expected phase relative to camera publication unverified; semaphore-receipt phase is distinct")
    elif not sync.get("expected_phase_evidence", "").strip():
        blockers.append("Expected phase lacks source/configuration evidence")
    phase_tolerance = sync.get("phase_tolerance_us", 25)
    if not math.isfinite(phase_tolerance) or phase_tolerance <= 0:
        raise ValueError("Invalid phase tolerance")
    if (phase.get("selected") or {}).get("absolute_expected_phase_error", {}).get("max_us", 0) > phase_tolerance:
        blockers.append("Expected four-to-one phase tolerance exceeded")
    if any(v["nonincreasing_intervals"] or v["observer_time_regressions"] for v in cadence_stats.values()):
        blockers.append("Timestamp/counter order regression")
    if any(v["incarnations"] != 1 for v in cadence_stats.values()):
        blockers.append("Source incarnation/segment changes require separate epoch review")
    if sum(t >= 0 for t in adc[0]) != len(series["adc"].counter) or sum(t >= 0 for t in adc_pub[0]) != len(series["adc"].counter):
        blockers.append("ADC clock mapping coverage incomplete")
    if sum(t >= 0 for t in camera[0]) != len(series["camera"].counter):
        blockers.append("Camera publication mapping coverage incomplete")
    if sum(t >= 0 for t in modal_pub[0]) != len(series["modal"].counter):
        blockers.append("Modal publication mapping coverage incomplete")
    if not association["qualified"]:
        blockers.append("Modal-camera origin unqualified")
    if phase.get("selected") and (phase["selected"]["counter_stride_failures"] or
                                   phase["selected"]["candidate_phase_discontinuities"] or
                                   phase["selected"]["unpaired_camera_frames"]):
        blockers.append("Four-to-one phase/counter discontinuities need review")
    causal = causal_audit(series["adc"], series["modal"], adc, adc_pub)
    causal["all_adc_frontend_delivery_delay"] = distribution(
        [available-stamp for available, stamp in zip(series["adc"].available, adc[0]) if stamp >= 0])
    if causal["no_eligible_adc"] or causal["older_than_one_wfs_period"] or causal["future_samples_admitted"]:
        blockers.append("Modal deadlines lack fresh causal ADC input")
    if causal["incomplete_500us_diagnostic_windows"]:
        blockers.append("Causal diagnostic window coverage incomplete")
    adc_order_errors = sum(a >= 0 and p >= 0 and a-e > p+pe
                           for a, e, p, pe in zip(adc[0], adc[1], adc_pub[0], adc_pub[1]))
    if adc_order_errors:
        blockers.append("ADC acquisition/publication order inconsistent")
    future_publications = sum(t >= 0 and t-error > available for data, mapping in
                              ((series["adc"], adc_pub), (series["camera"], camera), (series["modal"], modal_pub))
                              for t, error, available in zip(mapping[0], mapping[1], data.available))
    if future_publications:
        blockers.append("Source publication occurs after observer availability")
    publication_cutoffs = array("q", (t-e for t, e in zip(modal_pub[0], modal_pub[1]) if t >= 0))
    publication_causal = causal_audit(series["adc"], series["modal"], adc, adc_pub, deadlines=publication_cutoffs)
    publication_causal["available"] = bool(publication_cutoffs)
    publication_causal["deadline_reference"] = "modal_publication_lower_bound; conservative marker, actual processing deadline unknown"
    return {"schema": 1, "observation_only": True, "acquisition_kind": kind,
            "transport": transport, "clock_specs": specs, "clock_steps": mapper.steps,
            "mapping_uncertainty_limit_us": max_mapping_uncertainty_us,
            "mapped_counts": {"adc_acquisition": sum(t >= 0 for t in adc[0]),
                              "camera_publication": sum(t >= 0 for t in camera[0]),
                              "modal_publication": sum(t >= 0 for t in modal_pub[0])},
            "adc_timestamp_order_errors": adc_order_errors,
            "future_publication_errors": future_publications,
            "cadence": cadence_stats, "four_to_one_phase": phase, "modal_camera_association": association,
            "observer_receipt_phase": receipt_phase,
            "synchronization_model": synchronization_model(period, sync),
            "causal_availability": causal,
            "causal_publication_cutoff": publication_causal,
            "future_synco_constraints": synco_constraints(period, manifest.get("synco", {})),
            "timing_review_blockers": blockers, "offline_prediction_timing_ready": not blockers,
            "recorded_ddspc_inputs_qualified": False, "training_allowed": False, "dm_output_allowed": False,
            "exposure_qualified": False, "physical_application_qualified": False,
            "limitations": ["Observer availability includes polling/batching latency.",
                            "Observed discontinuities are resynchronization candidates, not confirmed actions.",
                            "Simulator phase does not qualify synchronized hardware or physical optical coupling.",
                            "DDSPC replay still needs qualified WFS timing and independently confirmed applied inputs."]}


def main():
    """Print a JSON analysis of local files; never connect to ImageStreamIO, INDI, hardware or services."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--max-mapping-uncertainty-us", type=float, default=25)
    parser.add_argument("--minimum-seconds", type=float, default=59.5)
    parser.add_argument("--require-timing-ready", action="store_true",
                        help="Exit 2 for unresolved timing provenance, distinct from transport failure")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.provenance.read_text()) if args.provenance else {}
        result = audit(args.directory, manifest, args.max_mapping_uncertainty_us, args.minimum_seconds)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        raise SystemExit(f"Invalid/incomplete timing evidence: {error}") from error
    print(json.dumps(result, indent=2, allow_nan=False))
    # Unqualified provenance is a reportable next-stage blocker, not an acquisition-test failure.
    if not result["transport"]["transport_pass"]:
        raise SystemExit(1)
    raise SystemExit(2 if args.require_timing_ready and not result["offline_prediction_timing_ready"] else 0)


if __name__ == "__main__":
    main()
