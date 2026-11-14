r"""\file native_timing_audit_test.py
\brief Deterministic ordinary-file tests of four-to-one phase, clock provenance and causal deadlines.
\ingroup loPredCtrlAcc_unit_test
"""
from copy import deepcopy
import ast
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from native_timing_audit import (ClockMap, Series, associations, audit, causal_audit, phase_audit,
                                 mapped_series, synchronization_model, synco_constraints)
from verify_native_observation import verify

REAL = 1700000000000000000
MONO = 5000000000
FIELDS = ("source", "event", "device", "file_inode", "image_inode", "generation", "segment", "counter",
          "counter1", "has_counter1", "acquired_raw_ns", "published_raw_ns", "received_mono_ns",
          "available_mono_ns", "clock_now_ns", "clock_bracket_ns")


def record(role, counter, acquired, published, available, counter1=0):
    """Construct synthetic source events on known clocks; this fixture is never native telemetry."""
    return dict(zip(FIELDS, (role, 0, 1, 10+role, 100+role, 1, 1, counter, counter1, int(role != 0),
                             REAL+acquired, REAL+published, MONO+available-100,
                             MONO+available, REAL+available-50, 100)))


def fixture(directory, frames=40):
    """Write an 8/2-kHz synthetic journal with a fixed 30-us phase and an inactive DM."""
    records = [record(0, n+1, n*125000+30000, n*125000+40000, n*125000+50000)
               for n in range(frames*4+16)]
    for n in range(frames):
        stamp = 1000000+n*500000
        records.append(record(2, n+1, stamp-100000, stamp, stamp+20000))
        records.append(record(1, n+1, stamp-100000, stamp+50000, stamp+70000, n+1))
    records.append(record(3, 1, 1000, 2000, 3000))
    write_fixture(directory, records)
    clocks = {role: {field: {"clock": "realtime", "same_host": True, "verified": True, "event": "synthetic_"+field,
                              "evidence": "SYNTHETIC FIXTURE contract"}
                     for field in ("acquisition", "publication")} for role in ("adc", "camera", "modal")}
    return records, {"schema": 1, "acquisition_kind": "simulator", "sources": clocks,
                     "association": {"method": "counter1", "verified": True, "evidence": "SYNTHETIC frozen origin"},
                     "synchronization": {"expected_reference": "camera_publication",
                                         "expected_publication_phase_us": 30, "counter_residue": 1,
                                         "expected_phase_evidence": "SYNTHETIC generation contract"}}


def write_fixture(directory, records):
    """Serialize only test-owned ordinary artifacts, including independently counted source summaries."""
    records = sorted(records, key=lambda r: r["available_mono_ns"])
    sources = []
    for slot, role in enumerate(("adc", "modal", "camera", "dm", "completion", "sdk")):
        s = dict.fromkeys(("counter_gaps", "restarts", "protocol_restarts", "clock_unknown", "clock_mismatches",
                           "invalid", "protocol_invalid", "protocol_incomplete", "sequence_gaps",
                           "association_failures", "metadata_associations", "busy", "races", "incompatible"), 0)
        s.update(role=role, samples=sum(r["source"] == slot for r in records),
                 acquisition_clock="realtime", publication_clock="realtime")
        sources.append(s)
    summary = {"schema": 1, "observation_only": True, "training_enabled": False, "exposure_qualified": False,
               "physical_application_qualified": False, "io_error": False, "started_ns": MONO,
               "stopped_ns": records[-1]["available_mono_ns"], "queue_drops": 0, "written": len(records),
               "adc_missing": 0, "sources": sources,
               "adc_capture": dict(dropped=0, invalid=0, timing_gaps=0, restarts=0)}
    (directory / "summary.json").write_text(json.dumps(summary))
    with (directory / "observations.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def as_series(records):
    """Own fixed test rows without any ImageStreamIO or runtime dependency."""
    s = Series()
    for r in records:
        s.append(r)
    return s


class NativeTimingTest(unittest.TestCase):
    """Verify safety and the separation between transport, diagnostic phase and qualified provenance."""

    def setUp(self):
        """Own a disposable ordinary-file recording for each independent test."""
        self.temporary = tempfile.TemporaryDirectory(prefix="ddspc-timing-audit-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.records, self.manifest = fixture(self.directory)

    def analyze(self):
        """Exercise the actual audit with a shortened fixture-only duration requirement."""
        return audit(self.directory, self.manifest, minimum_seconds=0)

    def test_scoped_verifier_defers_inactive_dm(self):
        """Inactive DM fails the historical full profile and passes explicitly scoped ADC/WFS acquisition."""
        old = verify(self.directory, minimum_seconds=0)
        self.assertFalse(old["transport_pass"])
        self.assertIn("dm: insufficient observations", old["failures"])
        new = verify(self.directory, minimum_seconds=0, profile="adc-wfs")
        self.assertTrue(new["transport_pass"], new)
        self.assertEqual(new["dm_transport"], "deferred")
        self.assertFalse(new["training_allowed"])
        with self.assertRaises(ValueError):
            verify(self.directory, profile="unknown")

    def test_required_losses_and_short_wfs_coverage_fail(self):
        """A selected profile cannot hide required-source gaps, losses, bad rates or insufficient time coverage."""
        path = self.directory / "summary.json"
        summary = json.loads(path.read_text())
        summary["sources"][1]["counter_gaps"] = 1
        path.write_text(json.dumps(summary))
        self.assertFalse(verify(self.directory, minimum_seconds=0, profile="adc-wfs")["transport_pass"])
        summary["sources"][1]["counter_gaps"] = 0
        summary["stopped_ns"] = MONO+60000000000
        path.write_text(json.dumps(summary))
        result = verify(self.directory, profile="adc-wfs")
        self.assertIn("modal: insufficient observer-time coverage", result["failures"])
        self.assertIn("camera: insufficient observer-time coverage", result["failures"])

    def test_fixed_four_to_one_phase_and_permissions(self):
        """Known synthetic origin/clock evidence recovers a counter-locked lattice without authorizing training."""
        result = self.analyze()
        self.assertTrue(result["transport"]["transport_pass"])
        self.assertTrue(result["offline_prediction_timing_ready"], result["timing_review_blockers"])
        phase = result["four_to_one_phase"]
        self.assertEqual(phase["selected_residue"], 1)
        self.assertAlmostEqual(phase["selected"]["offset_us"], 30, delta=.2)
        self.assertEqual(phase["selected"]["counter_stride_failures"], 0)
        self.assertAlmostEqual(phase["selected"]["drift_ppm"], 0, delta=.01)
        self.assertEqual(result["causal_availability"]["future_samples_admitted"], 0)
        for key in ("training_allowed", "dm_output_allowed", "exposure_qualified",
                    "physical_application_qualified", "recorded_ddspc_inputs_qualified"):
            self.assertFalse(result[key])
        self.assertFalse(phase["hardware_phase_lock_qualified"])
        self.assertEqual(result["future_synco_constraints"]["nominal_slack_us"], 15)

    def test_unknown_clocks_are_not_inferred_from_matching_numbers(self):
        """Correct-looking numeric epochs and observer declarations do not establish source clock provenance."""
        result = audit(self.directory, minimum_seconds=0)
        self.assertFalse(result["offline_prediction_timing_ready"])
        self.assertFalse(result["four_to_one_phase"]["available"])
        self.assertTrue(result["observer_receipt_phase"]["available"])
        self.assertFalse(result["modal_camera_association"]["qualified"])
        self.manifest["sources"]["camera"]["publication"]["verified"] = False
        self.assertFalse(self.analyze()["offline_prediction_timing_ready"])

    def test_incompatible_clock_and_provenance_claims(self):
        """TAI is not silently treated as realtime; attestation requires a clock, host and source evidence."""
        self.manifest["sources"]["camera"]["publication"]["clock"] = "tai"
        result = self.analyze()
        self.assertFalse(result["four_to_one_phase"]["available"])
        self.manifest["sources"]["camera"]["publication"]["same_host"] = False
        with self.assertRaises(ValueError):
            self.analyze()

    def test_mapping_brackets_steps_stale_anchors_and_drift(self):
        """Mapping preserves interval uncertainty and rejects excessive width, old anchors and clock-step crossings."""
        m = ClockMap({"realtime": [(1000000, REAL, 100)]})
        point = m.map(REAL, 1000000, "realtime")
        self.assertEqual(point[:2], (999950, 50))
        self.assertIsNone(m.map(REAL, 5000000, "realtime"))
        self.assertIsNone(ClockMap({"realtime": [(1000000, REAL, 100000)]}).map(REAL, 1000000, "realtime"))
        m = ClockMap({"realtime": [(1000000, REAL, 100), (2000000, REAL+1100000, 100)]})
        self.assertEqual(len(m.steps), 1)
        self.assertIsNone(m.map(REAL+500000, 2000000, "realtime"))
        drift = ClockMap({"realtime": [(1000000, REAL, 100), (2000000, REAL+1000100, 100)]})
        self.assertEqual(drift.steps, [])
        # A 10-us step cannot be hidden inside the separate 25-us mapping acceptance budget.
        small_step = ClockMap({"realtime": [(1000000, REAL, 100), (2000000, REAL+1010000, 100)]})
        self.assertEqual(len(small_step.steps), 1)
        self.assertGreater(drift.map(REAL+500000, 1500000, "realtime")[1], 50)

    def test_mapping_rejection_reasons_do_not_change_admission(self):
        """Every rejected mapping has a bounded explanation, with exactly the existing acceptance decisions."""
        normal = ClockMap({"realtime": [(1000000, REAL, 100)]})
        cases = [(normal, REAL, 1000000, "realtime", None),
                 (normal, 0, 1000000, "realtime", "invalid_timestamp"),
                 (normal, REAL, 1000000, "tai", "no_beacons_for_clock"),
                 (normal, REAL, 5000000, "realtime", "anchor_too_old"),
                 (ClockMap({"realtime": [(1000000, REAL, 100000)]}), REAL, 1000000,
                  "realtime", "uncertainty_exceeds_limit"),
                 (ClockMap({"realtime": [(1000000, REAL, 100), (2000000, REAL+1100000, 100)]}),
                  REAL+500000, 2000000, "realtime", "clock_step_crossing")]
        for mapper, raw, available, domain, reason in cases:
            diagnostics = {}
            result = mapper.map(raw, available, domain, diagnostics)
            self.assertEqual(result, mapper.map(raw, available, domain))
            if reason:
                self.assertEqual(diagnostics["rejections"], {reason: 1})
        s = as_series([record(0, n+1, n*125000+1000, n*125000+2000, n*125000+3000) for n in range(10)])
        for spec, reason in (({"clock": "unknown", "same_host": False, "verified": False}, "source_clock_unknown"),
                             ({"clock": "realtime", "same_host": False, "verified": False}, "same_host_domain_not_established")):
            diagnostics = {}
            mapped_series(s, spec, normal, "acquired", diagnostics)
            self.assertEqual(diagnostics["total"]-diagnostics["mapped"], sum(diagnostics["rejections"].values()))
            self.assertEqual(diagnostics["rejections"][reason], 10)
            self.assertEqual(len(diagnostics["examples"][reason]), 4)

    def test_audit_reports_beacon_coverage_and_per_field_failures(self):
        """A manifest cannot manufacture unrecorded TAI beacons; shared REALTIME anchors can support camera data."""
        self.manifest["sources"]["modal"]["publication"]["clock"] = "tai"
        result = self.analyze()
        self.assertEqual(result["mapping_diagnostics"]["modal_publication"]["rejections"], {"no_beacons_for_clock": 40})
        self.assertGreater(result["local_clock_beacon_coverage"]["realtime"]["retained_beacons"], 0)
        self.assertEqual(result["local_clock_beacon_coverage"]["tai"]["retained_beacons"], 0)
        self.assertFalse(result["synchronization_reference_qualified"])
        self.assertFalse(result["wfs_processing_deadline_qualified"])
        self.assertFalse(result["synco_slot_association_qualified"])

    def test_clock_step_and_restart_do_not_bridge_phase(self):
        """Clock discontinuities and source restarts block readiness instead of renumbering phase or joining old IDs."""
        midpoint = len(self.records)//2
        for r in self.records[midpoint:]:
            if r["source"] == 0:
                r["clock_now_ns"] += 1000000
        write_fixture(self.directory, self.records)
        result = self.analyze()
        self.assertTrue(result["clock_steps"])
        self.assertFalse(result["offline_prediction_timing_ready"])
        self.records, self.manifest = fixture(self.directory)
        for r in self.records:
            if r["source"] == 0 and r["counter"] > 80:
                r["generation"], r["counter"] = 2, r["counter"]-80
        write_fixture(self.directory, self.records)
        result = self.analyze()
        self.assertEqual(result["cadence"]["adc"]["incarnations"], 2)
        self.assertGreater(result["four_to_one_phase"]["selected"]["segments"], 1)
        self.assertFalse(result["offline_prediction_timing_ready"])

    def test_missing_samples_keep_counter_phase_and_report_discontinuity(self):
        """Dropping one residue sample does not regroup subsequent array indexes into a new phase."""
        self.records = [r for r in self.records if not (r["source"] == 0 and r["counter"] == 41)]
        write_fixture(self.directory, self.records)
        result = self.analyze()
        self.assertEqual(result["four_to_one_phase"]["selected_residue"], 1)
        self.assertGreater(result["four_to_one_phase"]["selected"]["unpaired_camera_frames"], 0)
        self.assertFalse(result["offline_prediction_timing_ready"])

    def test_wrong_phase_and_delay_adjustment_candidates(self):
        """Constant offset errors and abrupt delay changes remain visible, without claiming actual resynchronization."""
        self.manifest["synchronization"]["expected_publication_phase_us"] = 100
        result = self.analyze()
        self.assertIn("Expected four-to-one phase tolerance exceeded", result["timing_review_blockers"])
        for r in self.records:
            if r["source"] == 0 and r["counter"] > 80:
                r["acquired_raw_ns"] += 80000
                r["published_raw_ns"] += 80000
                r["available_mono_ns"] += 80000
                r["received_mono_ns"] += 80000
                r["clock_now_ns"] += 80000
        write_fixture(self.directory, self.records)
        result = self.analyze()
        self.assertGreater(result["four_to_one_phase"]["selected"]["candidate_phase_discontinuities"], 0)

    def test_modal_origin_requires_unique_evidenced_contract(self):
        """Equal counters or timestamps alone cannot qualify parent identity; constant tags and reuse fail."""
        rows = [r for r in self.records if r["source"] == 1]
        camera = as_series([r for r in self.records if r["source"] == 2])
        candidate = associations(as_series(rows), camera, {"method": "counter1"})
        self.assertEqual(candidate["candidates"]["counter1"]["matched"], 40)
        self.assertFalse(candidate["qualified"])
        for r in rows:
            r["counter1"] = 1
        result = associations(as_series(rows), camera, self.manifest["association"])
        self.assertFalse(result["qualified"])
        self.assertEqual(result["candidates"]["counter1"]["reused"], 39)
        duplicate = deepcopy(self.records[0])
        duplicate["source"], duplicate["counter"], duplicate["generation"] = 2, 1, 2
        camera.append(duplicate)
        result = associations(as_series(rows), camera, self.manifest["association"])
        self.assertGreater(result["candidates"]["counter1"]["ambiguous"], 0)

    def test_causal_delivery_latency_and_uncertain_boundary(self):
        """An earlier acquisition delivered after a deadline cannot be used; uncertainty may defer it to a later one."""
        rows = [record(0, 1, 100, 110, 120), record(0, 2, 200, 210, 400)]
        adc = as_series(rows)
        modal = as_series([record(1, 1, 0, 0, 300), record(1, 2, 0, 0, 500)])
        result = causal_audit(adc, modal, ([MONO+100, MONO+200], [0, 0], [0, 0]),
                              ([MONO+110, MONO+210], [0, 0], [0, 0]))
        self.assertEqual([r["last_counter"] for r in result["deadline_examples"]], [1, 2])
        rows = [record(0, 1, 100, 110, 120)]
        modal = as_series([record(1, 1, 0, 0, 130), record(1, 2, 0, 0, 200)])
        result = causal_audit(as_series(rows), modal, ([MONO+100], [50], [0]), ([MONO+110], [50], [0]))
        self.assertEqual(result["no_eligible_adc"], 1)
        self.assertEqual(result["deadline_examples"][0]["last_counter"], 1)
        self.assertEqual(result["future_samples_admitted"], 0)

    def test_phase_drift_unwrap_and_jitter(self):
        """A slow counter-labelled phase drift is measured across the +/- half-frame boundary."""
        adc_rows, camera_rows, times = [], [], []
        for n in range(1000):
            t = MONO+n*500000
            camera_rows.append(record(2, n+1, 0, 0, t-MONO))
            times.append(t+200000+n*100)
            adc_rows.append(record(0, 1+4*n, 0, 0, t-MONO+200000+n*100))
        result = phase_audit(as_series(adc_rows), as_series(camera_rows),
                             (times, [0]*1000, [0]*1000),
                             ([MONO+n*500000 for n in range(1000)], [0]*1000, [0]*1000), residue=1)
        self.assertAlmostEqual(result["selected"]["drift_ppm"], 200, delta=.01)

    def test_105us_publication_alternation_has_exact_step_decomposition(self):
        """Stable 500-us camera acquisition plus alternating publication delay produces 105-us phase changes."""
        adc = as_series([r for r in self.records if r["source"] == 0])
        camera = as_series([r for r in self.records if r["source"] == 2])
        for k in range(len(camera.counter)):
            camera.published[k] += 105000*(k % 2)
            camera.available[k] += 105000*(k % 2)
        adc_times = [t-REAL+MONO for t in adc.acquired]
        camera_times = [t-REAL+MONO for t in camera.published]
        result = phase_audit(adc, camera, (adc_times, [0]*len(adc_times), adc.epoch),
                             (camera_times, [0]*len(camera_times), camera.epoch), residue=1)
        self.assertAlmostEqual(result["adjacent_interval_median_us"], 605)
        self.assertAlmostEqual(result["period_us"], 500, delta=3)
        selected = result["selected"]
        self.assertEqual(selected["pair_anchor_count"], 1)
        self.assertTrue(result["counter_rule_is_imposed"])
        for row in selected["phase_step_examples"]:
            self.assertEqual(row["adc_counter_delta"], 4)
            self.assertEqual(row["camera_counter_delta"], 1)
            self.assertEqual(row["raw_camera_acquisition_step_ns"], 500000)
            self.assertEqual(row["mapped_adc_step_ns"], 500000)
            self.assertIn(row["mapped_reference_step_ns"], (395000, 605000))
            self.assertEqual(abs(row["mapped_phase_step_ns"]), 105000)
            self.assertEqual(row["mapped_minus_raw_phase_step_ns"], 0)

    def test_camera_marker_comparison_is_independent_of_adc_pairing(self):
        """Same-record acquisition/publication latency changes reveal publication alternation without any ADC join."""
        for r in self.records:
            if r["source"] in (1, 2) and r["counter"] % 2 == 0:
                for key in ("published_raw_ns", "received_mono_ns", "available_mono_ns", "clock_now_ns"):
                    r[key] += 105000
        write_fixture(self.directory, self.records)
        result = self.analyze()
        markers = result["camera_marker_timing"]
        self.assertEqual(markers["paired_mapped_records"], 40)
        self.assertAlmostEqual(markers["acquisition_marker_intervals"]["median_us"], 500, delta=1)
        self.assertAlmostEqual(markers["publication_minus_acquisition_marker_change"]["p99_abs_us"], 105, delta=1)
        self.assertFalse(result["synchronization_reference_qualified"])

    def test_nearest_any_sample_can_create_105us_but_frozen_residue_does_not(self):
        """A 20-us marker change across a 125-us nearest-sample boundary is a distinct 105-us pairing artifact."""
        adc = as_series([r for r in self.records if r["source"] == 0])
        camera = as_series([r for r in self.records if r["source"] == 2])
        for k in range(len(camera.counter)):
            camera.published[k] = REAL+1000000+k*500000+90000+20000*(k % 2)
        a = [t-REAL+MONO for t in adc.acquired]
        p = [t-REAL+MONO for t in camera.published]
        # Deliberately reproduce a different, nearest-any-sample heuristic only as a test foil.
        nearest = [min(range(len(a)), key=lambda i: abs(a[i]-t)) for t in p]
        phase = [a[i]-t for i, t in zip(nearest, p)]
        self.assertEqual({abs(b-c) for b, c in zip(phase, phase[1:])}, {105000})
        self.assertEqual({adc.counter[b]-adc.counter[c] for b, c in zip(nearest[1:], nearest)}, {3, 5})
        result = phase_audit(adc, camera, (a, [0]*len(a), adc.epoch), (p, [0]*len(p), camera.epoch), residue=1)
        for row in result["selected"]["phase_step_examples"]:
            self.assertEqual(row["adc_counter_delta"], 4)
            self.assertEqual(abs(row["mapped_phase_step_ns"]), 20000)

    def test_mapping_holes_do_not_rephase_fixed_counter_pairs(self):
        """Alternating missing camera mappings preserve counter-span cadence and do not shift subsequent ADC residue."""
        adc = as_series([r for r in self.records if r["source"] == 0])
        camera = as_series([r for r in self.records if r["source"] == 2])
        a = [t-REAL+MONO for t in adc.acquired]
        p = [t-REAL+MONO if k % 2 else -1 for k, t in enumerate(camera.published)]
        a[44] = -1
        result = phase_audit(adc, camera, (a, [0]*len(a), adc.epoch), (p, [0]*len(p), camera.epoch), residue=1)
        self.assertIsNone(result["adjacent_interval_median_us"])
        self.assertEqual(result["period_us"], 500)
        self.assertEqual(result["selected"]["unpaired_camera_frames"], 1)
        self.assertEqual(result["selected"]["pair_anchor_count"], 1)
        for row in result["selected"]["phase_step_examples"]:
            self.assertEqual(row["mapped_phase_step_ns"], 0)

    def test_whole_wfs_cycle_skip_cannot_hide_behind_wrapped_phase(self):
        """A producer can skip an upstream frame while keeping publication counters consecutive; timing must expose it."""
        for r in self.records:
            if r["source"] in (1, 2) and r["counter"] > 20:
                for key in ("acquired_raw_ns", "published_raw_ns", "received_mono_ns", "available_mono_ns", "clock_now_ns"):
                    r[key] += 500000
        write_fixture(self.directory, self.records)
        result = self.analyze()
        selected = result["four_to_one_phase"]["selected"]
        self.assertGreater(selected["candidate_phase_discontinuities"], 0)
        self.assertGreater(selected["absolute_expected_phase_error"]["max_us"], 400)
        self.assertFalse(result["offline_prediction_timing_ready"])

    def test_bounded_phase_slew_is_reported_without_claiming_resynchronization(self):
        """A gradual correction can change phase without crossing the large-step detector threshold."""
        for r in self.records:
            if r["source"] == 0:
                shift = max(0, min(10, r["counter"]-80))*5000
                for key in ("acquired_raw_ns", "published_raw_ns", "received_mono_ns", "available_mono_ns", "clock_now_ns"):
                    r[key] += shift
        write_fixture(self.directory, self.records)
        result = self.analyze()
        selected = result["four_to_one_phase"]["selected"]
        self.assertGreater(selected["candidate_phase_adjustments_or_jitter"], 0)
        self.assertEqual(selected["candidate_phase_discontinuities"], 0)
        self.assertFalse(result["four_to_one_phase"]["hardware_phase_lock_qualified"])

    def test_jitter_preserves_lattice_and_integer_identities(self):
        """Microsecond jitter is measured without changing full-width counters or learning a new sample grouping."""
        for r in self.records:
            r["counter"] += 2**53
            if r["source"] == 1:
                r["counter1"] += 2**53
            if r["source"] == 0:
                jitter = ((r["counter"]//4) % 3-1)*3000
                for key in ("acquired_raw_ns", "published_raw_ns", "received_mono_ns", "available_mono_ns", "clock_now_ns"):
                    r[key] += jitter
        write_fixture(self.directory, self.records)
        result = self.analyze()
        self.assertTrue(result["offline_prediction_timing_ready"], result["timing_review_blockers"])
        self.assertGreater(result["four_to_one_phase"]["selected"]["detrended_jitter"]["stddev_us"], 1)
        self.assertEqual(result["four_to_one_phase"]["selected"]["counter_stride_failures"], 0)

    def test_stale_adc_and_incomplete_windows_remain_explicit(self):
        """An old held value is reported stale rather than a new measurement, even though its receipt is causal."""
        adc = as_series([record(0, 1, 100, 110, 120)])
        modal = as_series([record(1, 1, 0, 0, 1000000)])
        result = causal_audit(adc, modal, ([MONO+100], [0], [0]), ([MONO+110], [0], [0]))
        self.assertEqual(result["older_than_one_wfs_period"], 1)
        self.assertEqual(result["incomplete_500us_diagnostic_windows"], 1)
        self.assertEqual(result["future_samples_admitted"], 0)

    def test_expected_wakeup_phase_is_not_camera_publication_phase(self):
        """The hardware delay-model reference cannot be relabeled as a publication or exposure event."""
        self.manifest["synchronization"]["expected_reference"] = "semaphore_receipt"
        with self.assertRaisesRegex(ValueError, "camera_publication"):
            self.analyze()

    def test_publication_cutoff_is_distinct_from_observer_deadline(self):
        """Later observer receipt must not be relabeled as the producer's processing/publication time."""
        result = self.analyze()
        observer = result["causal_availability"]["deadline_examples"][0]
        publication = result["causal_publication_cutoff"]["deadline_examples"][0]
        self.assertLess(publication["deadline_ns"], observer["deadline_ns"])
        self.assertLess(publication["last_counter"], observer["last_counter"])
        self.assertIn("actual processing deadline unknown", result["causal_publication_cutoff"]["deadline_reference"])

    def test_phase_and_clock_attestations_require_event_evidence(self):
        """Fitting a favorable phase or selecting an epoch is insufficient evidence to advance the timing gate."""
        self.manifest["synchronization"]["expected_phase_evidence"] = ""
        self.assertFalse(self.analyze()["offline_prediction_timing_ready"])
        self.manifest["sources"]["camera"]["publication"]["event"] = "unknown"
        with self.assertRaisesRegex(ValueError, "event"):
            self.analyze()

    def test_delay_model_reference_and_synco_budget(self):
        """Default 409.4-us delay refers to wakeup, and nominal 15-us DM slack is never scheduling authority."""
        model = synchronization_model(500000, {})
        self.assertAlmostEqual(model["modeled_delay_us"], 409.4)
        self.assertAlmostEqual(model["adc_lattice_phase_us"], 34.4)
        self.assertIn("semaphore_receipt", model["reference"])
        self.assertAlmostEqual(synchronization_model(500000, {"post_delay_us": -500})["modeled_delay_us"], 409.4)
        budget = synco_constraints(500000, {})
        self.assertEqual(budget["nominal_slack_us"], 15)
        self.assertEqual(budget["assumed_input_ready_lead_us"], 70)
        self.assertFalse(budget["scheduling_qualified"])
        self.assertFalse(budget["dm_output_allowed"])
        self.assertLess(synco_constraints(400000, {})["nominal_slack_us"], 0)

    def test_cli_is_read_only_and_reports_unqualified_data(self):
        """Both CLIs consume only ordinary files and preserve their bytes while printing separate reports."""
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        root = Path(__file__).resolve().parent
        for name, arguments in (("verify_native_observation.py", ["--profile", "adc-wfs"]),
                                ("native_timing_audit.py", ["--minimum-seconds", "0"])):
            result = subprocess.run([sys.executable, str(root/name), str(self.directory), *arguments],
                                    capture_output=True, text=True)
            report = json.loads(result.stdout)
            self.assertFalse(report["training_allowed"])
            if name.startswith("native_timing"):
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(report["offline_prediction_timing_ready"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.directory.iterdir()})
        result = subprocess.run([sys.executable, str(root/"native_timing_audit.py"), str(self.directory),
                                 "--minimum-seconds", "0", "--require-timing-ready"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_analysis_modules_have_no_runtime_or_mutating_imports(self):
        """The installed analysis entrypoints have only file-reading/numeric standard-library dependencies."""
        allowed = {"argparse", "array", "bisect", "collections", "csv", "json", "math", "pathlib",
                   "statistics", "struct", "verify_native_observation", "inspect_native_commands"}
        root = Path(__file__).resolve().parent
        for name in ("native_timing_audit.py", "verify_native_observation.py", "inspect_native_commands.py"):
            tree = ast.parse((root/name).read_text())
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.add(node.module.split(".")[0])
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, {"write_text", "write_bytes", "unlink", "mkdir", "chmod"})
            self.assertLessEqual(imports, allowed)


if __name__ == "__main__":
    unittest.main()
