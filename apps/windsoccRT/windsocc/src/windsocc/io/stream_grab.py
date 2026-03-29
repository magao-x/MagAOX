"""Debug CLI for grabbing a live MagAO-X stream in Python and running WindsoCC.

Optimized for high-rate streams (e.g. ~2 kHz): by default each sample reads the
current shmim buffer without blocking on a new frame, so consecutive rows may
repeat or skip updates relative to the writer. Use ``--wait-new-frame`` when you
need semaphore-synced frames at lower rates.

After each ``get_data``, the script compares ``image.md.cnt0`` to the previous
grab: a step greater than 1 means the writer advanced faster than consecutive
grabs (missed frames). With ``--wait-new-frame``, a step of 1 each time indicates
you are keeping up with new frames. Use ``--cnt0-diagnostics`` for per-grab
duplicate/skip detail when not waiting.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import numpy as np

from windsocc.realtime import (
    DEFAULT_FRAME_SHAPE,
    run_embedded_batch,
    validate_frame_batch,
)


def _shmim_cnt0(image) -> int | None:
    """Return ImageStreamIO ``cnt0`` (writer frame counter) if available."""
    try:
        return int(image.md.cnt0)
    except (AttributeError, TypeError, ValueError):
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Grab a live MagAO-X shmim stream in Python and hand it directly to WindsoCC. "
            "Defaults to non-blocking reads suitable for very fast batch collection; see "
            "--wait-new-frame for blocking behavior."
        )
    )
    parser.add_argument(
        "--stream-name",
        type=str,
        required=True,
        help="MagAO-X shmim stream name to open with magaox.shmim.Image.",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        required=True,
        help="Number of live frames to collect before handing the batch into the WindsoCC pipeline.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=DEFAULT_FRAME_SHAPE[0],
        help="Expected frame height for each 2D shmim frame.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=DEFAULT_FRAME_SHAPE[1],
        help="Expected frame width for each 2D shmim frame.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ws_config.yaml used for the realtime batch.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=".",
        help="Directory where camwfs_<timestamp> batch directories should be written.",
    )
    parser.add_argument(
        "--wait-new-frame",
        action="store_true",
        help=(
            "Block on the shmim semaphore for each sample (flush + wait). Default is off: "
            "read the current buffer as fast as possible without waiting for a new frame."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5.0,
        help="Used only with --wait-new-frame: max seconds to wait per frame before timeout.",
    )
    parser.add_argument(
        "--check-before-wait",
        action="store_true",
        help=(
            "Used only with --wait-new-frame: stat the shmim inode before awaiting the semaphore."
        ),
    )
    parser.add_argument(
        "--frames-per-cube",
        type=int,
        default=512,
        help="Number of raw frames to pack into each intermediate FITS cube.",
    )
    parser.add_argument(
        "--no-movie",
        action="store_true",
        help="Skip movie generation even if MAKE_MOVIE is enabled in config.",
    )
    parser.add_argument(
        "--save-distill-pngs",
        action="store_true",
        help="Keep distill PNG products instead of skipping them for latency.",
    )
    parser.add_argument(
        "--cleanup-intermediate",
        action="store_true",
        help="Delete heavier intermediate products after the Python pipeline completes.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging level for the debug script.",
    )
    parser.add_argument(
        "--cnt0-diagnostics",
        action="store_true",
        help=(
            "After each grab, compare shmim md.cnt0 to the previous sample: log gaps (skipped writer frames) "
            "and, when not using --wait-new-frame, duplicate cnt0 (same frame sampled twice). "
            "Summary is always logged at end when cnt0 is readable."
        ),
    )
    return parser.parse_args()


def load_shmim_image(stream_name: str):
    try:
        from magaox.shmim import Image
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import magaox.shmim.Image. Ensure the MagAO-X Python package and ImageStreamIOWrap "
            "are installed in this environment before running ws_debug_stream_grab."
        ) from exc

    return Image(stream_name)


def collect_live_batch(args) -> tuple[np.ndarray, datetime]:
    image = load_shmim_image(args.stream_name)

    batch_shape = (args.frame_count, args.frame_height, args.frame_width)
    expected_frame_shape = (args.frame_height, args.frame_width)

    logging.info(
        "Collecting %d frame(s) from shmim %s into batch shape %s (each frame %s, wait_new_frame=%s)",
        args.frame_count,
        args.stream_name,
        batch_shape,
        expected_frame_shape,
        args.wait_new_frame,
    )

    batch = np.empty(batch_shape, dtype=np.float32)
    first_timestamp = None
    wait = args.wait_new_frame

    prev_cnt0: int | None = None
    cnt0_first: int | None = None
    cnt0_last: int | None = None
    skipped_writer_frames = 0
    max_cnt0_gap = 0
    duplicate_cnt0_reads = 0
    cnt0_unavailable = False

    for frame_index in range(args.frame_count):
        frame = image.get_data(
            wait=wait,
            timeout_sec=args.timeout_seconds,
            check_before_wait=args.check_before_wait,
        )
        cnt0_now = _shmim_cnt0(image)
        if cnt0_now is None:
            cnt0_unavailable = True
        else:
            if cnt0_first is None:
                cnt0_first = cnt0_now
            if prev_cnt0 is not None:
                delta = cnt0_now - prev_cnt0
                if delta > 1:
                    gap = delta - 1
                    skipped_writer_frames += gap
                    max_cnt0_gap = max(max_cnt0_gap, delta)
                    msg = (
                        "cnt0 advanced by %d from %d to %d at grab %d/%d — "
                        "writer published %d frame(s) between consecutive grabs (possible miss)."
                    ) % (delta, prev_cnt0, cnt0_now, frame_index + 1, args.frame_count, gap)
                    if wait:
                        logging.warning(msg)
                    elif args.cnt0_diagnostics:
                        logging.warning(msg)
                elif delta == 0 and not wait:
                    duplicate_cnt0_reads += 1
                    if args.cnt0_diagnostics:
                        logging.debug(
                            "cnt0 unchanged at %d on grab %d/%d (same buffer sampled twice; expected without wait).",
                            cnt0_now,
                            frame_index + 1,
                            args.frame_count,
                        )
            prev_cnt0 = cnt0_now
            cnt0_last = cnt0_now

        frame_array = np.asarray(frame, dtype=np.float32)
        if frame_array.ndim != 2:
            frame_array = np.squeeze(frame_array)
        if frame_array.shape != expected_frame_shape:
            raise ValueError(
                f"Expected stream frames with shape {expected_frame_shape}, got {frame_array.shape} "
                f"on frame {frame_index + 1}."
            )

        if first_timestamp is None:
            first_timestamp = datetime.now(timezone.utc)
            logging.info(
                "First frame received from %s with dtype=%s shape=%s",
                args.stream_name,
                frame_array.dtype,
                frame_array.shape,
            )

        np.copyto(batch[frame_index], frame_array)

    validate_frame_batch(batch, (args.frame_height, args.frame_width))
    logging.info("Collected live batch with shape=%s dtype=%s", batch.shape, batch.dtype)

    if cnt0_unavailable:
        logging.info("shmim cnt0 not read (md.cnt0 missing); skip gap/duplicate counter diagnostics.")
    elif cnt0_first is not None and cnt0_last is not None:
        total_cnt0_span = cnt0_last - cnt0_first
        expected_steps = args.frame_count - 1
        logging.info(
            "cnt0 summary: first=%d last=%d span=%d (expected ~%d steps for consecutive grabs with no writer skips)",
            cnt0_first,
            cnt0_last,
            total_cnt0_span,
            expected_steps,
        )
        if skipped_writer_frames:
            logging.warning(
                "cnt0: implied %d writer frame(s) skipped between grabs (max single gap=%d).",
                skipped_writer_frames,
                max_cnt0_gap,
            )
        elif wait:
            logging.info("cnt0: no gaps >1 between grabs — consistent with receiving each new frame once.")
        if not wait and duplicate_cnt0_reads:
            logging.info(
                "cnt0: %d grab(s) saw the same cnt0 as the previous grab (non-blocking mode; not every writer frame).",
                duplicate_cnt0_reads,
            )
    return batch, first_timestamp or datetime.now(timezone.utc)


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    batch_frames, first_timestamp = collect_live_batch(args)

    logging.info(
        "Handing live batch to windsocc.realtime.run_embedded_batch with config=%s output_root=%s",
        args.config,
        args.output_root,
    )
    result = run_embedded_batch(
        batch_frames=batch_frames,
        first_timestamp=first_timestamp,
        config_path=args.config,
        output_root=args.output_root,
        frames_per_cube=args.frames_per_cube,
        no_movie=args.no_movie,
        save_distill_pngs=args.save_distill_pngs,
        cleanup_intermediate=args.cleanup_intermediate,
    )

    logging.info("WindsoCC batch finished: %s", result["run_dir"])


if __name__ == "__main__":
    main()
