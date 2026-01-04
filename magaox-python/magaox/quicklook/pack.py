import math
import os
import time
import typing
from scipy.stats import mode
from tqdm import tqdm
import concurrent.futures
import shutil
from concurrent import futures
import tempfile
import sys
import fixr
import fsspec
import logging
import numcodecs
from numcodecs import Blosc, Delta
import zarr
import numpy as np
from magaox import db
import xconf
import psycopg
from psycopg.sql import SQL, Identifier, Literal

from ..constants import FOLDER_TIMESTAMP_FORMAT
from .core import datestamp_strings_from_ts, PathRewriteConfig


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

CHUNK_DEFAULT_MB = 1000
TELEM_ENTRIES_CHUNK = 10_000
DEFAULT_STREAMS = [
    'camsci1',
    'camsci2',
    'camwfs',
    'camllowfs',
    'camflowfs',
    'camlowfs',
    'camacq',
    'camtip',
    'dm00disp',
    'dm01disp',
    'dm02disp',
]

@xconf.config
class StreamConfig:
    name: str = xconf.field(help="Name of the camera channel")
    chunk_size_mb: int = xconf.field(
        default=CHUNK_DEFAULT_MB, help="Number of frames per chunk"
    )


class EmptyKeyError(Exception):
    pass

def get_or_zeros(root, key, shape, chunks, dtype):
    if key not in root:
        return root.zeros(
                key, shape=shape, chunks=chunks, dtype=dtype
            )
    else:
        arr = root[key]
        if arr.shape != shape or (arr.dtype != np.object_ and arr.dtype != dtype):
            raise RuntimeError(f"Cannot overwrite {arr.dtype=} and {arr.shape=} with new shape {shape} and dtype {dtype}")
        else:
            return arr

def infer_dtype(val):
    if not np.isscalar(val):
        raise ValueError(
            f"Got {repr(val)} ({type(val)}) but this only works on scalar types"
        )
    elif isinstance(val, str):
        return str
    else:
        return np.result_type(val)


def tel2zarr(msg: dict, path: list[str], root, row_count, chunk_size) -> list[tuple[str, zarr.Array, callable]]:
    flattened_msg_keys = []
    for k in msg:
        if not isinstance(msg[k], dict):
            group_path = "/".join(path + [k])
            # infer dtype from msg[k]
            shape = (row_count,)
            if isinstance(msg[k], list) and len(msg[k]) > 0:
                shape += (len(msg[k]),)
                dtype = infer_dtype(msg[k][0])
            elif isinstance(msg[k], list):
                # length-zero lists cannot have useful information
                continue
            else:
                dtype = infer_dtype(msg[k])
            arr = get_or_zeros(root, group_path, shape=shape, chunks=(chunk_size,), dtype=dtype)

            def accessor(the_msg, k=k):
                for part in path:
                    the_msg = the_msg[part]
                return the_msg[k]

            flattened_msg_keys.append((group_path, arr, accessor))
        else:
            flattened_msg_keys.extend(
                tel2zarr(msg[k], path + [k], root, row_count, chunk_size)
            )
    return flattened_msg_keys


def datetime_to_seconds_nanos(dt):
    seconds = int(dt.timestamp())
    nanoseconds = int(dt.microsecond * 1000)
    return (seconds, nanoseconds)


def unpack_one_xrif(
    local_path, idx, frames_per_xrif_chunk, frames_tmp, times_tmp, log
) -> int:
    log.debug(f"{local_path=} {idx=} {frames_per_xrif_chunk=}")
    with open(local_path, "rb") as f:
        frames = fixr.XrifReader(f).copy_data()
        # n.b. after reading `frames`, file `f` has seeked (sought?) to
        # the beginning of the timing xrif archive concatenated onto
        # the image data
        times = fixr.XrifReader(f).copy_data()
        frames = frames.reshape((-1,) + frames.shape[2:])
        # time rows are [frame_num, acqsec, acqnsec, wrtsec, wrtnsec]
        times = times.reshape((-1, 5))

    if frames.shape[0] != times.shape[0]:
        log.warning(
            f"Discarding {frames.shape[0]} frames because xrif wrote {times.shape[0]} timestamps for this archive"
        )
        return idx, 0

    if frames.shape[1:] != frames_tmp.shape[1:]:
        log.warning(
            f"Skipping {frames.shape[0]} frames because {frames.shape[1:]=} but {frames_tmp.shape[1:]=}"
        )
        return idx, 0

    start_idx = idx * frames_per_xrif_chunk
    # zero for timestamp seconds is used as a sentinel for empty frames, which we should skip
    actual_frames_mask = times[:, 1] != 0
    n_actual_frames = np.count_nonzero(actual_frames_mask)
    n_frames_loaded = actual_frames_mask.shape[0]
    if n_frames_loaded != n_actual_frames:
        log.warning(f"Got {n_actual_frames=} from {local_path} but {n_frames_loaded=} and {frames_per_xrif_chunk=}")
        raise RuntimeError(f"this one {local_path}")
    if n_actual_frames != frames_per_xrif_chunk:
        log.debug(
            f"Got {n_actual_frames=} from {local_path} but expected {frames_per_xrif_chunk=}, filling in the frames we have"
        )
    actual_frames = frames[actual_frames_mask]
    actual_times = times[actual_frames_mask]
    assert not np.any(actual_times[:, 1] == 0), f"{actual_times=}"
    frames_tmp[start_idx : start_idx + n_actual_frames] = actual_frames
    times_tmp[start_idx : start_idx + n_actual_frames] = actual_times
    return idx, n_actual_frames


def preprocess_path(origin_host, origin_path):
    if origin_host == "exao2":
        local_path = origin_path.replace("/opt/MagAOX", "/srv/rtc/data")
    elif origin_host == "exao3":
        local_path = origin_path.replace("/opt/MagAOX", "/srv/icc/data")
    else:
        local_path = origin_path
    return local_path


def infer_common_xrif_cube_size_dtype(paths):
    frames = []
    planes = []
    height = []
    width = []
    shapes = set()
    img_dtype = None
    times_dtype = None
    for p in paths:
        with open(p, "rb") as fh:
            xr = fixr.XrifReader(fh)
            img_dtype = xr.array.dtype
            fr, pl, ht, wd = xr.shape
            shapes.add(xr.shape)
            frames.append(fr)
            planes.append(pl)
            height.append(ht)
            width.append(wd)
            xr2 = fixr.XrifReader(fh)
            times_dtype = xr2.array.dtype

    common_shape = (
        np.max(frames),
        mode(planes).mode,
        mode(height).mode,
        mode(width).mode,
    )
    log.debug(f"Guessed {common_shape=} from {shapes=}")
    return common_shape, img_dtype, times_dtype


def repack_xrif_channel(
    camera_channel: StreamConfig,
    channel_grouping_root: zarr.Group,
    cur: psycopg.Connection,
    path_rewrites: list[PathRewriteConfig],
    bounds,
    chunk_size_mb,
    pool: futures.ThreadPoolExecutor,
    temp_root: str,
) -> tuple[list[str], int]:
    other_files_args = bounds + (camera_channel.name,)
    other_files_q_body = SQL("""
    FROM file_origins
    WHERE creation_time
        BETWEEN {create_from} AND {create_to}
        AND origin_path LIKE '%%' || {channel_name} || '%%.xrif'
    """).format(create_from=bounds[0], create_to=bounds[1], channel_name=camera_channel.name)
    other_files_count_q = SQL("""SELECT COUNT(*) as count {}""").format(other_files_q_body)
    log.debug(f"Checking for xrifs using: {other_files_count_q}")
    cur.execute(other_files_count_q)
    xrifs_row_count = cur.fetchone()["count"]
    if xrifs_row_count == 0:
        log.debug(f"No {camera_channel.name} XRIF archives found to process")
        return [], 0, 0

    camera_root = channel_grouping_root.require_group(camera_channel.name)
    other_files_q = (
        SQL("""
    SELECT
        origin_host, origin_path, creation_time, size_bytes
    """)
        + other_files_q_body
    )

    # xrif archives contain an unpredictable number of frames.
    # guessing a shape from the largest (by bytes) archive can get
    # confused when there's an outlier xrif (i.e. changed ROI during obs)
    #
    # instead, maybe just find the most common shape, and drop the rest
    # with a warning?
    n_shape_samples = 10
    samples_q = SQL("{} ORDER BY random() DESC LIMIT {}").format(other_files_q, n_shape_samples)
    log.debug(f"Getting a sample of paths to guess data shape: {samples_q}")
    cur.execute(samples_q)
    ex_rows = cur.fetchall()
    ex_paths = []
    for r in ex_rows:
        real_path = r["origin_path"]
        for rw in path_rewrites:
            real_path = rw.rewrite(r["origin_host"], real_path)
        ex_paths.append(real_path)

    # skip planes per frame (for now?)
    (frames_per_xrif_chunk, _, img_height, img_width), img_dtype, times_dtype = (
        infer_common_xrif_cube_size_dtype(ex_paths)
    )
    bytes_per_pix = img_dtype.itemsize
    mb_per_frame = (img_height * img_width * bytes_per_pix) / 1024 / 1024
    chunk_size = int(chunk_size_mb / mb_per_frame)
    times_width = 5  # five time values per frame are stored (idx, acq sec, acq nsec, wrt sec, wrt nsec)

    n_frames = frames_per_xrif_chunk * xrifs_row_count

    os.makedirs(temp_root, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as td:
        rechunk = zarr.open_group(f"file://" + td)
        frames_tmp = rechunk.zeros(
            "frames",
            shape=(n_frames,) + (img_height, img_width),
            chunks=(frames_per_xrif_chunk,),
            dtype=img_dtype,
        )
        times_tmp = rechunk.zeros(
            "times",
            shape=(n_frames,) + (times_width,),
            chunks=(frames_per_xrif_chunk,),
            dtype=times_dtype,
        )

        futs = []
        full_files_q = SQL("{} ORDER BY creation_time ASC").format(other_files_q)
        cur.execute(full_files_q)
        log.debug("Submitting futures...")
        orig_total_bytes, final_total_bytes = 0, 0
        for idx, row in enumerate(cur):
            orig_total_bytes += row["size_bytes"]
            local_path = preprocess_path(row["origin_host"], row["origin_path"])
            unpack_args = (
                local_path,
                idx,
                frames_per_xrif_chunk,
                frames_tmp,
                times_tmp,
                log,
            )
            futs.append(pool.submit(unpack_one_xrif, *unpack_args))
        log.info(
            f"Loading {idx + 1} {camera_channel.name} archives, total compressed size {orig_total_bytes/1024/1024:1.1f} MiB"
        )

        failed = 0
        succeeded = 0
        total_frames = 0
        good_frames_mask = np.zeros(n_frames, dtype=bool)
        # for fut in tqdm(futures.as_completed(futs), total=len(futs)):
        for fut in futures.as_completed(futs):
            idx, n_actual_frames = fut.result()
            # assert not np.any(times_tmp[frames_per_xrif_chunk * idx : frames_per_xrif_chunk * idx + n_actual_frames, 1] == 0), "Zero times"
            if n_actual_frames < 1:
                failed += 1
            else:
                total_frames += n_actual_frames
                good_frames_mask[
                    frames_per_xrif_chunk * idx :
                    frames_per_xrif_chunk * idx + n_actual_frames
                ] = True
                succeeded += 1

        # assert np.count_nonzero(good_frames_mask) == total_frames, "mask mismatched"
        log.debug(
            f"Loaded {total_frames=} of an expected {n_frames} (loads {succeeded=} {failed=})"
        )
        if failed > 0:
            log.warning(f"Failed to load {failed} files, final array will contain gaps")

        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
        # filters = [Delta(dtype='u2')]  # test showed no delta compresses better / decompresses faster
        filters = []
        cam_frames = camera_root.zeros(
            f"frames",
            shape=(total_frames,) + (img_height, img_width),
            chunks=(chunk_size,),
            dtype=img_dtype,
            filters=filters,
            compressor=compressor,
        )
        cam_times = camera_root.zeros(
            f"times",
            shape=(total_frames,) + (times_width,),
            chunks=(chunk_size,),
            dtype=times_dtype,
        )
        log.debug(f"Created {cam_frames} with {filters=} {compressor=}")
        start = time.perf_counter()
        # The "cursor" represented by `start_idx` should point to the destination where a chunk of frames/times will be copied into.
        # e.g. Before copying a single-frame span, it will be 0 and afterwards it will be 1, the location where the next span starts.
        start_idx = 0
        n_loaded_frames = good_frames_mask.shape[0]
        n_chunks_from_loaded = max(1, math.ceil(n_loaded_frames / chunk_size))
        log.debug(f"Re-chunking to {total_frames=} of {chunk_size=}, using {n_loaded_frames=} in {n_chunks_from_loaded} chunks")
        for i in range(n_chunks_from_loaded):
            chunk_start = i * chunk_size
            chunk_end = min(n_loaded_frames, (i + 1) * chunk_size)
            good_chunk_mask = good_frames_mask[chunk_start:chunk_end]
            real_frames = np.count_nonzero(good_chunk_mask)
            frames_contiguous = frames_tmp[
                chunk_start:chunk_end
            ][good_chunk_mask]
            # assert frames_contiguous.shape[0] == real_frames
            cam_frames[start_idx : start_idx + real_frames] = frames_contiguous
            times_contiguous = times_tmp[
                chunk_start:chunk_end
            ][good_chunk_mask]
            assert times_contiguous.shape[0] == real_frames
            assert not np.any(times_contiguous[:, 1] == 0), f"{good_chunk_mask=} {times_contiguous=}"
            cam_times[start_idx : start_idx + real_frames] = times_contiguous
            start_idx += real_frames

        # After all frames are copied, start_idx should be equal to the length.
        assert start_idx == total_frames, f"{start_idx=} {total_frames=}"
        dt = time.perf_counter() - start
        log.info(f"Compressed and wrote {total_frames} frames and times in {dt:1.1f} sec")
        log.debug(f"{cam_frames.nbytes=} {cam_frames.nbytes_stored=}")
        final_total_bytes += cam_frames.nbytes_stored
        log.debug(f"{cam_times.nbytes=} {cam_times.nbytes_stored=}")
        final_total_bytes += cam_times.nbytes_stored

    return ex_paths, orig_total_bytes, final_total_bytes


def repack_telem(telem_root: zarr.Group, cur: psycopg.Connection, bounds, chunk_size=TELEM_ENTRIES_CHUNK):
    cur.execute(
        """
    SELECT DISTINCT device, ec FROM telem WHERE ts BETWEEN %s AND %s;
    """,
        bounds,
    )
    device_event_code_pairs = cur.fetchall()

    for dev_ec in device_event_code_pairs:
        log.debug(f"Processing {dev_ec}")
        telem_query_args = (dev_ec["device"], dev_ec["ec"]) + bounds
        cur.execute(
            """
SELECT COUNT(*)
FROM telem
WHERE
    device = %s
    AND ec = %s
    AND ts BETWEEN %s AND %s
""",
            telem_query_args,
        )
        row_count = cur.fetchone()["count"]
        log.debug(f"{dev_ec['device']} {dev_ec['ec']} has {row_count} rows")

        telem_range_query = "SELECT * FROM telem WHERE device = %s AND ec = %s AND ts BETWEEN %s AND %s ORDER BY ts ASC"
        cur.execute(telem_range_query + " LIMIT 1", telem_query_args)
        one_example = cur.fetchone()

        this_dev_ec_path = f"{dev_ec['device']}/{dev_ec['ec']}"
        this_ec_root = telem_root.require_group(this_dev_ec_path)
        telem_element_sequence = tel2zarr(
            one_example["msg"], [], this_ec_root, row_count, chunk_size=1000
        )
        ts_dtype = [("sec", "i4"), ("nsec", "i4")]

        ts_arr = get_or_zeros(this_ec_root, "ts", shape=(row_count,), chunks=(chunk_size,), dtype=ts_dtype)
        chunks = max(1, row_count // chunk_size + 1)
        log.debug(f"Using {row_count=} {chunk_size=} gives {chunks=}")
        # allocate chunks per telem element
        per_telem_chunks = []
        buffer_size = min(chunk_size, row_count)
        for path, arr, accessor in telem_element_sequence:
            per_telem_chunks.append(
                np.zeros((buffer_size,) + arr.shape[1:], dtype=arr.dtype)
            )
            log.debug(f"Preallocating {buffer_size} {arr.dtype} elements for {path}")
        ts_chunk = np.zeros(buffer_size, dtype=ts_dtype)
        for i in range(chunks):
            log.debug(f"Chunk {i+1}")
            q = telem_range_query
            if chunks == 1:
                q = telem_range_query
            else:
                q = telem_range_query + f" LIMIT {chunk_size}"
            if i != 0:
                q += f" OFFSET {i * chunk_size}"
            log.debug(f"Querying: {q}")
            cur.execute(q, telem_query_args)

            # _count = 0
            for idx, row in enumerate(cur):
                # log.debug(f"Chunk {i+1} row {idx}")
                sec, nsec = datetime_to_seconds_nanos(row["ts"])
                ts_chunk[idx]["sec"] = sec
                ts_chunk[idx]["nsec"] = nsec
                # log.debug(f"{ts_chunk[idx]=}")
                for buffer_array, (path, _, accessor) in zip(
                    per_telem_chunks, telem_element_sequence
                ):
                    # log.debug(f"{path} {buffer_array[idx]=}")
                    buffer_array[idx] = accessor(row["msg"])
                # _count += 1

            # assert _count == idx + 1
            slice_start = i * chunk_size
            chunk_len = (
                idx + 1
            )  # for partial chunks: after loop, `idx` is last index, exclusive bound is one more
            slice_stop = slice_start + chunk_len

            # assign chunk into zarr
            for buffer_array, (path, arr, accessor) in zip(
                per_telem_chunks, telem_element_sequence
            ):
                log.debug(f"{path} {arr} {slice_start=} {slice_stop=}")
                arr[slice_start:slice_stop] = buffer_array[:chunk_len]
            ts_arr[slice_start:slice_stop] = ts_chunk[:chunk_len]


def pack_one_obs(
    span,
    streams: list[StreamConfig],
    root: zarr.Group,
    conn,
    path_rewrites: list[PathRewriteConfig],
    pool: futures.ThreadPoolExecutor,
    temp_root: str,
):
    cur = conn.cursor()
    bounds = span.begin, span.end
    paths_packed = []
    orig_total_bytes, final_total_bytes = 0, 0

    stream_root = root.require_group('stream')
    for stream in streams:
        log.info(f"Checking for {stream.name}...")
        cam_files_packed, orig_bytes_packed, final_bytes_packed = (
            repack_xrif_channel(
                stream,
                stream_root,
                cur,
                path_rewrites,
                bounds,
                stream.chunk_size_mb,
                pool,
                temp_root,
            )
        )
        paths_packed.extend(cam_files_packed)
        orig_total_bytes += orig_bytes_packed
        final_total_bytes += final_bytes_packed
        if final_bytes_packed > 0:
            log.debug(
                f"Packed {len(cam_files_packed)} files, compressed {orig_bytes_packed/1024/1024:1.1f} MiB -> {final_bytes_packed/1024/1024:1.1f} MiB ({orig_bytes_packed / final_bytes_packed:1.2f})"
            )
    repack_telem(root.require_group("telem"), cur, bounds, chunk_size=TELEM_ENTRIES_CHUNK)
    return paths_packed, orig_total_bytes, final_total_bytes
