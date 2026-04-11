"""Resolve ffmpeg for Matplotlib ``FFMpegWriter`` (system PATH or imageio-ffmpeg)."""

from __future__ import annotations

import logging
import shutil

import matplotlib as mpl

logger = logging.getLogger(__name__)


def configure_matplotlib_ffmpeg_path() -> None:
    """Set ``matplotlib.rcParams['animation.ffmpeg_path']`` to a working ffmpeg binary.

    Matplotlib invokes ``ffmpeg`` by name; many conda/env setups omit it from ``PATH``.
    The ``imageio-ffmpeg`` dependency ships a portable binary—use it when the system
    executable is not found.
    """
    path = shutil.which("ffmpeg")
    if path is not None:
        mpl.rcParams["animation.ffmpeg_path"] = path
        return
    try:
        import imageio_ffmpeg
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is not on PATH and imageio-ffmpeg could not be imported. "
            "Install ffmpeg (e.g. `conda install ffmpeg`) or install the windsocc "
            "dependency `imageio-ffmpeg`."
        ) from exc
    try:
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is not on PATH and imageio-ffmpeg did not provide a binary. "
            "Install ffmpeg (e.g. `conda install ffmpeg`) or reinstall `imageio-ffmpeg`."
        ) from exc
    logger.debug("ffmpeg not on PATH; using imageio-ffmpeg binary at %s", bundled)
    mpl.rcParams["animation.ffmpeg_path"] = bundled
