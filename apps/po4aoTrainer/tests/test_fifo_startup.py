## \file test_fifo_startup.py
# \brief Check the installed INDI dependency against the framework's FIFO constructor.
# \ingroup po4aoTrainer_unit_test
"""Exercise local temporary FIFOs only; no instrument, server or worker is started."""

import os
import pwd
import stat
from contextlib import ExitStack

import pytest
from purepyindi2 import transports


@pytest.mark.parametrize("server_fifo_exists", [False, True])
def test_framework_fifo_constructor(tmp_path, monkeypatch, server_fifo_exists):
    """Accept XDevice's default control path whether or not indiserver is present."""
    monkeypatch.setattr(
        transports.IndiFifoConnection, "MAGAOX_USER", pwd.getpwuid(os.getuid()).pw_name
    )
    if server_fifo_exists:
        os.mkfifo(tmp_path / "indiserver.ctrl")
    make_fifo = transports.make_fifo_and_open
    with ExitStack() as handles:

        def open_fifo(path, mode):
            """Close all fixture handles even if the dependency constructor raises."""
            return handles.enter_context(make_fifo(path, mode))

        monkeypatch.setattr(transports, "make_fifo_and_open", open_fifo)
        connection = transports.IndiFifoConnection(
            name="po4aoTrainer", fifos_root=str(tmp_path)
        )
        for suffix in ("in", "out", "ctrl"):
            assert stat.S_ISFIFO((tmp_path / f"po4aoTrainer.{suffix}").stat().st_mode)
        assert connection._reader is None
        assert connection._writer is None
