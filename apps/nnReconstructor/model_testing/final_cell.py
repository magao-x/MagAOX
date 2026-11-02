r"""\file final_cell.py
\brief Replace the notebook's TensorRT export cell with this shared-catalog export.

Run after the existing SavedModel evaluation/export cell on the supported GPU host.
The app scans ENGINE_DIR at startup; this cell does not restart or enable it.
"""

import sys
from pathlib import Path

sys.path.insert(0, "/home/rlandman/MagAOX/apps/nnReconstructor/model_testing")
from export_catalog_engine import export_catalog_engine

ENGINE_DIR = Path("/home/rlandman/NNReconstructor_2026B/engines")
engine_path = export_catalog_engine(MODEL_DIR, ENGINE_DIR)
