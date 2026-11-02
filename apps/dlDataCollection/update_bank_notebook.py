#!/usr/bin/env python3
r"""\file update_bank_notebook.py
\brief Add collector access and optional dated paths without replacing a bank recipe.

Run once against Generate_random_modevals.ipynb. This edits the notebook only;
the operator runs its new permissions cell on the machine holding the banks.
"""

import argparse
import ast
import json
import shutil
from pathlib import Path

ACCESS_CELL = """# Allow the xsup collector to read this bank automatically.
# Run this cell alone to repair an existing bank without regenerating it.
import os
import shutil
import subprocess

if shutil.which("setfacl") is None:
    raise RuntimeError("Bank access was not configured: setfacl is required on the bank host.")

bank = BANK_DIR.resolve(strict=True)
if not (bank / "manifest.conf").is_file():
    raise RuntimeError(f"No completed bank manifest in {bank}; finish saving the bank first.")

# Parent directories need traversal only; other home-directory contents stay private.
parents = [str(path) for path in bank.parents if path.stat().st_uid == os.geteuid()]
if parents:
    subprocess.run(["setfacl", "-m", "u:xsup:--x", "--", *parents], check=True)
subprocess.run(["setfacl", "-R", "-m", "u:xsup:r-X", "--", str(bank)], check=True)
print("Collector read access configured for xsup:", bank)
"""


def dated_bank_path(cells):
    """Replace one literal BANK_DIR assignment with editable date/name parameters."""
    for cell in cells:
        source = "".join(cell.get("source", []))
        if "# Automatic dated bank location." in source:
            return
    matches = []
    for cell in cells:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code" or "BANK_DIR" not in source:
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "BANK_DIR" for target in node.targets
            ):
                matches.append((cell, source, node))
    if len(matches) != 1:
        raise ValueError("Expected one literal BANK_DIR = Path(...) setting; notebook left unchanged.")
    cell, source, node = matches[0]
    value = node.value
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id == "Path" and len(value.args) == 1 and not value.keywords):
        raise ValueError("BANK_DIR must be a literal Path(...) for automatic conversion.")
    bank = Path(ast.literal_eval(value.args[0]))
    if not bank.is_absolute() or len(bank.parent.name) != 8 or not bank.parent.name.isdigit():
        raise ValueError("Expected BANK_DIR under banks/YYYYMMDD/name; custom paths are left unchanged.")
    replacement = (
        "# Automatic dated bank location.\n"
        "from datetime import date\n"
        f"BANK_ROOT = Path({str(bank.parent.parent)!r})\n"
        "BANK_DATE = date.today().strftime('%Y%m%d')  # Local date; override for another observing date.\n"
        f"BANK_NAME = {bank.name!r}  # Recipe name, independent of the physical ND; change suffix for a new bank.\n"
        "BANK_DIR = BANK_ROOT / BANK_DATE / BANK_NAME  # Or enter a full Path(...) here.\n"
    )
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = replacement.splitlines(keepends=True)
    cell["source"] = lines
    cell["execution_count"] = None
    cell["outputs"] = []


def update_notebook(path, dated_paths=False):
    """Insert access setup, optionally date the bank path, and preserve a backup."""
    path = Path(path)
    notebook = json.loads(path.read_text())
    cells = [cell for cell in notebook["cells"] if cell.get("id") != "collector-bank-access"]
    saves = [
        index
        for index, cell in enumerate(cells)
        if cell["cell_type"] == "code"
        and "generate_bank(" in "".join(cell["source"])
        and "BANK_DIR" in "".join(cell["source"])
    ]
    if len(saves) != 1:
        raise ValueError("Expected one generate_bank(BANK_DIR, ...) save cell; notebook left unchanged.")
    if dated_paths:
        dated_bank_path(cells)
    cells.insert(
        saves[0] + 1,
        {
            "cell_type": "code",
            "id": "collector-bank-access",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": ACCESS_CELL.splitlines(keepends=True),
        },
    )
    suffix = ".before-dated-paths.ipynb" if dated_paths else ".before-xsup-access.ipynb"
    backup = path.with_name(path.stem + suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
    notebook["cells"] = cells
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    return backup


def main():
    """Update the selected notebook without running it or accessing any instrument."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook",
        nargs="?",
        type=Path,
        default=Path.home() / "NNReconstructor_2026B" / "Generate_random_modevals.ipynb",
    )
    parser.add_argument(
        "--dated-paths", action="store_true",
        help="Default an existing banks/YYYYMMDD/name setting to today's date, preserving the recipe.",
    )
    args = parser.parse_args()
    backup = update_notebook(args.notebook, dated_paths=args.dated_paths)
    print(f"Updated {args.notebook}; original preserved at {backup}")
    print("Reopen the notebook. Run All configures xsup access after each bank save.")
    print("For an existing bank, run only the new collector-access cell.")
    if args.dated_paths:
        print("The parameter cell now exposes BANK_ROOT, BANK_DATE, BANK_NAME and the final BANK_DIR.")
        print("Existing banks are not moved. Reuse an existing bank or choose a new name if the path exists.")


if __name__ == "__main__":
    main()
