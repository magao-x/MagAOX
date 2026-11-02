r"""\file test_bank_notebook_access.py
\brief Check recipe-preserving updates and collector-only bank access commands.
"""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("update_bank_notebook", Path(__file__).parents[1] / "update_bank_notebook.py")
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)
ACCESS_CELL, update_notebook = updater.ACCESS_CELL, updater.update_notebook


class BankNotebookAccessTests(unittest.TestCase):
    """Exercise the notebook update and its Linux permission cell without hardware."""

    def test_update_preserves_recipe_backup_and_is_repeatable(self):
        """Keep user settings and outputs intact and insert exactly one access cell."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "Generate_random_modevals.ipynb"
            cells = [
                {"cell_type": "code", "source": ["BANK_DIR = 'my recipe'\n"]},
                {"cell_type": "code", "source": ["generate_bank(BANK_DIR, layers, calibration)\n"]},
                {"cell_type": "markdown", "source": ["My instructions"]},
            ]
            original = json.dumps({"cells": cells, "metadata": {"operator": "rlandman"}})
            path.write_text(original)
            backup = update_notebook(path)
            update_notebook(path)
            updated = json.loads(path.read_text())
            self.assertEqual(backup.read_text(), original)
            self.assertEqual(updated["metadata"], {"operator": "rlandman"})
            self.assertEqual(updated["cells"][0:2], cells[0:2])
            self.assertEqual(updated["cells"][3], cells[2])
            self.assertEqual(updated["cells"][2]["id"], "collector-bank-access")
            self.assertEqual(len(updated["cells"]), 4)

    def test_access_is_targeted_and_paths_are_not_shell_commands(self):
        """Grant traversal on owned ancestors and read-only bank access using literal argv."""
        with tempfile.TemporaryDirectory() as folder:
            bank = Path(folder) / "bank with spaces; $(ignored)"
            bank.mkdir()
            (bank / "manifest.conf").write_text("format=dlDataCollection-1\n")
            with patch("shutil.which", return_value="/usr/bin/setfacl"), patch("subprocess.run") as run:
                exec(ACCESS_CELL, {"BANK_DIR": bank})
            self.assertEqual(
                run.call_args_list[-1].args[0],
                ["setfacl", "-R", "-m", "u:xsup:r-X", "--", str(bank.resolve())],
            )
            self.assertEqual(
                run.call_args_list[0].args[0][:5],
                ["setfacl", "-m", "u:xsup:--x", "--", str(bank.resolve().parent)],
            )
            self.assertTrue(all(call.kwargs == {"check": True} for call in run.call_args_list))

    def test_missing_tool_or_manifest_stops_without_permission_changes(self):
        """A failed access setup cannot announce that the bank is ready."""
        with tempfile.TemporaryDirectory() as folder:
            for tool in (None, "/usr/bin/setfacl"):
                with patch("shutil.which", return_value=tool), patch("subprocess.run") as run:
                    with self.assertRaises(RuntimeError):
                        exec(ACCESS_CELL, {"BANK_DIR": Path(folder)})
                    run.assert_not_called()

    def test_access_failure_is_reported(self):
        """An ACL error propagates instead of reporting successful collector access."""
        with tempfile.TemporaryDirectory() as folder:
            bank = Path(folder)
            (bank / "manifest.conf").write_text("format=dlDataCollection-1\n")
            with patch("shutil.which", return_value="/usr/bin/setfacl"), patch(
                "subprocess.run", side_effect=subprocess.CalledProcessError(1, "setfacl")
            ), patch("builtins.print") as output:
                with self.assertRaises(subprocess.CalledProcessError):
                    exec(ACCESS_CELL, {"BANK_DIR": bank})
                output.assert_not_called()

    def test_unrecognized_notebook_is_left_unchanged(self):
        """Do not guess where to insert permissions into an unrelated notebook."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "other.ipynb"
            original = '{"cells": [], "metadata": {}}'
            path.write_text(original)
            with self.assertRaises(ValueError):
                update_notebook(path)
            self.assertEqual(path.read_text(), original)

    def test_dated_paths_preserve_recipe_and_remain_editable(self):
        """Use the local date, retaining root/name, custom raw path and original recipe."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "Generate_random_modevals.ipynb"
            recipe = (
                "from pathlib import Path\n"
                "BANK_DIR = Path('/home/rlandman/NNReconstructor_2026B/banks/20261015/ND1')\n"
                "RAW_DATA_DIR = Path('/data/custom/ND2')\n"
                "S = 0.5\n"
                "layers = [('custom', 70000, -2.5, 50, 1000)]\n"
            )
            original = json.dumps({"cells": [
                {"cell_type": "code", "source": recipe.splitlines(keepends=True)},
                {"cell_type": "code", "source": ["generate_bank(BANK_DIR, layers, calibration)\n"]},
            ]})
            path.write_text(original)
            backup = update_notebook(path, dated_paths=True)
            update_notebook(path, dated_paths=True)
            self.assertEqual(backup.read_text(), original)
            cells = json.loads(path.read_text())["cells"]
            source = "".join(cells[0]["source"])
            self.assertIn("S = 0.5\nlayers = [('custom', 70000, -2.5, 50, 1000)]\n", source)
            namespace = {}
            exec(source, namespace)
            from datetime import date
            self.assertEqual(namespace["BANK_DIR"], Path(
                '/home/rlandman/NNReconstructor_2026B/banks') / date.today().strftime('%Y%m%d') / 'ND1')
            self.assertEqual(namespace["RAW_DATA_DIR"], Path('/data/custom/ND2'))
            self.assertEqual(source.count("BANK_DATE ="), 1)
            self.assertEqual(len(cells), 3)
            # A deliberate full-path override survives later notebook updates.
            cells[0]["source"] = (source + "\nBANK_DIR = Path('/my/explicit/bank')\n").splitlines(keepends=True)
            path.write_text(json.dumps({"cells": cells}))
            update_notebook(path, dated_paths=True)
            self.assertIn("BANK_DIR = Path('/my/explicit/bank')", path.read_text())

    def test_dated_paths_reject_unrecognized_layout_without_editing(self):
        """Never guess a root for an arbitrary custom bank path."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "Generate_random_modevals.ipynb"
            original = json.dumps({"cells": [
                {"cell_type": "code", "source": ["BANK_DIR = Path('/my/custom/bank')\n"]},
                {"cell_type": "code", "source": ["generate_bank(BANK_DIR, layers, calibration)\n"]},
            ]})
            path.write_text(original)
            with self.assertRaises(ValueError):
                update_notebook(path, dated_paths=True)
            self.assertEqual(path.read_text(), original)


if __name__ == "__main__":
    unittest.main()
