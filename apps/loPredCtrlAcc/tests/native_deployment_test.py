r"""\file native_deployment_test.py
\brief Check native build/configuration and xctrl contracts without installation, INDI, services or hardware.
\ingroup loPredCtrlAcc_unit_test
"""
import ast
import configparser
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


APP = Path(__file__).resolve().parents[1]
ROOT = APP.parents[1]


class NativeDeploymentTest(unittest.TestCase):
    """Verify real deployment inputs using private fixtures and read-only Make dry runs."""

    def test_offline_dependency_targets_use_one_absolute_build_root(self):
        """Relative and absolute overrides must identify the same .o/.d targets after a header/ABI change."""
        with tempfile.TemporaryDirectory(prefix="ddspc-build-root-") as temporary:
            wrapper = Path(temporary) / "Makefile"
            wrapper.write_text(f"include {APP / 'tests/Makefile.offline'}\nprint-root:\n\t@echo $(BUILD_DIR)\n")
            command = ["make", "-s", "-f", str(wrapper), "BUILD_DIR=relative-build", "print-root"]
            result = subprocess.check_output(command, cwd=temporary, text=True).strip()
            self.assertEqual(result, str(Path(temporary).resolve() / "relative-build"))
            self.assertFalse((Path(temporary) / "relative-build").exists())

    def test_default_config_is_continuous_and_preserves_calibration(self):
        """Installed defaults acquire all inputs without a journal directory, output or automatic learning."""
        config, original = configparser.ConfigParser(), configparser.ConfigParser()
        # MagAO-X supports unsectioned application keys; ConfigParser needs a synthetic section for those keys.
        config.read_string("[application]\n" + (APP / "loPredCtrlAcc.conf").read_text())
        original.read(APP / "examples/loPredCtrlAcc-adc-shadow.conf.example")
        for key in ("profile", "zero0", "zero1", "volts_per_count0", "volts_per_count1",
                    "sensitivity0", "sensitivity1", "polarity0", "polarity1"):
            self.assertEqual(config["adc"][key], original["adc"][key])
        self.assertTrue(config.getboolean("adc", "enabled"))
        self.assertEqual(config["shmimMonitor"]["shmimName"], "aol1_modevalWFS")
        self.assertEqual(config["dm"]["channelName"], "dm10disp10_aux")
        self.assertNotIn("predict", config["parameters"])
        self.assertNotIn("framegrabber", [s.lower() for s in config.sections()])

    def test_online_worker_example_adds_no_trust_or_output_override(self):
        """Example enables bounded processing, retaining unknown native clocks and all physical output guards."""
        config = configparser.ConfigParser()
        config.read_string("[application]\n" + (APP / "examples/loPredCtrlAcc-online-shadow.conf.example").read_text())
        self.assertTrue(config.getboolean("online_shadow", "enabled"))
        self.assertTrue(config.getboolean("parameters", "shadow"))
        self.assertFalse(config.getboolean("parameters", "own_shmim"))
        self.assertEqual(config["native_shadow"]["modal_acquisition_clock"], "unknown")
        self.assertLessEqual(config.getint("online_shadow", "budget_us"),
                             config.getint("online_shadow", "period_us") // 2)
        self.assertNotIn("framegrabber", [s.lower() for s in config.sections()])
        self.assertFalse(any("verified" in key for section in config.sections() for key in config[section]))

    def test_make_cannot_disable_output_guard_and_install_preserves_configuration(self):
        """Evaluate the real app Makefile with a fake include; dry-run commands never install or launch anything."""
        with tempfile.TemporaryDirectory(prefix="ddspc-install-contract-") as temporary:
            root = Path(temporary)
            app = root / "apps/loPredCtrlAcc"
            app.mkdir(parents=True)
            (root / "Make").mkdir()
            (app / "Makefile").write_bytes((APP / "Makefile").read_bytes())
            (root / "Make/magAOXApp.mk").write_text(
                'MAGAOX_PREFIX ?= /opt/MagAOX\n'
                'OBJNAME := $(TARGET).o\n'
                'all:\n\t@echo CXXFLAGS=$(CXXFLAGS) OPTIMIZE=$(OPTIMIZE)\n'
                'install: all\n\t@echo BINARY_INSTALL=$(TARGET)\n')
            command = ["make", "-n", "-f", "Makefile", "DDSPC_SHADOW_ONLY=no",
                       "OPTIMIZE=-ffast-math", "CXXFLAGS=-ffast-math"]
            output = subprocess.check_output(command + ["all"], cwd=app, text=True)
            self.assertIn("-DDDSPC_OPERATIONAL_BUILD", output)
            self.assertIn("-DEIGEN_DONT_PARALLELIZE", output)
            self.assertIn("-std=c++20", output)
            self.assertGreater(output.rfind("-fno-fast-math"), output.rfind("-ffast-math"))
            output = subprocess.check_output(command + ["install", f"MAGAOX_PREFIX={root}/stage"], cwd=app, text=True)
            self.assertIn('test -e', output)
            self.assertIn("Keeping existing loPredCtrlAcc.conf", output)
            self.assertIn("BINARY_INSTALL=loPredCtrlAcc", output)
            self.assertNotIn("xctrl", output)
            self.assertNotIn("proclist", output)
            self.assertFalse((root / "stage").exists())

    def test_actual_xctrl_discovery_and_restart_scope(self):
        """Load only pure function ASTs, avoiding xctrl's module-level sudo/tmux/process inspection."""
        tree = ast.parse((ROOT / "scripts/xctrl").read_text())
        names = {"_read_processes", "_identify_indiserver_name", "validate_targets"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"failure": lambda message: None, "sys": sys}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "xctrl_pure_contract", "exec"), namespace)
        with tempfile.TemporaryDirectory(prefix="ddspc-proclist-") as temporary:
            path = Path(temporary) / "proclist_RTC.txt"
            path.write_text((APP / "deployment/proclist.fragment").read_text() + "isRTC xindiserver\n")
            processes = namespace["_read_processes"](path)
            self.assertEqual(processes["loPredCtrlAcc"], "loPredCtrlAcc")
            restart = namespace["validate_targets"](processes, ["loPredCtrlAcc"], "RTC", for_restart=True)
            self.assertEqual(set(restart), {"loPredCtrlAcc", "isRTC"})
            stop = namespace["validate_targets"](processes, ["loPredCtrlAcc"], "RTC")
            self.assertEqual(stop, ["loPredCtrlAcc"])


if __name__ == "__main__":
    unittest.main()
