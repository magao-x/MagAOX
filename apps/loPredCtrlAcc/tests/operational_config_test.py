r"""\file operational_config_test.py
\brief Inspect disabled experiment configurations and explicit build profiles without instrument access.
\ingroup loPredCtrlAcc_unit_test
"""
import configparser
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
APP = ROOT / 'apps/loPredCtrlAcc'


def configuration(path):
    """Read MagAO-X unsectioned application options without connecting to a producer."""
    result = configparser.ConfigParser()
    result.read_string('[application]\n' + path.read_text())
    return result


class OperationalConfigTest(unittest.TestCase):
    """Test startup inhibition, independent identities and real Make flag expansion."""

    def test_experiment_pair_is_independent_and_disabled(self):
        """The supplied pair needs real calibration, synchronization, route and bounds before actuation."""
        sim = configuration(ROOT / 'apps/accelSim/acc000.conf')
        ctrl = configuration(APP / 'examples/operational/loPredCtrlAcc.conf')
        self.assertEqual(sim['dm']['channelName'], 'dm01disp10_aux')
        self.assertEqual(ctrl['dm']['channelName'], 'dm10disp10_aux')
        self.assertEqual(ctrl['native_shadow']['dm_stream'], ctrl['dm']['channelName'])
        self.assertEqual(ctrl['native_inputs']['command_domain'], 'corrective_requested')
        self.assertFalse(sim.getboolean('dm', 'enabled'))
        self.assertEqual(sim.getfloat('dm', 'scale'), 0)
        self.assertFalse(sim.getboolean('dm', 'output_authorized'))
        self.assertFalse(ctrl.getboolean('output', 'authorized'))
        self.assertFalse(ctrl.getboolean('output', 'routing_verified'))
        self.assertFalse(ctrl.getboolean('output', 'correction_input_verified'))
        self.assertFalse(ctrl.getboolean('parameters', 'exploration_authorized'))
        self.assertTrue(ctrl.getboolean('parameters', 'shadow'))
        self.assertFalse(ctrl.getboolean('native_shadow', 'record'))
        self.assertEqual(sim.getint('sim', 'fps'), 8000)
        self.assertEqual(sim.getint('synchro', 'referenceFps'), 2000)
        self.assertEqual(sim.getint('framegrabber', 'circBuffLength'), 128)
        self.assertEqual(sim.getint('accel', 'numChannels'), 2)
        self.assertEqual(ctrl.getint('online_shadow', 'standard_slots'), 4)
        self.assertEqual(ctrl.getint('online_shadow', 'training_stride'), 1)
        self.assertEqual((APP / 'examples/operational/acc000.conf').read_bytes(),
                         (ROOT / 'apps/accelSim/acc000.conf').read_bytes())
        for config, section in ((sim, 'dm'), (ctrl, 'output')):
            for key in ('max_abs_um', 'max_step_um', 'ack_timeout_us'):
                self.assertEqual(config.getfloat(section, key), 0)
        for config in (sim, ctrl):
            self.assertEqual(config['syncopation']['timingStream'], '')
        self.assertEqual(ctrl['native_inputs']['calibration'], '')
        self.assertEqual(ctrl.getint('native_inputs', 'sync_epoch'), 0)

    def test_profiles_require_an_explicit_operational_build(self):
        """A caller cannot disable the default guard merely by overriding DDSPC_SHADOW_ONLY."""
        for name in ('accelSim', 'loPredCtrlAcc'):
            with self.subTest(application=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                app = root / 'apps' / name
                app.mkdir(parents=True)
                (root / 'Make').mkdir()
                (app / 'Makefile').write_bytes((ROOT / 'apps' / name / 'Makefile').read_bytes())
                (root / 'Make/magAOXApp.mk').write_text('probe:\n\t@echo $(CXXFLAGS)\n')
                default = subprocess.check_output(['make', '-s', 'probe', 'DDSPC_SHADOW_ONLY=no'], cwd=app, text=True)
                self.assertIn('-DDDSPC_SHADOW_ONLY', default)
                active = subprocess.check_output(['make', '-s', 'probe', 'DDSPC_OPERATIONAL_BUILD=yes'], cwd=app, text=True)
                self.assertIn('-DDDSPC_OPERATIONAL_BUILD', active)
                self.assertNotIn('-DDDSPC_SHADOW_ONLY', active)
                # Even an accidentally conflicting pair of macros is fail-closed in the actual wrapper.
                self.assertIn('!defined( DDSPC_SHADOW_ONLY )',
                              (ROOT / 'libMagAOX/app/syncoOutput.hpp').read_text())


if __name__ == '__main__':
    unittest.main()
