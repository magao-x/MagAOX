r"""\file operational_config_test.py
\brief Inspect operational RTC configuration and build profiles without instrument access.
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

    def test_rtc_configuration(self):
        """Both published copies select the same existing WFS, DM and calibrated sensor sources."""
        config = configuration(APP / 'loPredCtrlAcc.conf')
        self.assertEqual((APP / 'loPredCtrlAcc.conf').read_bytes(),
                         (APP / 'examples/operational/loPredCtrlAcc.conf').read_bytes())
        self.assertEqual(config['dm']['channelName'], 'dm01disp05_aux')
        self.assertEqual(config.getint('dm', 'threadPrio'), 20)
        self.assertEqual(config.getfloat('syncopation', 'writeLeadUs'), 125)
        # Assisted slots render at cutoff - renderLeadUs and publish at once.
        self.assertLessEqual(config.getfloat('dm', 'renderLeadUs'), config.getfloat('syncopation', 'writeLeadUs'))
        # dm_slots and acc_ratio are detected from the scheduler and the ACC samples per WFS frame.
        self.assertEqual(config.getint('parameters', 'dm_slots'), 0)
        self.assertEqual(config.getint('parameters', 'acc_ratio'), 0)
        self.assertEqual(config.getfloat('parameters', 'ripple'), 10)
        self.assertFalse(config.getboolean('parameters', 'assisted_diagnostic'))
        self.assertEqual(config['shmimMonitor']['shmimName'], 'aol1_modevalWFS')
        # The always-on main-loop modal output keeps its stream and can never be the WFS input.
        self.assertNotIn('modal_output', config['parameters'])
        self.assertFalse(config.getboolean('parameters', 'own_shmim'))
        self.assertNotEqual(config['framegrabber'].get('shmimName'), config['shmimMonitor']['shmimName'])
        self.assertEqual(config['syncopation']['timingStream'], 'dmtweeter_synco_timing')
        simulator = configuration(ROOT / 'apps/accelSim/acc000.conf')
        self.assertEqual(config['dm']['modeCube'], simulator['dm']['modeCube'])
        self.assertEqual(config['adc']['shmimName'], simulator['framegrabber']['shmimName'])
        self.assertNotIn('output', config)
        reference = configuration(APP / 'tests/operational_parameters.ini')
        for key in ('num_modes', 'history', 'future', 'gain', 'gamma', 'covariance', 'regularization'):
            self.assertEqual(config.getfloat('parameters', key), reference.getfloat('parameters', key))
        self.assertEqual(config.getboolean('parameters', 'qrd'), reference.getboolean('parameters', 'qrd'))

    def test_build_profiles(self):
        """The normal binary is operational and the observer profile retains compile-time inhibition."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = root / 'apps/loPredCtrlAcc'
            app.mkdir(parents=True)
            (root / 'Make').mkdir()
            (app / 'Makefile').write_bytes((APP / 'Makefile').read_bytes())
            (root / 'Make/magAOXApp.mk').write_text('probe:\n\t@echo $(CXXFLAGS)\n')
            for args, flag in (([], '-DDDSPC_OPERATIONAL_BUILD'),
                               (['DDSPC_OPERATIONAL_BUILD=no'], '-DDDSPC_SHADOW_ONLY')):
                self.assertIn(flag, subprocess.check_output(['make', '-s', 'probe', *args], cwd=app, text=True))


if __name__ == '__main__':
    unittest.main()
