"""Regression checks for schema and constructor parsing in the log generator."""

import pathlib
import tempfile
import unittest
from unittest import mock

import generateTemplatedCatch2Tests as generator


class GeneratorTests(unittest.TestCase):
    """Exercise schema layouts that previously produced invalid C++ tests."""

    def test_root_table_excludes_struct_fields(self):
        """Honor root_type, inline comments, and structs declared before tables."""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "tests").mkdir()
            schemas = root / "types/schemas"
            schemas.mkdir(parents=True)
            (schemas / "example.fbs").write_text("""
                namespace Example.logs;
                struct Point { x:float; y:float; }
                table Payload {
                    frame:ulong; // table comment must not hide this field
                    points:[Point];
                    old:int (deprecated);
                }
                root_type Payload;
            """)
            with mock.patch.object(generator, "__file__", str(root / "tests/generator.py")):
                name, fields, structs = generator.getSchemaFieldInfo("example")
        self.assertEqual(name, "Payload")
        self.assertEqual(fields, (("frame", "ulong"), ("points", "[Point]")))
        self.assertEqual(structs["Point"]["type"], "Example::logs::Point")
        self.assertEqual(structs["Point"]["members"], [("x", "float"), ("y", "float")])

    def test_pupilfit_constructor_and_timestamp(self):
        """An out-of-class definition is not another constructor overload."""
        info = generator.makeTestInfoDict(str(pathlib.Path(__file__).parent / "../types/telem_pupilfit.hpp"), {})
        self.assertEqual(len(info["messageTypes"]), 1)
        fields = {field["name"]: field for field in info["messageTypes"][0]}
        self.assertEqual(list(fields), ["frame", "acquired", "reference", "numPupils", "valid", "pupils"])
        self.assertEqual(fields["acquired"]["schemaMembers"], [("tv_sec", "acq_sec"), ("tv_nsec", "acq_nsec")])
        self.assertEqual(fields["reference"]["schemaName"], "reference")
        self.assertEqual(fields["numPupils"]["schemaName"], "num_pupils")
        self.assertEqual(fields["pupils"]["type"], "std::vector<MagAOX::logger::PupilFitResult>")

    def test_setpoints_are_structs_not_integers(self):
        """A struct name containing 'int' must not become a random integer."""
        info = generator.makeTestInfoDict(str(pathlib.Path(__file__).parent / "../types/telem_pupilfit_state.hpp"), {})
        self.assertEqual(len(info["messageTypes"]), 1)
        fields = info["messageTypes"][0]
        self.assertEqual(fields[0]["type"], "std::vector<MagAOX::logger::PupilFitSetpoint>")
        self.assertEqual(generator.getTestValFromType("PupilFitSetpoint"), "{}")
        self.assertEqual(fields[1]["schemaName"], "requested_mode")

    def test_nested_table_mapping(self):
        """Existing camera parameters still use nested-table accessors."""
        info = generator.makeTestInfoDict(str(pathlib.Path(__file__).parent / "../types/telem_stdcam.hpp"), {})
        accessors = [field.get("schemaName") for field in info["messageTypes"][0]]
        self.assertIn("roi()->xcen", accessors)
        self.assertIn("tempCtrl()->statusStr", accessors)
        self.assertIn("shutter()->state", accessors)

    def test_multiline_parameter_comment(self):
        """Advance through all lines of a parameter's documentation block."""
        lines = ["messageT ( uint32_t frame, /* input frame\n", " additional description\n",
                 " */\n", " bool valid );\n"]
        fields = generator.getMessageFieldInfo([0], lines, (("frame", "uint"), ("valid", "bool")), {}, "example")
        self.assertEqual([field["name"] for field in fields[0]], ["frame", "valid"])


if __name__ == "__main__":
    unittest.main()
