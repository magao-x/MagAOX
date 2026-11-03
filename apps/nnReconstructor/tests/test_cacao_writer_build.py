## \file test_cacao_writer_build.py
# \brief Compile the real helper through its app Makefile with conflicting header fixtures.
# \ingroup nnReconstructor_unit_test
"""Exercise build isolation without CUDA, EDT, milk, or instrument access."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


class CacaoWriterBuildTest(unittest.TestCase):
    """Keep the app's forced header and isolate the milk-only helper from it."""

    def test_conflicting_headers(self):
        """Reject mixed milk includes and conflicting BOOL typedefs at compile time."""
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="nn-cacao-build-") as directory:
            root = Path(directory)
            app = root / "apps/nnReconstructor"
            for name in (
                "apps/nnReconstructor",
                "Make",
                "libMagAOX",
                "milk/CommandLineInterface",
                "other-milk/CommandLineInterface",
            ):
                (root / name).mkdir(parents=True, exist_ok=True)
            for name in ("Makefile", "cacaoWriter.cpp", "cacaoWriter.hpp"):
                shutil.copyfile(source / name, app / name)
            (root / "libMagAOX/libMagAOX.hpp").write_text("typedef int BOOL;\n")
            (app / "nnReconstructor.cpp").write_text(
                "static_assert(sizeof(BOOL) == sizeof(int));\n"
            )
            (root / "Make/magAOXApp.mk").write_text(
                "SELF_DIR := $(abspath ../../Make)\n"
                "CXXFLAGS := -std=c++20 -Wall -Wextra -Werror -pthread "
                "-I$(SELF_DIR)/../other-milk "
                "-include $(abspath $(SELF_DIR)/../libMagAOX/libMagAOX.hpp)\n"
                "nnReconstructor.o: nnReconstructor.cpp\n"
                "\t$(CXX) $(CXXFLAGS) -c $< -o $@\n"
            )
            (root / "milk/CommandLineInterface/CLIcore.h").write_text(
                "#include <stdint.h>\ntypedef uint_fast8_t BOOL;\n"
                '#include "CommandLineInterface/function_parameters.h"\n'
            )
            shutil.copyfile(
                source / "tests/cacaoWriterTestAbi.hpp",
                root / "milk/CommandLineInterface/function_parameters.h",
            )
            (root / "other-milk/CommandLineInterface/function_parameters.h").write_text(
                '#error "Mixed milk installations"\n'
            )
            result = subprocess.run(
                [
                    "make",
                    "-C",
                    str(app),
                    f"CXX={os.environ.get('CXX', 'clang++')}",
                    f"CACAO_INCLUDES=-I{root}/milk -I{root}/milk/CommandLineInterface",
                    "nnReconstructor.o",
                    "cacaoWriter.o",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((app / "nnReconstructor.o").is_file())
            self.assertTrue((app / "cacaoWriter.o").is_file())


if __name__ == "__main__":
    unittest.main()
