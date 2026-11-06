r"""\file prepare_phase8.py
\brief Extract committed Phase 8 numerical and asynchronous sources without checkout or network access.
"""
import prepare_phase6

if __name__ == "__main__":
    prepare_phase6.REFERENCE = "bbe9ffc0365b8f246371de5af2b8b015a3f01183"
    prepare_phase6.main("Phase 8", ("tests/async_shadow.hpp", "tests/async_shadow.cpp"))
