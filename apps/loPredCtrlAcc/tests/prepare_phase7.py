r"""\file prepare_phase7.py
\brief Reuse the pinned-source extractor for the committed Phase 7 numerical oracle.
"""
import prepare_phase6

if __name__ == "__main__":
    prepare_phase6.REFERENCE = "6ae20f859ce308d1d5734154a7c3e5c280a7efb4"
    prepare_phase6.main("Phase 7")
