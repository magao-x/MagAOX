# NN / PO4AO baseline

Created 2026-11-02 on `rlandman/nn-po4ao-baseline` at the user's request.

- Repository/PO4AO base: `4ff2634124a5a6fcf428e22ce2807dc31b21b823` (`rlandman/po4ao-pr`).
- Exact `apps/nnReconstructor/` donor: `4353737e331671cd6a9b6aad5d92039c27f26715` (correctness/controls, no speedups).
- Exact `apps/dlDataCollection/` donor: `41f21c44d95b4acc2c6870c7901db6f01f140068` (includes metadata and latest Start-preflight fixes).
- Collection-specific test/build registration merged into the PO4AO base; its existing test group/list entry retained. Collection engineering plans imported from the same donor.
- Accepted requirements copied byte-for-byte from the user's attachment into `agents/plans/2026-10/nn-po4ao-gui-requirements.md` (SHA256 `279d10d5b7292604cfedde1bbde929f80d2eaa1b54810f68211b6795c054f5e6`).

The three app subtrees were compared against their exact donors. Neither the NN speedups nor the original checkout's uncommitted edits were imported. No whole donor branch was merged. This baseline does not include the new trainer/GUI or upstream GUI backports; those belong on the child implementation branch.

Validation: NN app-local CPU suite passed 461 assertions in 27 cases; PO4AO CPU helper suite passed 79 assertions in 14 cases. Collection CPU tests passed 216 assertions in 16 cases. Its legacy INDI dependencies were compiled separately as C++14 for Apple libc++, then linked into the C++20 test with OpenSSL; no source change was needed. Existing INDI warnings about a temporary string and deprecated adapters were observed. These tests do not establish CUDA/TensorRT ABI, timing, installed configuration or instrument behavior. No instrument connection, installation or operation occurred.

Follow the accepted requirements and current user-provided AGENTS.md, including the prohibition on agent execution on instrument hosts. The child implementation branch is `rlandman/nn-po4ao-gui`. Keep this baseline branch fixed after its verification commit.
