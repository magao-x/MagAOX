# Locally recovered installed utility source

These non-executable text snapshots preserve the exact utility contents from the earlier authorized read-only inspection. They were recovered on 2026-11-08 from a local session record, not fetched again from an instrument. No utility was executed for this review.

Provenance: local session `01a1afa1-ca1d-7530-8868-b6d416a5a26a`, saved tool result at line 347 of `rollout-2026-11-06T07-52-58-01a1afa1-ca1d-7530-8868-b6d416a5a26a.jsonl`. Extracted bytes reproduce the SHA-256 values recorded with that inspection:

| Snapshot | Inspected path | SHA-256 |
| --- | --- | --- |
| `howfs_mfilt_switch.source.txt` | `/usr/local/bin/howfs_mfilt_switch` | `bf4c315b992a2b085b014957419c4dcbb814320fed3be1f51a211374e6608524` |
| `cacao-calib-apply.source.txt` | `/usr/local/milk/bin/cacao-calib-apply` | `7dd743af16c8c103b37f001d95e4891c1f676cf0ec0da73999df18e2fca88845` |

`howfs_mfilt_switch` lines 12–29 stop the consumers and restart only acquWFS. `cacao-calib-apply` lines 121–135 enqueue starts, including its one WFS writer start at line 127; lines 147–151 publish calibration markers. The intervening sleeps are not command-completion acknowledgements. Thus observing a marker with a stopped writer is not ruled out by this utility source. It does not demonstrate that the sequence occurred on hardware.

The checked-in post-hook at `scripts/cacao/RTC/tweeter-vispyr-rootdir-scripts/tweeter-post-calib-apply` matches the previously inspected SHA-256 `1a91404f4f39ce996bdd76dc64d50fb592518637315901670c1eee1edbb0996b`; it adds no later WFS writer start. Configuration helpers, controller scheduling and installed ABI behavior have not been revalidated here. Operator acceptance must compare installed utilities/hooks/versions with this evidence before relying on the single-restart assumption. These snapshots are review evidence, not installable replacements.
