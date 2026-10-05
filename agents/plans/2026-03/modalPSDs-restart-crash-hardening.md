Prompt: after reviewing agent_context.md, please analyze the app modalPSDs (app/modalPSDs) looking for data races, restart bugs, memory management problems, etc. This app routinely crashes when the inputs resize or are recreated.

Circular Buffer refactor: in order to provide backwards compatibility and ease transition to newer code, I would like to use a template specialization in mx::sigproc::circularBuffer.  So circularBufferBase will have a 4th template parameter of `bool fixed_size = false`.  When false it's the current code.  When true, it's a new version that is fixed size and thread safe, using atomics, etc.  The derived classes then also have this template parameter and can then switch to the fixed size version with minimal impact.  I recognize that some minimal changes to the interface may be necessary.

## Analysis

The `apps/modalPSDs` app has a high-probability crash path during input resize or recreation.

The most important issue is that the app stores raw `float *` pointers from the source shmim into its internal circular buffer:

- `m_ampCircBuff` is `mx::sigproc::circularBufferIndex<realT *, cbIndexT>`
- `processImage()` pushes `curr_src` directly into that buffer
- the PSD worker later dereferences historical entries from that buffer while computing means and PSDs

That design can be valid in principle with `streamCircBuff`, because the old shmim remains valid until the last process releases it. In other words, `modalPSDs` can control when those pointers become invalid. The problem in the current implementation is that this lifetime is not coordinated explicitly. The `shmimMonitor` base class restarts and closes the old image mapping when the source inode, size, or type changes, and `modalPSDs` does not ensure that the PSD thread has fully stopped using any pointers into that old mapping before the close occurs. Once the old mapping is released by this process, previously buffered pointers can become invalid and the PSD thread can segfault.

There is also a second class of bug around restart synchronization:

- `allocate()` tears down and rebuilds FFT work buffers, output shmims, and circular buffer sizing
- `psdThreadExec()` reads and writes those same objects concurrently
- coordination relies on plain `bool` flags (`m_psdRestarting`, `m_psdWaiting`) with no atomics, mutex, or condition variable

Because those flags are unsynchronized, there is no strong guarantee that the PSD thread has stopped touching old buffers before `allocate()` frees and reallocates them.

A third issue is that the circular buffer itself is not synchronized:

- `processImage()` mutates `m_ampCircBuff`
- `psdThreadExec()` concurrently reads `size()`, `maxEntries()`, `latest()`, `mono()`, and `at()`
- `allocate()` can call `maxEntries()` and clear/reset the buffer while the PSD thread is active

The underlying implementation is a plain `std::vector`-backed circular buffer with no internal locking, so this is a true data race even if the source shmim never restarts.

There are additional smaller shared-state races:

- `m_fps`, `m_psdTime`, and `m_psdAvgTime` are updated under `m_indiMutex` but read from the PSD thread without matching synchronization
- `shmimMonitorT::m_restart` is also a plain `bool`, written from callbacks and polled from the shmim monitor thread

Taken together, the app is vulnerable to:

- use-after-unmap / stale-pointer dereference during shmim recreation
- use-after-free on FFT work buffers or output streams during restart
- inconsistent restart handoff caused by racy flags
- steady-state undefined behavior from concurrent access to `m_ampCircBuff`

An important operational requirement further constrains the correct fix:

- pre-restart history must be discarded on resize/recreate
- the resize corresponds to a basis change, so buffered samples from the old stream are no longer scientifically valid

That means the restart path does not need to preserve continuity across the old and new shmims. It only needs to guarantee that the old mapping remains valid until all references to the old basis have been abandoned.

## Likely Root Cause

The crash pattern during input resize or recreation is most likely caused by old-basis source-frame pointers surviving in `m_ampCircBuff` after the old shmim mapping has been closed and replaced.

The restart coordination races likely widen that failure window and may also introduce independent crashes when output streams or FFT work memory are destroyed while the worker thread is still using them.

## Remediation Plan

### 1. Make restart lifetime explicit while keeping zero-copy buffering

Do not make copying the default fix.

Because `streamCircBuff` keeps the old shmim valid until the last accessor releases it, `modalPSDs` should use that contract explicitly:

- continue to use source-frame pointers in steady state
- on restart, stop admitting new work from the old stream
- force the PSD worker to abandon all buffered old-basis history
- clear the circular buffer before the old shmim is released by this process
- only then allow the old mapping to be closed and the new stream to become active

Requirements:

- no pointer into the old shmim may remain reachable after restart acknowledgement
- pre-restart history must always be discarded
- the worker must restart accumulation from the new basis only

This is the most important fix and should be done before tuning allocation details.

### 2. Refactor `mx::sigproc::circularBuffer` for a fixed-size SPSC mode

To preserve low latency in `processImage()`, avoid coarse locking around steady-state circular-buffer access.

Preferred direction:

- extend `mx::sigproc::circularBufferBase` with a fourth template parameter, e.g. `bool fixed_size = false`
- keep `fixed_size = false` behavior identical to the current implementation for backward compatibility
- add a new `fixed_size = true` specialization intended for fixed-capacity, single-producer/single-consumer use
- thread that option through the derived buffer types so callers can switch with minimal code churn

Design intent for the new mode:

- fixed-capacity after `maxEntries()`
- no `push_back()` or other structural vector growth in steady state
- low-latency producer path suitable for `processImage()`
- single-producer/single-consumer safety using atomics and well-defined publication order
- restart/reconfiguration still handled outside the steady-state fast path

This mode should be described as SPSC-safe rather than generically thread-safe. That narrower contract matches `modalPSDs`, keeps the implementation simpler, and reduces the risk of over-designing the mxlib change.

Important interface constraint:

- the current API returns references from `at()` and `operator[]`
- for POD / trivially copyable types, returning by value is acceptable and safer
- if the fixed-size specialization uses atomic publication and snapshot-based reads, reference-returning accessors should be removed
- it is acceptable to change the non-fixed-size versions to return by value as well to keep semantics consistent and avoid exposing mutable references into shared storage

Likely options:

- make `at()` return by value in all modes
- add a snapshot/load-style accessor family for the fixed-size mode and update `modalPSDs` to use it for coherent sequence reads

The key goal is to keep the old API intact for legacy users while allowing the new mode to use the right semantics for correctness.

#### Proposed mxlib API

Target declaration shape:

```cpp
template <typename _derivedT, typename _storedT, typename _indexT, bool fixed_size = false>
class circularBufferBase;

template <typename _storedT, typename _indexT, bool fixed_size = false>
class circularBufferIndex;
```

For the fixed-size SPSC mode, the intended public API is:

```cpp
struct snapshotT
{
    indexT earliest;      ///< Earliest readable slot in the current snapshot.
    indexT latest;        ///< Latest published slot in the current snapshot.
    indexT validEntries;  ///< Number of currently valid entries.
    indexT maxEntries;    ///< Fixed capacity.
    uint64_t mono;        ///< Publication generation for retry validation.
    bool full;            ///< Whether the buffer has reached fixed capacity.
};

snapshotT snapshot() const;

storedT at( indexT refEntry, indexT idx ) const;

bool windowReadable( const snapshotT &sn, indexT refEntry, indexT count ) const;

bool loadSequence( indexT     refEntry,
                   indexT     count,
                   storedT   *dest,
                   snapshotT &sn,
                   int        maxRetries = 3 ) const;

bool loadLatestSequence( indexT     count,
                         storedT   *dest,
                         snapshotT &sn,
                         int        maxRetries = 3 ) const;
```

Intended semantics:

- `snapshot()` describes the current readable region and publication generation.
- `at()` returns by value, not by reference.
- `loadSequence()` copies a logical sequence into caller-owned storage.
- `loadLatestSequence()` copies the newest `count` entries into caller-owned storage.
- `loadSequence()` and `loadLatestSequence()` retry if publication advances during the copy.
- readable windows must exclude the producer's writable slot.

Contract assumptions for the fixed-size mode:

- `storedT` is POD or trivially copyable
- exactly one producer thread calls `nextEntry()`
- exactly one consumer thread performs snapshot/load operations
- the producer must never lap the consumer
- `maxEntries()` or any reset/reconfiguration is externally synchronized and not concurrent with steady-state access

This gives `modalPSDs` a clear fast path:

- `processImage()` publishes a new pointer with minimal overhead
- `psdThreadExec()` snapshots the pointer window it needs into local storage
- PSD calculations then proceed entirely from local pointer arrays rather than repeated live buffer lookups

### 3. Add a real synchronization boundary between allocation and PSD computation

Replace the ad hoc `m_psdRestarting` / `m_psdWaiting` protocol with explicit synchronization.

Recommended shape:

- one mutex protecting PSD worker state and all restart-sensitive resources
- one condition variable for worker sleep/wake and restart handoff
- a restart state machine that makes the worker acknowledge quiescence before any old mapping is released or any free/reallocate work begins

Resources that should be covered by the same synchronization contract:

- circular buffer sizing and contents
- source-stream pointer reachability and restart generation
- `m_tsWork`
- `m_fftWork`
- `m_psdBuffer`
- `m_freqStream`
- `m_rawpsdStream`
- `m_avgpsdStream`
- PSD sizing metadata (`m_nModes`, `m_tsSize`, `m_meanSize`, `m_df`, etc.)

### 4. Remove unsynchronized concurrent access to the circular buffer

Make the producer and consumer interact through one of these patterns:

- the new fixed-size SPSC circular-buffer mode in mxlib
- a snapshot handoff on top of that mode if needed for `modalPSDs`

The current `mx::sigproc::circularBufferIndex` use is not safe as shared mutable state across threads without an external lock, even if the pointer lifetime model is otherwise sound. The preferred fix is to move that synchronization responsibility into a low-overhead fixed-size SPSC mode rather than adding hot-path mutex contention in `modalPSDs`.

### 5. Snapshot runtime parameters before PSD calculations

Avoid reading mutable runtime configuration directly throughout the compute loop.

Instead:

- copy `m_fps`, `m_psdTime`, `m_psdAvgTime`, overlap size, and related sizing values into a restart-protected worker configuration block
- update that configuration only during the same synchronized restart/reallocate path

This prevents INDI callbacks from racing with active PSD calculations.

### 6. Harden restart and shutdown sequencing

Review the full lifecycle for these transitions:

- source stream disappears
- source stream inode changes
- source stream dimensions change
- source stream basis changes and history must be invalidated
- FPS changes enough to force restart
- `psdTime` changes
- application shutdown during allocate wait
- application shutdown during active PSD upload

Success criteria:

- no busy-spin waits on plain shared flags
- shutdown cannot deadlock waiting for a worker state that is never published
- restart always clears old-basis history before releasing the old mapping
- output shmims are destroyed only after the PSD worker is fully stopped or fully quiesced

### 7. Add targeted tests for restart-sensitive logic

The existing unit test coverage is callback-focused and does not exercise the failure mode.

Add tests for:

- restart requests when FPS changes
- reallocation when the number of modes changes
- worker quiescence before resources are replaced
- circular buffer/history cleared before the old shmim is released
- no old-basis samples consumed after a simulated resize/recreate

Where full integration tests are difficult, isolate a smaller worker-state object so the restart handshake can be unit tested without real shmims.

### 8. Add runtime diagnostics while hardening

During implementation and validation, add temporary or permanent logging around:

- restart requested
- worker acknowledged quiescent
- buffers/workspaces destroyed
- buffers/workspaces recreated
- first frame accepted after restart
- PSD uploads resumed

This should make it easier to confirm that crashes correlate with stale data access versus restart sequencing.

## Proposed Implementation Order

1. Design and implement the fixed-size SPSC circular-buffer mode in mxlib with backward-compatible default behavior.
2. Update `modalPSDs` to use that mode and eliminate hot-path structural buffer changes.
3. Introduce explicit worker/restart synchronization and remove the current flag polling.
4. Make restart clear all buffered history before the old shmim is released.
5. Refactor worker configuration so mutable sizing/runtime values are applied only at restart boundaries.
6. Add restart-focused tests and validation logging.
7. Run resize/recreate stress testing against a source stream that changes shape repeatedly.

## Risks / Open Questions

- The local app fix depends on being able to defer release of the old shmim mapping until `modalPSDs` has fully abandoned old-basis pointers.
- The main mxlib design question is how much API compatibility can be preserved for `at()` and `operator[]` in the new fixed-size SPSC mode without forcing incorrect reference semantics.
- `shmimMonitorT::m_restart` remains a plain `bool` in the base class. Converting it to `std::atomic` would be a system-wide perturbation, so that change should be deferred unless local fixes prove insufficient.
- The preferred implementation style is to keep the `modalPSDs` changes primarily in the header, following current MagAO-X conventions, unless a specific code path becomes too difficult to maintain inline.
