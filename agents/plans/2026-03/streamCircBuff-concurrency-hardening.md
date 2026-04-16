Prompt: streamCircBuff is intended to preserve frame order. Let's make a new design doc in agents/plans to tackle this.

## Analysis

The `apps/streamCircBuff` app currently bridges two threaded subsystems:

- `dev::shmimMonitor`, which watches the source shmim and calls `processImage()` on each new source frame
- `dev::frameGrabber`, which runs its own thread and publishes frames into the output circular-buffer shmim

Today, the handoff between those two subsystems is:

- a single raw pointer member, `m_currSrc`
- a counting semaphore, `m_smSemaphore`

The current flow is:

1. `processImage()` stores `curr_src` into `m_currSrc`
2. `processImage()` posts `m_smSemaphore`
3. the framegrabber thread wakes in `acquireAndCheckValid()`
4. `loadImageIntoStream()` copies from `m_currSrc` into the output shmim

This design is not sufficient if frame order must be preserved.

### Observed Concurrency and Lifetime Problems

#### 1. Single-pointer handoff cannot preserve ordered delivery

Only one source pointer is stored at a time.

If multiple source frames arrive before the framegrabber thread consumes them:

- the latest `processImage()` overwrites `m_currSrc`
- the semaphore count still increases once per frame
- the framegrabber thread may wake multiple times and copy the same newest frame more than once
- one or more earlier frames are dropped

So the current design can duplicate frames and lose ordering under load.

#### 2. Raw source-pointer lifetime is not coordinated with restart

`m_currSrc` points into the source shmim mapping owned by `shmimMonitor`.

When the source shmim is recreated or resized:

- `shmimMonitor` detects the change
- restart proceeds
- the old input shmim may be closed
- the framegrabber thread may still be holding or about to use `m_currSrc`

There is no explicit acknowledgement that the framegrabber side has finished with the old source pointer before the old mapping is released.

This creates a stale-pointer / use-after-unmap crash path.

#### 3. Copy parameters are not snapshotted with the source frame

`loadImageIntoStream()` uses:

- `m_currSrc`
- `pixget`
- `shmimMonitorT::m_width`
- `shmimMonitorT::m_height`

Those values are not published as one coherent frame descriptor.

A restart can change:

- source pointer generation
- source data type
- source dimensions

between the producer-side publication and the consumer-side copy.

That means even when the pointer itself is still valid, the consumer can copy with mismatched metadata.

#### 4. Existing restart/reconfig boundary is not enough

`allocate()` sets `m_reconfig = true`, which causes the framegrabber thread to leave its main loop and rebuild the output shmim.

But this does not protect the producer/consumer handoff itself:

- there is no flush or invalidation of pending source frames
- there is no guarantee that the framegrabber has stopped using the old source generation before restart proceeds
- there is no explicit drain or discard policy for queued old frames

#### 5. Minor cleanup issue

`m_smSemaphore` is initialized but not explicitly destroyed in `appShutdown()`.

This is not the central bug, but it should be cleaned up while touching this area.

## Design Requirements

The fixed design should satisfy all of the following:

- preserve source frame order
- never duplicate a frame due to producer/consumer handoff races
- never consume a frame from an old shmim generation after restart
- ensure source pointer lifetime remains valid until the consumer has finished with the frame
- keep restart/recreate behavior explicit and testable
- keep performance reasonable for the expected frame rates

Operationally, it is acceptable to:

- backpressure the source path if the output side falls behind, or
- drop frames according to an explicit policy

but it is not acceptable to silently reorder frames or duplicate the newest frame while pretending to preserve order.

## Likely Root Cause

The core architectural problem is that `streamCircBuff` is trying to hand off an ordered stream across threads using:

- one mutable pointer
- one counting semaphore

That combination does not encode frame identity or generation.

It only signals "some frame arrived," not "this specific frame arrived and is still valid."

## Remediation Plan

### 1. Replace the single-pointer handoff with an explicit ordered frame queue

The producer side (`processImage()`) should publish a frame descriptor, not just a raw pointer.

Minimum information per queued frame:

- source pointer
- width
- height
- data type / copy function identity
- source generation id
- optional frame timestamp / sequence info if helpful

The queue must preserve FIFO ordering.

Acceptable implementation options:

- fixed-size SPSC ring of frame descriptors
- bounded `std::deque` protected by a mutex/condition variable

Because ordered correctness is more important here than ultra-low-latency publication, a mutex/condvar queue may be acceptable if contention is low. If performance pressure appears later, the queue can be moved to an SPSC ring design.

### 2. Make source-generation lifetime explicit

The queue must carry a generation id associated with the currently open input shmim.

On any source recreate/resize/reopen:

- increment the source generation
- stop accepting or consuming old-generation work
- drain or discard any queued old-generation frames
- ensure the framegrabber side acknowledges quiescence before the old shmim mapping is released

This is the same principle used in the `modalPSDs` hardening:

- restart must define a clear boundary
- no stale work may cross that boundary

### 3. Publish coherent frame descriptors

Do not let the consumer reconstruct copy parameters from mutable global members.

Instead, the producer should publish a coherent descriptor containing:

- source pointer
- dimensions
- type / type size / pixel accessor identity
- any timestamps needed for output metadata

Then the consumer copies using only the queued descriptor.

This avoids mismatches where:

- pointer comes from one source generation
- dimensions or datatype come from another

### 4. Define overload behavior explicitly

If the output side falls behind, the app must use an explicit policy.

Recommended choices to decide between:

- blocking producer until space is available
- bounded queue with explicit oldest-drop or newest-drop policy
- hard error / restart if the queue overflows

Because the app is intended to preserve frame order, "overwrite the latest descriptor and keep incrementing a semaphore" is not acceptable.

The preferred policy is likely:

- bounded FIFO queue
- producer blocks or briefly waits when full
- restart clears the queue

That keeps semantics clear and avoids silent data corruption.

### 5. Add a real restart handoff between shmimMonitor and frameGrabber sides

Introduce explicit synchronization for restart-sensitive state:

- queue contents
- current source generation
- queued descriptor validity
- any in-flight frame being copied

Recommended shape:

- one mutex for queue and generation state
- one condition variable for "frame available"
- one condition variable or flag for restart/quiescence acknowledgement

This should ensure:

- restart can wait until no old-generation frame is being copied
- queued old-generation frames are discarded before source reopen completes
- frame order is preserved within a generation

### 6. Snapshot output copy state cleanly

The framegrabber thread should copy from a local descriptor obtained from the queue.

`loadImageIntoStream()` should operate on that descriptor, not on shared mutable globals like:

- `m_currSrc`
- `shmimMonitorT::m_width`
- `shmimMonitorT::m_height`

That implies introducing a small app-local "pending frame" structure owned by `streamCircBuff`.

### 7. Clean up semaphore / signaling design

Once a queue plus condition variable is in place, `m_smSemaphore` may no longer be needed.

If retained for some reason, its lifecycle must be fully managed:

- initialize once
- destroy in shutdown
- avoid leaving producer/consumer state split across both semaphore counts and queue state

The cleaner design is likely:

- remove `m_smSemaphore`
- use queue + condition variable only

### 8. Add tests for ordered and restart-sensitive behavior

Current `streamCircBuff` tests are placeholder-only.

Add tests for:

- single frame handoff
- multiple queued frames preserving FIFO order
- no duplicate-consume behavior when producer gets ahead
- restart clearing queued old-generation frames
- consumer rejecting stale-generation frames
- coherent descriptor copy sizing across a simulated resize/recreate

Where full integration is hard, isolate a small queue/generation state object and test that directly.

## Proposed Internal Structure

A likely app-local structure:

```cpp
struct pendingFrameT
{
    char    *src{ nullptr };
    size_t   width{ 0 };
    size_t   height{ 0 };
    uint8_t  dataType{ 0 };
    size_t   typeSize{ 0 };
    uint64_t generation{ 0 };
    timespec acquisitionTime{ {0, 0} };
};
```

Possible associated state:

```cpp
std::mutex m_queueMutex;
std::condition_variable m_queueCond;
std::condition_variable m_restartCond;

std::deque<pendingFrameT> m_pendingFrames;
size_t m_maxPendingFrames{ ... };

uint64_t m_sourceGeneration{ 0 };
bool m_restartPending{ false };
bool m_consumerBusy{ false };
```

The consumer then:

- waits for a frame descriptor
- pops the oldest descriptor
- copies using only descriptor-local metadata
- publishes the output frame

The restart path then:

- marks restart pending
- waits for consumer not busy on old generation
- clears queued frames
- increments generation
- resumes

## Success Criteria

- repeated source restarts/resizes do not crash
- no duplicated latest frame under producer burst conditions
- queued frames are consumed in FIFO order
- old-generation frames are never copied after restart
- output stream dimensions and copied pixel interpretation always match the queued source descriptor

## Follow-Up Notes

- A queue-based handoff is likely the safest first implementation.
- If needed later, the queue can be optimized into a fixed-size SPSC descriptor ring.
- The test surface here is better than `modalPSDs` because the core bug is in app-local handoff logic, not in FFT work.
