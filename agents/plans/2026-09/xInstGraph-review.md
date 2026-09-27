# xInstGraph review — 2026-09-27

## Scope and evidence

Reviewed `apps/xInstGraph` in MagAOX (`6ff2b673`) and the local
`/home/jrmales/Source/instGraph` checkout (`77c85b6`, `main`). The installed
`instGraphXML.hpp` matches that checkout. This is a review only; no app or
library implementation was changed.

The existing `xInstGraph_test` builds. It cannot start without adding
`/usr/local/lib` to `LD_LIBRARY_PATH`. With that path set, it fails at
`apps/xInstGraph/tests/xInstGraph_test.cpp:151`: after a power update, the
expected output file does not exist (4 assertions passed, 1 failed). The
fixture has no puts or `fsmstate` extra, so the library's update methods never
call `save_file`. The remaining findings are from code inspection unless noted.

## Findings

### F01 — High: output publication depends on incidental graph changes

`xInstGraph::loadConfigImpl()` calls `hideLinks()` and `hidePuts()` only after
constructing/configuring nodes (`apps/xInstGraph/xInstGraph.hpp:189-405`). The
library's `hideLinks()` and `hidePuts()` mutate the XML in memory without saving
it (`instGraph/src/instGraphXML.cpp:999-1034`). A node constructor or static
state change may have already saved a file with visible links/puts; if no later
state change occurs, that stale appearance persists. Conversely, a graph with
no effective value/put update may never produce an output file at all, as the
existing test demonstrates. Publish a complete initial snapshot after all
configuration and visibility changes, then test both no-change and changed
graphs.

### F02 — High: graph writes can fail silently and expose intermediate state

`instGraphXML::stateChange()`, `valuePut()`, and `valueExtra()` save directly to
the configured output file and ignore `pugi::xml_document::save_file()`'s
result (`instGraph/src/instGraphXML.cpp:833-996`). Put propagation and node
updates can invoke several saves for one INDI message
(`instGraph/src/instIOPut.cpp:163-197`), so a reader can observe a partially
updated graph; a missing/unwritable output directory is not reported to the
app. Add a checked, explicit publication step after a complete update and
replace the destination atomically.

### F03 — High: the output path is not protected from the input path

The app prepends `m_configDir` to `graph.file`, accepts `graph.outputPath`
without checking whether it names the same file, and unconditionally removes
the output path at shutdown (`apps/xInstGraph/xInstGraph.hpp:156-177,506-511`).
If the paths alias, a graph update overwrites the source diagram and shutdown
deletes it. The default output path is the relative `tmp.drawio`
(`instGraph/src/instGraphXML.hpp:187`), which the app can also remove without
having created it. Resolve and validate both paths, require a deliberate
output location, and remove only a file owned by this run.

### F04 — High: unknown node types silently leave graph nodes unmonitored

The config loop handles five exact `type` strings but has no error branch for
an unknown value (`apps/xInstGraph/xInstGraph.hpp:205-399`). `xn` stays null,
the section is skipped, and `appStartup()` still enters `READY`
(`apps/xInstGraph/xInstGraph.hpp:440-499`). A typo can therefore yield a graph
that appears operational while a node never receives INDI updates. Reject
unknown types and report the section and value; also validate required graph
nodes against configured handlers.

### F05 — High: invalid FSM target states can turn an unknown state on

`fsmNode::loadConfigDerived()` converts `targetStates` without checking the
result (`apps/xInstGraph/xigNodes/fsmNode.hpp:361-367`). `str2Code()` returns
`-999` for an invalid configured state (`libMagAOX/app/stateCodes.cpp:57-127`),
and the FSM handler can also produce `-999` for an unknown received state
(`apps/xInstGraph/xigNodes/fsmNode.hpp:395-418`). The two then match, so an
`active` node can enable puts for an unrecognized state. Reject invalid target
state names at config load and test unknown incoming states.

### F06 — High: `alwaysOn` puts can remain on when a motion stage turns off

`stdMotionNode::togglePutsOff()` skips every input/output named in `m_alwaysOn`
(`apps/xInstGraph/xigNodes/stdMotionNode.hpp:543-569`). Those puts are enabled
by the multi-put `togglePutsOn()` path (`:431-455`, `:477-503`). After the FSM
leaves `READY`, the skipped puts can still show light/beam flow, although the
documented meaning of `alwaysOn` is “on if any are on”
(`apps/xInstGraph/README.md:38`). Clear them when the stage is off; add a
transition test from an active preset to a non-ready FSM state.

### F07 — High: multi-put nodes assume an opposite-side put exists

For multiple `presetPutName` values, `togglePutsOn()` dereferences
`m_node->inputs().begin()->second` or `outputs().begin()->second` without an
emptiness check (`apps/xInstGraph/xigNodes/stdMotionNode.hpp:405-421,458-475`).
The surrounding `try` blocks cannot catch an invalid iterator dereference.
Validate the graph topology during config load and guard the runtime path.
Also validate configured put names before accepting the node.

### F08 — Medium: `indiProp` can stay off after an FSM threshold clears

`indiPropNode` runs the base FSM handler first and only changes its own puts
when the tracked property's comparison changes
(`apps/xInstGraph/xigNodes/indiPropNode.hpp:258-369`). With `fsmAction=threshOff`,
an off-target FSM state turns the puts off; returning to a target FSM state
does not restore them (`apps/xInstGraph/xigNodes/fsmNode.hpp:420-433`). If the
tracked property remains true, later identical messages also do nothing. Keep
the comparison and FSM gate as separate state and recompute the effective put
state whenever either changes.

### F09 — Medium: power states outside `On` are reported as `OFF`

`pwrOnOffNode::handleSetProperty()` treats every present `state` other than
exactly `"On"` as off (`apps/xInstGraph/xigNodes/pwrOnOffNode.hpp:55-77`). An
unknown/intermediate or differently cased value is therefore published as
`OFF`. `toggleOff()` also sets `m_pwrState` to `1`, the same value as `toggleOn()`
(`:79-101`); that field currently has no reader. Define and test the accepted
power vocabulary, retain an unknown state when appropriate, and correct or
remove the unused field.

### F10 — Medium: node ownership is leaked

The app allocates each node with `new` and stores raw pointers in `m_nodes`
(`apps/xInstGraph/xInstGraph.hpp:64-70,205-399`). Its destructor deletes only
the INDI property descriptors (`:125-131`), so nodes leak on normal shutdown
and also when later configuration throws. `xigNode` has no virtual destructor
(`apps/xInstGraph/xigNodes/xigNode.hpp:30-92`), which must be addressed before
deleting derived nodes through base pointers. Use owned node storage with
exception-safe construction.

### F11 — Medium: typed put IDs are parsed with reversed `rfind` arguments

In `instGraph/src/instGraphXML.cpp:79-87`, `value.rfind( fc, '.' )` calls the
`rfind(char, position)` overload. It searches for the character represented
by the colon index, rather than for `'.'` before that index. A typed ID such
as `output.power:node:put` then fails the direction-prefix checks at `:92-151`
instead of selecting the requested put type. Add parser tests for typed input
and output IDs and fix the argument order. The provided demos use untyped IDs,
so this was not exercised by the app test.

### F12 — Medium: current test and user documentation miss the runtime contract

The sole app test expects an output for a graph without any write-triggering
element and currently fails (`apps/xInstGraph/tests/xInstGraph_test.cpp:46-159`).
It does not check output contents, callback fan-out, startup validation, path
safety, or shutdown behavior after failed startup. The README lists node types
as `fsmNode`, `pwrOnOffNode`, and `stdMotionNode`, while the parser requires
`fsm`, `pwrOnOff`, and `stdMotion` (`apps/xInstGraph/README.md:11-17`;
`apps/xInstGraph/xInstGraph.hpp:240,276,312`). It also omits `indiProp` and
`static`. Update the fixture and documentation with executable examples of
each supported node type.

### F13 — Low: header definitions are unsafe across multiple translation units

The app intentionally follows the header-based application pattern, but
`xInstGraph.hpp` defines all its methods without `inline`
(`apps/xInstGraph/xInstGraph.hpp:120-553`). Several helper headers likewise
contain non-inline free or member definitions, including
`xigNodes/xigNode.hpp:14-22` and `xigNodes/fsmNode.hpp:20-58,290-307`.
Including them from multiple translation units in one target can cause linker
multiple-definition errors. Mark intentionally header-defined functions
`inline` or move definitions to a source file while retaining test access.

## Suggested upgrade order

1. Define the output publication contract and fix F01-F03 in coordination
   with `instGraph`; add an initial snapshot and checked, atomic writes.
2. Reject invalid configuration and graph topology (F04, F05, F07, F11).
3. Correct state transitions and add focused regression tests (F06, F08,
   F09).
4. Resolve ownership, documentation, and header linkage (F10, F12, F13).

`instGraph` changes belong in its own repository; the other changes belong in
MagAOX. No fix in one repository should assume an uninstalled library change
is already deployed.
