# xInstGraph review — 2026-09-27

## Scope and evidence

The original review examined `apps/xInstGraph` in MagAOX (`6ff2b673`) and
`/home/jrmales/Source/instGraph` (`77c85b6`, then `main`). At that point the
installed `instGraphXML.hpp` matched the library checkout. The findings below
preserve the original evidence; the commit annotations record subsequent fixes.

At review time, the existing `xInstGraph_test` failed after a power update
because its graph fixture had no write-triggering puts or `fsmstate` extra.
The original test also needed `/usr/local/lib` on `LD_LIBRARY_PATH` in that
environment. The expanded app test suite and library tests now pass in the
source checkouts; installation of the later fixes has not been verified here.

## Findings

### F01 — High: output publication depends on incidental graph changes

**Fixed in:** MagAOX `d2903358` (initial snapshot); instGraph `59a8adc` (explicit serialization).

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

**Fixed in:** MagAOX `7a8769f3`; instGraph `59a8adc`.

`instGraphXML::stateChange()`, `valuePut()`, and `valueExtra()` save directly to
the configured output file and ignore `pugi::xml_document::save_file()`'s
result (`instGraph/src/instGraphXML.cpp:833-996`). Put propagation and node
updates can invoke several saves for one INDI message
(`instGraph/src/instIOPut.cpp:163-197`), so a reader can observe a partially
updated graph; a missing/unwritable output directory is not reported to the
app. Add a checked, explicit publication step after a complete update and
replace the destination atomically.

### F03 — High: the output path is not protected from the input path

**Fixed in:** MagAOX `d2903358`.

The app prepends `m_configDir` to `graph.file`, accepts `graph.outputPath`
without checking whether it names the same file, and unconditionally removes
the output path at shutdown (`apps/xInstGraph/xInstGraph.hpp:156-177,506-511`).
If the paths alias, a graph update overwrites the source diagram and shutdown
deletes it. The default output path is the relative `tmp.drawio`
(`instGraph/src/instGraphXML.hpp:187`), which the app can also remove without
having created it. Resolve and validate both paths, require a deliberate
output location, and remove only a file owned by this run.

### F04 — High: unknown node types silently leave graph nodes unmonitored

**Fixed in:** MagAOX `bf716bbb`.

The config loop handles five exact `type` strings but has no error branch for
an unknown value (`apps/xInstGraph/xInstGraph.hpp:205-399`). `xn` stays null,
the section is skipped, and `appStartup()` still enters `READY`
(`apps/xInstGraph/xInstGraph.hpp:440-499`). A typo can therefore yield a graph
that appears operational while a node never receives INDI updates. Reject
unknown types and report the section and value; also validate required graph
nodes against configured handlers.

### F05 — High: invalid FSM target states can turn an unknown state on

**Fixed in:** MagAOX `69655626`.

`fsmNode::loadConfigDerived()` converts `targetStates` without checking the
result (`apps/xInstGraph/xigNodes/fsmNode.hpp:361-367`). `str2Code()` returns
`-999` for an invalid configured state (`libMagAOX/app/stateCodes.cpp:57-127`),
and the FSM handler can also produce `-999` for an unknown received state
(`apps/xInstGraph/xigNodes/fsmNode.hpp:395-418`). The two then match, so an
`active` node can enable puts for an unrecognized state. Reject invalid target
state names at config load and test unknown incoming states.

### F06 — High: `alwaysOn` puts can remain on when a motion stage turns off

**Fixed in:** MagAOX `046c75ae`.

`stdMotionNode::togglePutsOff()` skips every input/output named in `m_alwaysOn`
(`apps/xInstGraph/xigNodes/stdMotionNode.hpp:543-569`). Those puts are enabled
by the multi-put `togglePutsOn()` path (`:431-455`, `:477-503`). After the FSM
leaves `READY`, the skipped puts can still show light/beam flow, although the
documented meaning of `alwaysOn` is “on if any are on”
(`apps/xInstGraph/README.md:38`). Clear them when the stage is off; add a
transition test from an active preset to a non-ready FSM state.

### F07 — High: multi-put nodes assume an opposite-side put exists

**Fixed in:** MagAOX `546a5f19`.

For multiple `presetPutName` values, `togglePutsOn()` dereferences
`m_node->inputs().begin()->second` or `outputs().begin()->second` without an
emptiness check (`apps/xInstGraph/xigNodes/stdMotionNode.hpp:405-421,458-475`).
The surrounding `try` blocks cannot catch an invalid iterator dereference.
Validate the graph topology during config load and guard the runtime path.
Also validate configured put names before accepting the node.

### F08 — Medium: `indiProp` can stay off after an FSM threshold clears

**Fixed in:** MagAOX `16cc099f`.

`indiPropNode` runs the base FSM handler first and only changes its own puts
when the tracked property's comparison changes
(`apps/xInstGraph/xigNodes/indiPropNode.hpp:258-369`). With `fsmAction=threshOff`,
an off-target FSM state turns the puts off; returning to a target FSM state
does not restore them (`apps/xInstGraph/xigNodes/fsmNode.hpp:420-433`). If the
tracked property remains true, later identical messages also do nothing. Keep
the comparison and FSM gate as separate state and recompute the effective put
state whenever either changes.

### F09 — Medium: power states outside `On` are reported as `OFF`

**Fixed in:** MagAOX `32a1b956`.

`pwrOnOffNode::handleSetProperty()` treats every present `state` other than
exactly `"On"` as off (`apps/xInstGraph/xigNodes/pwrOnOffNode.hpp:55-77`). An
unknown/intermediate or differently cased value is therefore published as
`OFF`. `toggleOff()` also sets `m_pwrState` to `1`, the same value as `toggleOn()`
(`:79-101`); that field currently has no reader. Define and test the accepted
power vocabulary, retain an unknown state when appropriate, and correct or
remove the unused field.

### F10 — Medium: node ownership is leaked

**Fixed in:** MagAOX `d6d37c98`.

The app allocates each node with `new` and stores raw pointers in `m_nodes`
(`apps/xInstGraph/xInstGraph.hpp:64-70,205-399`). Its destructor deletes only
the INDI property descriptors (`:125-131`), so nodes leak on normal shutdown
and also when later configuration throws. `xigNode` has no virtual destructor
(`apps/xInstGraph/xigNodes/xigNode.hpp:30-92`), which must be addressed before
deleting derived nodes through base pointers. Use owned node storage with
exception-safe construction.

### F11 — Medium: typed put IDs are parsed with reversed `rfind` arguments

**Fixed in:** instGraph `7c3d0a3`.

In `instGraph/src/instGraphXML.cpp:79-87`, `value.rfind( fc, '.' )` calls the
`rfind(char, position)` overload. It searches for the character represented
by the colon index, rather than for `'.'` before that index. A typed ID such
as `output.power:node:put` then fails the direction-prefix checks at `:92-151`
instead of selecting the requested put type. Add parser tests for typed input
and output IDs and fix the argument order. The provided demos use untyped IDs,
so this was not exercised by the app test.

### F12 — Medium: current test and user documentation miss the runtime contract

**Fixed in:** MagAOX `56f67b57` (examples and callback fan-out), `bf716bbb` (config tests), `7a8769f3` (publication tests).

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

**Fixed in:** MagAOX `f1b35b72`.

The app intentionally follows the header-based application pattern, but
`xInstGraph.hpp` defines all its methods without `inline`
(`apps/xInstGraph/xInstGraph.hpp:120-553`). Several helper headers likewise
contain non-inline free or member definitions, including
`xigNodes/xigNode.hpp:14-22` and `xigNodes/fsmNode.hpp:20-58,290-307`.
Including them from multiple translation units in one target can cause linker
multiple-definition errors. Mark intentionally header-defined functions
`inline` or move definitions to a source file while retaining test access.

## Verification and deployment notes

The MagAOX app test suite passed with 219 assertions in 16 cases after F12;
F13 changed only header linkage, and the app build plus a two-translation-unit
link check passed afterward. The instGraph Catch2 suite passed after F11.
The user installed the fixes and confirmed the bad `[fwfpm]` configuration
failed validation without crashing. The local `stdMotionNode` suite passed
191 assertions in 6 cases after the directional-default follow-up.

F07 validates motion-stage put topology at startup. The installed
`[fwfpm]` config used `presetDir=input` without a `presetPutName`, while the
graph exposed input `in`; the old fixed default was `out`. Follow-up MagAOX
`11638e41` selects `in` when `presetDir=input` and keeps `out` when output is
selected. Explicit names still take precedence and are validated against the
graph. The local installed-style configuration passed `--config.validate`
after this change; it has not yet been installed on exao1.

## Original suggested upgrade order

1. Define the output publication contract and fix F01-F03 in coordination
   with `instGraph`; add an initial snapshot and checked, atomic writes.
2. Reject invalid configuration and graph topology (F04, F05, F07, F11).
3. Correct state transitions and add focused regression tests (F06, F08,
   F09).
4. Resolve ownership, documentation, and header linkage (F10, F12, F13).

`instGraph` changes belong in its own repository; the other changes belong in
MagAOX. No fix in one repository should assume an uninstalled library change
is already deployed.
