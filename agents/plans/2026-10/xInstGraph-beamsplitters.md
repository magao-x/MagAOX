# Task

We need a system to deal with beamsplitters in xInstGraph.  These are subtly different than the simple multi-put selections we have currently in that it's not a simple matter of on/off based on position for each of the outputs.  Some examples:

stagebs:
position | output-wfs | output-sci
---------|------------|------------
out      |   off      |  on
65-35    |   on       |  on
ha-ir    |   on       |  on


fwfpm:
position | output-out | output-refl
---------|------------|-------------
open     |  on        | off
lyotlg   |  on        | off
mirror   |  off       | on

Beamsplitters are essentially by construction stdMotionStage.  But the logic is complicated, at least at the configuration level.  It may require a way to configure each position, but this may not be elegant in the pseudo-TOML configuration language.

Please consider this problem and propose possible solutions. Do not alter this prompt, fill in below Finding.

# Finding

## Scope and baseline

- Reviewed 2026-10-01 at MagAOX commit `911b0db5` on
  `jrmales/instgraph-updates`.
- Inspected the motion handler, app configuration/publication path, adjacent
  instGraph library's put/link propagation, and mxlib's INI/configuration parser.
- Read the local `/opt/MagAOX/config/instgraph.conf`, `magaox.drawio`,
  `stagebs.conf`, and `fwfpm.conf` as configuration examples. These are a local
  snapshot, not confirmation of the configuration installed on exao1.
- The user approved the recommended configuration contract and implementation
  plan on 2026-10-01. Implementation and software verification are complete;
  deployed configuration/graph changes and hardware checks remain pending.

## Current behavior and limitations

### 1. Preset names and put names are currently coupled

[`stdMotionNode::togglePutsOn()`](../../../apps/xInstGraph/xigNodes/stdMotionNode.hpp)
uses two cases:

- A single `presetPutName` accepts any usable preset name and turns all puts on.
- Multiple `presetPutName` entries select the put whose name equals the published
  preset, plus entries in `alwaysOn`. The opposite side must have exactly one put.

A beamsplitter instead needs an explicit relation from a preset to a set of
puts. Several presets can share a route, and one preset can activate several
puts. The supplied examples require these sets:

| Node | Published preset | Active outputs |
| --- | --- | --- |
| stagebs | out | sci |
| stagebs | 65-35 | wfs, sci |
| stagebs | ha-ir | wfs, sci |
| fwfpm | open | out |
| fwfpm | lyotlg | out |
| fwfpm | mirror | refl |

`alwaysOn=sci` cannot express which unrelated preset names should also activate
`wfs`. Similarly, `open` and `lyotlg` cannot both select `out` with the current
name-equality rule. Renaming graph puts or controller presets would couple
instrument display topology to controller naming without representing the
required relation.

The existing `parkedPresetValid()` also checks the name against put names for
multi-put nodes. A mapped route must validate the name against its route table
instead, including when `parkable=true` and the FSM is `POWEROFF`.

### 2. The local graph already contains the required puts

The local `stagebs` node has input `in`, outputs `wfs` and `sci`, and internal
links from `in` to both outputs. Its local config uses `type=static`, with both
outputs enabled regardless of stage position.

The local `fwfpm` node has input `in`, outputs `out` and `refl`, and an internal
link from `in` to `out`. Its local config currently selects `presetDir=input`
with the single input `in`, so it cannot distinguish the two output routes.
A mapping for the requested table should select `presetDir=output`.

There is no `in` to `refl` internal link in this local graph. Assigning the
common input's current state to `refl` once would not make that branch follow
later upstream changes: propagation follows the declared links. Add that link
to the source graph for deployment, and require links covering every controlled
branch in the proposed mapping contract. This conclusion follows from the
library propagation code, rather than a hardware observation.

The local controller files list `65-35,ha-ir` for stagebs, and several FPM
filters including `open,lyotlg`; they do not list stagebs `out` or FPM `mirror`.
The prompt's tables define the requested behavior. Deployment must use the
actual published switch element names, including any additional filters or
aliases that need a route. xInstGraph cannot verify controller preset existence
at configuration load time without receiving the controller's property.

### 3. Turning a linked output off once is insufficient

In the adjacent library's
[`instIOPut::state()`](../../../../instGraph/src/instIOPut.cpp), an enabled
output with an internal input link gets its state from
[`instNode::checkOutputLinks()`](../../../../instGraph/src/instNode.cpp).
A direct `state(off)` can therefore leave it on, and subsequent input/beam
updates can recompute it again. `enabled(false)` prevents activation, but the
setter alone does not change or propagate the existing state.

The current `noAutoOn` option addresses this for specifically listed outputs.
Explicit routing should automatically manage enablement for every controlled
put: disable and set excluded puts off, then enable selected puts and recompute
from the common side. Unavailable nodes must also disable their controlled
puts so an upstream update cannot restore a path while their FSM is inactive.
This behavior should be confined to the new mapping mode.

A selected route is permission for light to propagate. It does not guarantee
that upstream light is available. Preserve instGraph's `waiting` state when the
common input has no incoming beam; do not force selected outputs to `on`.

### 4. The existing config language supports compact rows

mxlib's [`iniFile`](../../../../mxlib/include/app/iniFile.hpp) stores literal
section/keyword pairs, and
[`appConfigurator`](../../../../mxlib/include/app/appConfigurator.hpp) accepts
comma-separated string vectors. A temporary probe using the installed parser
confirmed that `presetRoute.65-35`, `presetRoute.ha-ir`, and an explicit empty
value parse correctly. Dots do not create nested TOML tables.

A handler can discover row keys in its own section through the existing public
`m_unusedConfigs`, then consume each through `configUnused()` so normal
unknown-setting reporting still works. Snapshot the matching keys before
consuming them. This requires no change to the node type factory or mxlib.

Parser limitation: repeated occurrences of one keyword in a file concatenate
values without a separator. The handler cannot reliably detect duplicate rows
after that information is lost. Document one occurrence per route key; rejecting
all repeated file keys would require a separate parser change. Also trim and
validate vector tokens explicitly rather than treating empty tokens as puts.

## Possible solutions

| Approach | Example | Advantages | Costs / limitations |
| --- | --- | --- | --- |
| Per-preset rows in `stdMotion` | `presetRoute.65-35=wfs,sci` | Matches the supplied tables; supports several active puts, aliases, and a known position with no active path; compact existing node sections | Needs route lookup, validation, and propagation control in the motion handler |
| Per-put preset lists | `outputPresets.wfs=65-35,ha-ir` | Easy to see when an individual output is available; naturally supports simultaneous outputs | Repeats preset names across lists; harder to audit an entire position or distinguish a configured blocking position from an unknown one |
| Separate position sections | `[stagebs.route.65-35]` with `outputsOn=wfs,sci` | Room for future inputs, annotations, or wavelength-specific data | More sections; discovery, ownership validation, and typo handling become more involved; apparent nesting remains a naming convention |
| Dedicated `beamSplitter` handler | `type=beamSplitter` with route rows | Can specialize a different device contract later | These devices currently share the stdMotionStage INDI interface, FSM gate, and parking rules; risks duplicating those rules or requires extracting a shared base; new factory/type/docs integration |

An external TOML/JSON routing file or custom encoded matrix would add another
parser/file lifecycle for these small tables. The existing scalar/vector
configuration supports the required relation directly.

## Recommendation: optional per-preset mapping in stdMotion

Keep `type=stdMotion` and enable mapping mode when its section contains at least
one `presetRoute.<published-name>` key. With no route rows, retain existing
configuration and behavior.

Proposed configuration for the task's examples:

```ini
[stagebs]
type=stdMotion
presetDir=output
presetRoute.out=sci
presetRoute.65-35=wfs,sci
presetRoute.ha-ir=wfs,sci

[fwfpm]
type=stdMotion
presetPrefix=filter
presetDir=output
presetRoute.open=out
presetRoute.lyotlg=out
presetRoute.mirror=refl
```

`device` continues to default to the graph node name. `presetDir` continues to
default to output, but is explicit above to make the routing direction clear.
Add `parkable=true` only for controllers publishing `parked.current`; mapping
itself does not subscribe to parking.

### Proposed configuration contract

1. The suffix after `presetRoute.` is the exact, case-sensitive INDI Switch
   element name. Hyphenated names need no quoting. Use names representable as
   literal INI keys; do not introduce quoting or escape syntax in this change.
2. Each value is the set of puts on `presetDir` that can propagate in that
   position. All other puts on that side are blocked. Several rows may name the
   same put, and one row may name several puts.
3. Mapping mode owns all puts on the selected side and the single common put on
   the opposite side. Require internal links from the common input to each
   controlled output, or from each controlled input to the common output for
   input-side selection. Retain the current one-common-put topology constraint;
   arbitrary many-input/many-output crossbars need a separate contract. Reject
   missing links during configuration, identifying the endpoints needed.
4. A row with an empty value, such as `presetRoute.closed=`, represents a known
   position with no active path. It preserves the position label and leaves
   all puts off, including the common put.
5. Reject empty/reserved `none` row names, unknown or wrong-side puts, empty
   list tokens, and repeated put names within a row. Configuration errors should
   identify the node, row, and offending value. Controller-name typos cannot
   generally be rejected before telemetry arrives; unmatched received names
   leave all puts off.
6. Reject explicit `presetPutName`, `alwaysOn`, and `noAutoOn` with route rows.
   The map supplies both selection and enablement; every active put appears in
   each applicable row. This avoids overlapping sources of routing truth and
   avoids redundant declarations of the route domain.
7. For the initial implementation, reject tracking key/element options in
   mapping mode. Existing tracking activation turns every put on and would
   bypass the routing table. Unmapped tracking nodes retain their current
   behavior. Supporting tracking together with mapped presets requires an
   explicit rule for the route while tracking.

Items 3, 6, and 7 deliberately keep the first implementation's contract small;
they are proposed choices for review, not restrictions already in the app.

### Proposed runtime behavior

| FSM / telemetry | Mapped behavior |
| --- | --- |
| READY, exactly one valid mapped preset selected | Apply that row; position label is the published preset |
| POWEROFF, `parkable=true`, affirmative parking, valid mapped preset | Apply the same row; retain the actual POWEROFF FSM label |
| Known mapped preset with an empty row, in either usable state | Keep the preset label; all puts off |
| No selection, `none`, wrong property type, or multiple selected names | Block all puts; position status is `---` |
| Selected name absent from the map | Block all puts; position status is `---`; no fallback to the legacy matching rule |
| OPERATING, HOMING, NOTHOMED, ERROR, disconnected states, or unparked POWEROFF | Block all puts under the existing non-tracking availability rule |

For a valid output-side route, enable the common input and request its on state.
Selected outputs use the existing link calculation from that input's effective
`on`/`waiting` state. Disable excluded outputs before changing the common input,
and explicitly set them off so stale states and downstream beams update.

For an input-side route, use the reciprocal topology: enable the selected inputs,
let their incoming beams determine their effective states, and compute the
common output through its declared links. Keep excluded inputs disabled so
upstream beam changes cannot re-enable them.

Using declared links also ensures these effective states continue to change
when another node updates the upstream beam, without new stage telemetry.
Synthesizing missing links in the handler is an alternative using the existing
library API, but would introduce implicit topology absent from the source graph;
explicit links make that topology reviewable in the same drawing.

Initialize mapped puts disabled/off before startup actions can propagate light
from other handlers; no FSM or preset has been established yet. When the route
is unavailable or empty, disable controlled puts before clearing
their states. When it becomes available again, restore enablement only for that
row and the common put. Tests must exercise both transitions.

Keep graph put labels as their port names (`wfs`, `sci`, `out`, `refl`). Publish
the selected position in the node's `state` extra and keep `fsmstate` unchanged.
Use the app's existing complete-snapshot publication after the callback; no
additional output file or partial XML publication is needed.

## Implementation plan (approved 2026-10-01)

1. Add documented mapping state and configuration parsing to
   `stdMotionNode.hpp`. Discover only this node's route rows, validate graph
   topology/put/link references and conflicting options, and consume recognized
   keys. Initialize the controlled puts disabled/off. Keep the no-map path
   unchanged. No new node type is needed.
2. Add a shared valid-selection/route lookup used by READY and parked POWEROFF
   decisions. Require the Switch property type and exactly one usable selected
   name in mapping mode. Do not apply target positions or numeric inference.
3. Implement mapped put/enablement updates for both directions, preserving
   upstream waiting, internal link propagation, and the actual FSM. Recompute
   after relevant FSM, preset, and enabled parking updates. Distinguish a known
   empty row from an unknown selection when updating position labels.
4. Extend the existing Catch2 node suite with the two exact tables, both
   directions, simultaneous puts, several names sharing one route, empty rows,
   malformed selections, unknown names, configuration failures, and every route
   transition, including missing-link validation. Use real internal links and
   upstream beams to verify blocked puts remain off after subsequent propagation,
   including while unavailable.
5. Extend the app suite with linked draw.io fixtures. Deliver Def/SetProperty
   messages through normal dispatch, assert output colors/labels and retained
   POWEROFF, and cover startup message ordering and parking opt-in. Keep all
   existing unmapped motion and tracking regressions passing.
6. Update the README and this engineering record, complete the changed-file
   documentation pass, add per-case Doxygen groups/real-API references, run
   clang-format, the node/app suites, a serialized xInstGraph build, and a focused
   Doxygen link check. Commit functional work with execution notes, followed by
   documentation as appropriate to repo commit discipline.
7. Prepare deployment config examples using the actual controller preset names.
   Change stagebs from static to mapped stdMotion and fwfpm to output-side
   mapping. Add the missing fwfpm `in` to `refl` link to the source graph.
   Inventory additional FPM filters/aliases before deploying; an omitted name
   intentionally has no active route. Operationally verify both
   split paths, bypass/reflection transitions, upstream light loss/restoration,
   and any supported parked power-off behavior.

The existing instGraph enablement/link APIs appear sufficient; no library or
controller change is proposed. Confirm that conclusion with the propagation
regressions before changing the deployed graph/config.

## Analysis verification and follow-up boundaries

Source inspection and the temporary parser probe supported the initial findings.
No application code or deployed configuration was changed during that analysis,
and no app test suite or hardware check was run until implementation below.

- Binary route availability is the scope. The names `65-35` and `ha-ir` do not
  request intensity, splitting-ratio, polarization, or wavelength modeling.
- Multiple common-side puts and tracking with mappings remain future design
  choices if an instrument node needs them.
- Duplicate-key detection remains a shared INI parser limitation, rather than
  an app guarantee that cannot be implemented from the parsed values alone.
- Property deletion/disconnect cache invalidation remains the existing app-wide
  limitation documented in the parked-routing plan.

## Execution notes (2026-10-01)

- Implemented optional per-preset route rows in stdMotionNode, with complete
  selection masks, exact-name lookup, known empty rows, and the existing
  parking opt-in and true-FSM contract. Nodes without rows keep their prior path.
- Route loading consumes only this node's rows, trims put tokens, rejects
  invalid names/puts and conflicting options, and checks every controlled
  branch's common-side topology and internal link. Link errors name the actual
  draw.io input/output cell IDs. All mapped puts initialize disabled/off.
- Linked propagation regressions verify both directions, both supplied tables,
  every pair of position transitions, and upstream light loss/restoration without
  new stage telemetry. Unmapped names, malformed selections, unavailable FSM
  states, and disabled parking keep paths blocked under upstream updates.
- App regressions use two mapped stages in series and real Def/Set dispatch to
  check complete published colors, unchanged port labels, known empty rows,
  parking opt-in, initial message order, retained POWEROFF, and configuration
  failure before publication when graph links are missing.
- Prepared README examples for both requested tables, including the FPM
  reflected-branch link and the need to inventory actual controller names. The
  local deployed config and graph were not modified. Add the FPM internal link
  and actual route rows when deploying xInstGraph; additional local FPM filters
  and aliases require explicit routing decisions.
- Final focused Catch2 suites passed: stdMotionNode 5546 assertions / 15 cases,
  xInstGraph 1252 / 21 (6798 assertions / 36 cases total). This includes the
  existing unmapped motion, tracking, parking, and app publication regressions.
- Focused Doxygen HTML contains all eight new cases, with verified links from
  the real route parsing, lookup, mask application, and app APIs under test.
  No warnings were emitted for the changed C++ files. The declaration/member
  documentation pass and clang-format checks are complete.
- The final xInstGraph application built successfully with
  `make -C apps/xInstGraph -j1`. `clang-format --dry-run --Werror` and
  `git diff --check` passed. No instGraph library or controller changes were
  needed; rebuild and install only xInstGraph for this feature.
