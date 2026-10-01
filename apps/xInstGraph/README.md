# xInstGraph

xInstGraph publishes a draw.io representation of the instrument graph and updates it from INDI properties.

## Graph input and output

Set the graph options in the application's config file:

    [graph]
    file=magaox.drawio
    outputPath=/path/to/published.drawio
    clobberOutput=false

The input file name is resolved in the application's config directory. The output path is required; a relative output path is resolved from the process's current directory. The output must never refer to the input file, including through a symbolic or hard link.

The app builds the graph in a private staging file and publishes the initial output before it starts receiving INDI updates. By default, startup fails if the output path already exists. Set clobberOutput=true to replace an existing regular output file at startup. Directories and symbolic links are not valid existing destinations. A file left by a previous run therefore needs either an explicit clobber setting or operator cleanup before startup.

For each matching INDI message, the app applies all node changes in memory, writes one complete snapshot to a new staging file, and atomically replaces the output. The clobber setting applies only at startup; later updates replace the file published by this run. A failed update is logged and stops the app. The previous complete snapshot stays in place until shutdown, when the app removes it only if the path still refers to its owned file. An external replacement is neither overwritten nor removed.

## Node configuration

Every node in the draw.io XML file must have a matching configuration section with a `type` key. The type names are case sensitive and accept these values:

| type | Node handler |
| --- | --- |
| fsm | Finite state machine status |
| indiProp | INDI property and element comparison |
| pwrOnOff | Power switch status |
| static | Fixed put states |
| stdMotion | Standard motion stage |

Use `type=static` for a node whose state is fixed rather than driven by INDI. Startup rejects missing or unsupported types, graph nodes without handler sections, and typed sections that do not name a graph node. The error names the offending section or graph node. Unrelated application sections without `type` are allowed.

If loading stops before node handlers read their settings, xInstGraph reports the load error without labeling those unread settings as unrecognized. Run configuration validation again after fixing that error to check for any remaining unknown settings.

For example:

    [stage]
    type=stdMotion
    device=stageDevice

A complete minimal graph can use these draw.io cell IDs. The `out` put on the
motion node matches its default `presetPutName`; a multi-put motion node needs
one put on the opposite side as well.

```xml
<mxfile><diagram><mxGraphModel><root>
  <mxCell id="0"/><mxCell id="1" parent="0"/>
  <mxCell id="node:camera"/><mxCell id="output:camera:out"/>
  <mxCell id="node:shutter"/><mxCell id="output:shutter:out"/>
  <mxCell id="node:power"/><mxCell id="output:power:out"/>
  <mxCell id="node:stage"/><mxCell id="output:stage:out"/>
  <mxCell id="node:source"/><mxCell id="input:source:in"/>
</root></mxGraphModel></diagram></mxfile>
```

With that graph saved as `instrument.drawio` in the config directory, these
sections configure every node. Replace the property keys and output path with
those for the instrument:

```ini
[graph]
file=instrument.drawio
outputPath=/tmp/instgraph-published.drawio
clobberOutput=false

[camera]
type=fsm
device=camera
fsmAction=active
targetStates=READY

[shutter]
type=indiProp
propKey=shutter.position
propEl=state
propVal=OPEN

[power]
type=pwrOnOff
pwrKey=power.channel

[stage]
type=stdMotion
device=stage

[source]
type=static
inputsOn=in
```

The `fsm` handler reads `<device>.fsm` element `state` by default. With
`fsmAction=active`, only a listed `targetStates` value turns its puts on.
`indiProp` compares one element of `propKey` to `propVal`; it can also use
`fsmAction` and `targetStates` as a gate. `pwrOnOff` accepts exact `On` and
`Off`; `Int` and unknown values leave its puts off and display `INT` or `UNK`.
`stdMotion` follows `<device>.presetName` by default. The static handler sets
listed `inputsOn`, `inputsOff`, `outputsOn`, and `outputsOff` puts at startup.

The app test in `tests/xInstGraph_test.cpp` loads all five handler types,
checks initial and updated XML, and checks delivery to two handlers sharing an
INDI property.

### pwrOnOff

| key | type | required | description |
| --- | --- | --- | --- |
| pwrKey | string | yes | INDI key (device.property) of the power switch |

### stdMotion

| key | type | required | default | description |
| --- | --- | --- | --- | --- |
| device | string | no | node name | INDI device name |
| parkable | bool | no | false | Subscribe to parking state and allow parked power-off preset routing |
| presetPrefix | string | no | preset | Preset property prefix, usually preset or filter |
| presetDir | string | no | output | Side selected by the preset: input or output |
| presetPutName | vector<string> | no | `out` for output, `in` for input | Put names selected by the preset |
| alwaysOn | vector<string> | no | empty | Puts that are on when any put is on |
| noAutoOn | vector<string> | no | empty | Outputs not automatically turned on by an internal input link |
| trackingReqKey | string | no | empty | INDI key for the tracking request switch |
| trackingReqElement | string | no | empty | Element of the tracking request property |
| trackerKey | string | no | empty | INDI key for the tracking status switch |
| trackerElement | string | no | empty | Element of the tracking status property |

When `presetPutName` is omitted, it defaults to `out` for
`presetDir=output` and `in` for `presetDir=input`. Explicit put names always
take precedence. The selected name must exist on that side of the graph node;
for example, `[fwfpm]` with `presetDir=input` selects its input `in` without
another setting. The tracking request and status key/element pairs must be
supplied together.

#### Parked stages while powered off

Set `parkable=true` in a stage's `stdMotion` section only when its controller
publishes the Number property `<device>.parked`, element `current`:

```ini
[stageName]
type=stdMotion
parkable=true
```

The option declares parking support; the published `current` value reports
whether the stage is parked now. A nonzero numeric value allows preset routing
while `fsm.state=POWEROFF`; the graph's FSM label still says `POWEROFF`.

`parkable` defaults to false. When omitted or false, the graph does not subscribe
to parking, ignores unsolicited parking updates, and keeps puts off in
`POWEROFF`. This avoids unresolved-property notices for controllers without the
parking interface. With `parkable=true`, a missing parking property still receives
the normal retry/backoff diagnostic.

When upgrading from the version that subscribed to parking automatically, add
`parkable=true` to the existing sections for stages that support parking to retain
their powered-off preset routing.

The parked route uses the published `presetName` or `filterName` selection,
according to `presetPrefix`. It requires exactly one selected name other than
`none`. For a multi-put node, that name must match a configured `presetPutName`.
The normal input/output mapping, `alwaysOn`, and `noAutoOn` rules then apply.
False or malformed parking, an absent or ambiguous selection, or an unmatched
multi-put name leaves all puts off, including `alwaysOn`. An arbitrary numeric
position with no named preset does not identify a graph route.

While parked and powered off, the preset route takes priority over tracking
request and status flags. Tracking resumes under the existing READY/OPERATING
rules when the FSM changes. Parking does not enable routing in other unavailable
states such as HOMING, NOTHOMED, NOTCONNECTED, or ERROR.

FSM, parking, and preset messages arrive separately, so the graph recomputes
from the latest received values after each update. Parking initially defaults
to false. Property deletion and connection loss currently do not invalidate
the graph handlers' cached values; parked routing shares that existing
freshness limitation.
