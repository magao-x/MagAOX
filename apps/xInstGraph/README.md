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

### pwrOnOff

| key | type | required | description |
| --- | --- | --- | --- |
| pwrKey | string | yes | INDI key (device.property) of the power switch |

### stdMotion

| key | type | required | default | description |
| --- | --- | --- | --- | --- |
| device | string | no | node name | INDI device name |
| presetPrefix | string | no | preset | Preset property prefix, usually preset or filter |
| presetDir | string | no | output | Side selected by the preset: input or output |
| presetPutName | vector<string> | no | out | Put names selected by the preset |
| alwaysOn | vector<string> | no | empty | Puts that are on when any put is on |
| noAutoOn | vector<string> | no | empty | Outputs not automatically turned on by an internal input link |
| trackingReqKey | string | no | empty | INDI key for the tracking request switch |
| trackingReqElement | string | no | empty | Element of the tracking request property |
| trackerKey | string | no | empty | INDI key for the tracking status switch |
| trackerElement | string | no | empty | Element of the tracking status property |

The tracking request and status key/element pairs must be supplied together.
