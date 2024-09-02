
## Node Configuration

Each node has a section in the config file with the same name as the node.  This name must correspond to the name of the 
node in the `.drawio` XML file.  This looks like:

```
[name]
type=xxxx
```

The *required* `type` keyword is used to specify the type of node.  The possible values are:

- fsmNode
- pwrOnOffNode
- stdMotionNode

Each of these has specific configuration options

### pwrOnOffNode

The power-on/off node shows the status of a power switched device. 

| key            |  type            |  required    | default     |  Description |
|----------------|------------------|--------------|-------------|--------------|
| pwrKey         | string           | Y            |             | The INDI key (device.propery) of the power switch  |

### stdMotionNode

The standard motion node shows the status of an standard motion stage, which includes focus stages, filter wheels, linear actuators, etc. 

| key            |  type            |  required    | default     |  Description |
|----------------|------------------|--------------|-------------|--------------|
| device         | string           | N            | <node name> | The INDI device name, defaults to the node name specified in the `drawio` file  |
| presetPrefix   | string           | N            | preset      | The prefix for preset, usually `preset` or `filter` |
| presetDir      | string           | N            | output      | Allowed values are `input` or `output`. Specifies which put (in or out) which corresponds to the preset. |
| presetPutName  | vector\<string\> | N            | out         | Names for the put which correspond to the preset. If more than one, the preset controls which one is on. |
| trackerKey     | string           | N            |             | If set, this is the INDI key of the tracking property |
| trackerElement | string           | N            |             | If set, this is the element of the tracking property |
