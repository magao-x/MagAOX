# INDI value display

`valueDisplay` shows one or two numeric INDI elements near the upper-left corner
of an rtimv display. It uses the existing `indiDictionary` plugin and requires no
camera plugin. Add a `[valueDisplay]` section to the viewer's configuration,
alongside its existing `[indi]` connection settings.

| Setting | Meaning | Default |
| --- | --- | --- |
| `key1`, `key2` | Full `device.property.element` key for each row | Empty; row disabled |
| `label1`, `label2` | Optional plain-text label | INDI element name |
| `precision1`, `precision2` | Decimal places, integer from 0 to 15 | `3` |

Either row can be used alone. Rows appear in slot order. Without either key,
the plugin unloads normally. Malformed keys or invalid precision reject the
configuration with a startup diagnostic.

For camflowfs-fit, using two decimal places:

```ini
[valueDisplay]
key1=camflowfs-fit.values.x_rms
precision1=2
key2=camflowfs-fit.values.y_rms
precision2=2
```

For values `0.05` and `0.01`, this displays:

```text
x_rms: 0.05
y_rms: 0.01
```

For camwfs-fit, using the default three decimal places:

```ini
[valueDisplay]
key1=camwfs-fit.average.dx
key2=camwfs-fit.average.dy
```

For values `-0.482002258300781` and `-1.89847564697266`, this displays:

```text
dx: -0.482
dy: -1.898
```

Add `label1=Horizontal offset`, for example, to override a label. Labels are
literal text. Numbers retain their signs and always use fixed-point notation;
small values can round to zero, and large numbers or long labels can be clipped.

Unavailable values leave their labels visible, such as `dx: `. Missing,
malformed, nonfinite, or oversized input is blank without a warning. A valid
value remains displayed between publications. It is cleared when the INDI
provider invalidates the connection, deletes the property/device, or replaces a
definition without that element. Rebuild `indiDictionary` with this change to
include replacement-definition invalidation.

The values use the native global-statistics font and color. Their background
opacity follows rtimv's `fontLuminance` calculation at the value widget's own
image position: transparent over dark pixels and darker over bright pixels.
The background refreshes even when the INDI values have not changed.
They appear below the save/loop heading, and below native global statistics
when those are visible. Resizing or moving the statistics readout updates the
placement. Small windows clip the value display to the remaining space; help
stays above it. Movable box-statistics and arbitrary other plugin overlays do
not participate in this placement policy.

Adaptive background opacity requires a viewer that exposes
`mtxTry_fontLuminance(QTextEdit*, bool)` through Qt's meta-object interface.
Rebuild rtimv/rtimvClient with the companion appearance update, then rebuild this
plugin. An older viewer still loads the plugin and uses a transparent background.

## Build and test

Build with the same Qt and rtimv headers/library used by the viewer:

```sh
make -C gui/rtimv/plugins/valueDisplay
make -C gui/rtimv/plugins/indiDictionary
```

The plugin is included in the root `rtimv_plugins_all` target. Its qmake project
also accepts `RTIMV_INCLUDEPATH` and `RTIMV_LIBDIR` for an uninstalled rtimv build.
Use the normal workstation plugin installation process after verification.

Run the focused offline suite from `gui/rtimv/plugins/tests`:

```sh
qmake value_display_test.pro
make -f makefile.value_display_test
QT_QPA_PLATFORM=offscreen \
  VALUE_DISPLAY_PLUGIN="$PWD/../valueDisplay/librtimv_valueDisplayPlugin.so" \
  ./value_display_test
```

The tests use an in-memory dictionary and fake provider; they do not connect to
an INDI server or operate hardware. `VALUE_DISPLAY_PLUGIN` enables the shared
plugin load/unload check. Set `VALUE_DISPLAY_RENDER_PATH` to a PNG path to save
an offscreen render of the values beneath native statistics.
