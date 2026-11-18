# tcsInterface guider acquisition calibration

Recorded 2026-11-18. Status: analysis and conditional calibration from one
measurement; no application code or operational configuration changed.

The guider-to-instrument acquisition offsets have changed. The current model
combines a vector rotating with zenith distance and a constant azimuth/elevation
offset. One measured transfer determines a correction at that elevation, but
cannot distinguish changes in those two components. The proposed values below
retain the local configuration's constant offsets and solve for its rotating
vector.

## Algorithm and configuration

Source reference: `ccc26f4f`,
[tcsInterface.hpp](../../../apps/tcsInterface/tcsInterface.hpp), functions
`acquireFromGuider()`, `getTelData()`, `setupConfig()`, and `loadConfigImpl()`.

For elevation `E` in degrees, define:

```text
Z = 90 - E
q = zdSign * Z * pi / 180
c = cos(q)
s = sin(q)

A = az0*c - el0*s + azoff
B = az0*s + el0*c + eloff
```

`A` and `B` are the signed azimuth/elevation offset commands. The implementation
uses the latest `m_telZd` received from TCS, rather than calculating `Z` from
elevation internally, and uses `3.14159` for pi with float intermediates.

| Key in `[acqFromGuider]` | Meaning |
| --- | --- |
| `az0`, `el0` | Rotating vector components at zero zenith distance, or elevation 90 degrees |
| `azoff`, `eloff` | Constant components added after rotation |
| `zdSign` | Rotation direction, normally +1 or -1 |
| `focus` | Separate focus offset applied after the angular transfer |

At zenith, the total angular command is `(az0 + azoff, el0 + eloff)`.
Hour angle, telescope azimuth, parallactic angle, and actual rotator angle do not
enter the model. The source comment referring to the current rotator angle is
misleading: the calculation uses zenith distance.

An acquisition request sends `aeg A B`, followed by
`zimr (m_acqFocus * m_pyrNudge_F_sign)`. It does not read a guider centroid or
iterate toward the target. Each new request repeats the transfer and focus
command. The `[ACQUIRE] aeg ...` and `[ACQUIRE] zimr ...` log entries record the
commands issued.

The acquisition coefficients are loaded at startup. There are no live setters
for these coefficients in this app; an operator applying a new configuration
must restart `tcsi` through the normal procedure. Focus is calibrated separately.

## Local reference configuration

The workstation file `/opt/MagAOX/config/tcsi.conf` on `mx11` contains:

```ini
[acqFromGuider]
zdSign=1
az0=-26.07
azoff=0
el0=11.22
eloff=-21.0
focus=0
```

This is a local reference, not a verified copy of the active instrument
configuration. Its `zdSign=+1` differs from the source default of `-1`.
The calculation below assumes the active configuration has this sign and these
constant offsets.

## Measurement supplied by the user

The user confirmed that the measurements are **full moves from the guider
position**, rather than additional corrections after Acquire. The raw row was:

```text
1:10:06 49.90 -40 -10
```

| Hour angle | Elevation in degrees | Full azimuth move | Full elevation move |
| --- | --- | --- | --- |
| +1:10:06 | 49.90 | -40 | -10 |

Hour angle is interpreted as +1.16833333 hours and is retained for later
diagnostics. The observation date/time was not supplied; the date at the top is
the documentation date.

Offset units and the command used to obtain these moves still need confirmation.
The calculation treats the reported values as signed `aeg` command values.
Image displacements or raw azimuth-coordinate differences may require sign or
projection conversion before fitting.

## Conditional calibration from one measurement

Hold `zdSign=+1`, `azoff=0`, and `eloff=-21` fixed. Invert the rotation after
subtracting the constant offsets:

```text
q = 40.10 * pi / 180  (radians)

az0_new =  cos(q)*(A - azoff) + sin(q)*(B - eloff)
el0_new = -sin(q)*(A - azoff) + cos(q)*(B - eloff)

az0_new = -23.51149611
el0_new =  34.17908060
```

The conditional angular configuration is:

```ini
[acqFromGuider]
zdSign=1
az0=-23.511496
el0=34.179081
azoff=0
eloff=-21.0
```

Leave the active `focus` value unchanged. These are replacement coefficients,
not increments to add to the existing coefficients.

At elevation 49.90 degrees, the local reference configuration predicts
approximately `(-27.168568, -29.209885)`. The proposed coefficients reproduce
`(-40, -10)` to the precision relevant to this measurement. The inverse and
forward calculations were checked offline using mathematical pi; the source's
pi approximation and float intermediates introduce a small rounding difference.

This solution assumes the rotating component changed while the constant
components stayed fixed. Holding the rotating vector fixed and solving for
`azoff/eloff` would also match this observation. A single point cannot select
between these explanations or validate predictions at other elevations.

## Calibration with additional measurements

For fixed `zdSign`, stack two rows per observation and solve jointly by linear
least squares, preferably using QR or SVD:

```text
p = [az0, el0, azoff, eloff]^T

[ cos(q_i)  -sin(q_i)  1  0 ] p = [ A_i ]
[ sin(q_i)   cos(q_i)  0  1 ]     [ B_i ]
```

For the user's full-transfer measurements, the fitted `p` replaces the old
values. If future data instead record additional signed corrections after
Acquire, fit coefficient increments and add them to the configuration used for
those acquisitions, keeping `zdSign` fixed.

Two distinct elevations determine the four coefficients algebraically, but
provide no redundancy. Closely spaced elevations poorly separate rotating and
constant components. Prefer several well-separated elevations across the normal
observing range, repeats to measure scatter, and independent validation points.
Use measurement uncertainties as weights when available.

For a simpler update, compare full-transfer measurements with the existing
model. If the residual vectors are constant across elevation, update only
`azoff/eloff`. If the residual vectors become constant after inverse rotation,
update only `az0/el0`.

Retain hour angle as a diagnostic and compare rising and setting observations
at similar elevations. Repeatable differences at the same elevation cannot be
represented by this configuration. A single tracked star couples hour angle and
elevation; observations of multiple stars help distinguish those dependencies.
Keep `zdSign` fixed initially. If orientation may have changed, compare fits for
both signs using at least three distinct elevations and independent validation;
either sign can fit two distinct elevations exactly.

## Follow up

- Confirm the active `[acqFromGuider]` configuration and the measurement's units
  and command convention.
- Record additional full transfers as
  `HA_hours elevation_deg az_offset el_offset`, using the same guider reference
  and final instrument reference. Include TCS zenith distance when available,
  since it is the app's direct input. Retain the raw measurements and
  configuration with each calibration set.
- Obtain at least one substantially separated elevation to solve all four
  coefficients, then additional points to assess residuals and repeatability.
- Have the operator apply and validate the chosen configuration on the
  instrument. Instrument measurements and commands are performed by the user;
  this analysis used local files and offline arithmetic only.
