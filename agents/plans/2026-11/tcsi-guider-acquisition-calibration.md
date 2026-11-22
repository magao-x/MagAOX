# tcsInterface guider acquisition calibration

Recorded 2026-11-18; updated 2026-11-21. Status: preliminary calibration from two
full-transfer measurements with `zdSign=+1`. Both points determine the four
angular coefficients exactly; a third measurement is needed to test the model.
The recorded quantities are only synchronized to roughly a couple of minutes,
so the fit uses nominal elevations rather than precisely simultaneous samples.
No application code or operational configuration was changed by this work.

The guider-to-instrument acquisition offsets have changed. The current model
combines a vector rotating with zenith distance and a constant azimuth/elevation
offset. The first measurement produced a conditional solution that retained the
local configuration's constant offsets. The second measurement allows both
components to be fitted. The two-point candidate below supersedes the one-point
candidate for evaluation, subject to consistent measurement references and
command conventions.

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

On 2026-11-18, the workstation file `/opt/MagAOX/config/tcsi.conf` on `mx11`
contained:

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
The first calculation assumes this sign and these constant offsets. The
two-point fit retains the sign but solves for both constant offsets as well.
A local read on 2026-11-21 found the one-point candidate documented below in
`tcsi.conf`, with `focus=0`; this does not establish what the instrument loaded.

## Measurements supplied by the user

The user confirmed that the measurements are **full moves from the guider
position**, rather than additional corrections after Acquire. The raw rows were:

```text
1:10:06 49.90 -40 -10
E00:23:39 75.0902 -38.61 5.947
```

| Date supplied | Hour angle as supplied | Elevation in degrees | Full azimuth move | Full elevation move |
| --- | --- | --- | --- | --- |
| 2026-11-18 | 1:10:06 | 49.90 | -40 | -10 |
| 2026-11-21 | E00:23:39 | 75.0902 | -38.61 | 5.947 |

The first hour angle is interpreted as +1.16833333 hours. Interpreting the second
point's `E` prefix as east of the meridian gives -0.39416667 hours. The raw
notation is preserved for confirmation; hour-angle sign does not affect this
fit. Observation timestamps were not supplied, so the dates above record when
the measurements were supplied rather than asserting observation dates.

On 2026-11-21, the user clarified that the individual quantities may be recorded
at different times, with timing only good to roughly a couple of minutes. Treat
this as approximate relative timing precision, not a stated statistical standard
deviation or strict bound. Preserve the supplied digits for reproducibility;
they do not establish simultaneous coordinate and offset accuracy.

For the second point, the user also reported a prediction of
`(-31.542, 5.947)`. The measured minus reported-predicted difference is
`(-7.068, 0)`.

Offset units and the command used to obtain these moves still need confirmation.
The calculation treats the reported values as signed `aeg` command values.
Image displacements or raw azimuth-coordinate differences may require sign or
projection conversion before fitting.

## Initial conditional calibration from one measurement

Hold `zdSign=+1`, `azoff=0`, and `eloff=-21` fixed. Invert the rotation after
subtracting the constant offsets:

```text
q = 40.10 * pi / 180  (radians)

az0_new =  cos(q)*(A - azoff) + sin(q)*(B - eloff)
el0_new = -sin(q)*(A - azoff) + cos(q)*(B - eloff)

az0_new = -23.51149611
el0_new =  34.17908060
```

The initial conditional angular configuration was:

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

## Preliminary calibration from two measurements

Using the two full-transfer measurements and fixing `zdSign=+1`, solve for
`az0`, `el0`, `azoff`, and `eloff` together. With mathematical pi, the solution is:

```ini
[acqFromGuider]
zdSign=1
az0=-30.960767
el0=19.713872
azoff=-3.619276
eloff=-5.137001
```

These are replacement angular coefficients. Leave the active `focus` value
unchanged. The calculation assumes both measurements used the same guider and
instrument references, the same angular command convention, and a stable
acquisition geometry between measurements.

Each pair below is `(azimuth offset, elevation offset)` in command units.

| Elevation in degrees | Measured | One-point model | Two-point model |
| --- | --- | --- | --- |
| 49.90 | (-40.000, -10.000) | (-40.000, -10.000) | (-40.000, -10.000) |
| 75.0902 | (-38.610, 5.947) | (-31.514, 5.979) | (-38.610, 5.947) |

The recomputed one-point prediction at exactly 75.0902 degrees is
`(-31.51412488, 5.97887980)`. This differs from the user's reported prediction by
approximately `(+0.028, +0.032)`. Preserve the reported prediction as an
independent record. An elevation near 75.032 degrees gives approximately the
reported vector under the one-point model, about 0.058 degrees below the recorded
elevation. This small discrepancy is consistent with the user's stated timing
precision; it is not evidence by itself of a configuration difference. The
source's pi approximation is much smaller. The fit uses the reported elevation
and measured full moves, not either prediction.

The elevations are separated by 25.1902 degrees. The design matrix has rank 4
and a 2-norm condition number of approximately 9.06. Four scalar measurements
determine four coefficients, leaving no residual degrees of freedom: exact
agreement is expected and does not validate the model or `zdSign`. Either sign
can fit two distinct elevations. These two points also do not separate hour-angle
dependence from elevation dependence.

Independent offline calculations agree on the fitted coefficients. Forward
evaluation with the rounded coefficients and the source's approximate pi and
float arithmetic reproduces both measured pairs within approximately 0.00003
command units. This numerical agreement does not establish measurement accuracy.
The coefficients remain the nominal two-point solution; no timing correction
can be inferred from the approximate timing precision alone.

At elevation 60 degrees, the new candidate predicts approximately
`(-40.289, -3.545)`; the previous one-point candidate predicts
`(-37.451, -3.156)`. An independent full-transfer measurement near that elevation
would help test the new model. Additional elevations and repeats are needed to
measure residual scatter and distinguish systematic dependence.

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

Account for timing uncertainty in the elevation input as well as scatter in the
measured offsets. Estimate the local elevation rate from nearby recorded TCS
samples, then use `delta_E ~= (dE/dt)*delta_t` to explore plausible coordinate
shifts. A couple of minutes of timing uncertainty does not imply a fixed
elevation uncertainty across targets or hour angles. For a small elevation
error in degrees, the model sensitivities are:

```text
dA/dE =  zdSign * (B - eloff) * pi/180
dB/dE = -zdSign * (A - azoff) * pi/180
```

At the second point, the one-point model has sensitivities of approximately
`(0.471, 0.550)` command units per degree. An elevation decrease of about
0.058 degrees therefore changes its prediction by approximately `(-0.027, -0.032)`,
consistent with the small recorded prediction discrepancy. Use timing-aware
weights or perturb the input elevations and refit when enough measurements are
available. Any assumed timing distribution must be stated explicitly. Do not
derive a formal coefficient uncertainty from an exact two-point fit alone.

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
  and command convention, and that both points used consistent guider and
  instrument references.
- Record additional full transfers as
  `HA_hours elevation_deg az_offset el_offset`, using the same guider reference
  and final instrument reference. Include TCS zenith distance when available,
  since it is the app's direct input. Retain the raw measurements and
  configuration with each calibration set.
- Obtain a third independent elevation, for example near 60 degrees, to test
  the two-point model, then additional points and repeats to assess residuals
  and repeatability. Retain hour angle to check for dependence beyond elevation.
- Where available, retain timestamps for offsets and coordinates together with
  nearby TCS samples, so the local elevation rate and timing uncertainty can be
  included in later fits. Differences at the few-hundredths level between the
  reported and recomputed second-point predictions are consistent with the
  stated recording precision.
- Have the operator apply and validate the chosen configuration on the
  instrument. Instrument measurements and commands are performed by the user;
  this analysis used local files and offline arithmetic only.
