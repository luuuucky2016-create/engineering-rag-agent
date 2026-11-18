# Voltage Drop in Cables

## Why it matters
Every cable has resistance, so current flowing through it causes a voltage drop along its length. Excessive drop makes motors run hot, lights dim and equipment misbehave. Long runs and high currents are the usual causes.

## Formulas (resistive approximation)
- Single-phase: Vd = 2 × I × L × R / 1000
- Three-phase (line-to-line): Vd = √3 × I × L × R / 1000

Where Vd is the voltage drop in volts, I is the load current in amperes, L is the one-way cable length in metres, and R is the conductor resistance in ohms per kilometre (Ω/km). The factor 2 in the single-phase formula accounts for the outgoing and return conductors.

Percentage drop = Vd / V_supply × 100.

## Typical copper conductor resistance (DC, 20 °C, IEC 60228 class 2 maximums)
| Size (mm²) | R (Ω/km) |
|---|---|
| 2.5 | 7.41 |
| 4 | 4.61 |
| 6 | 3.08 |
| 10 | 1.83 |
| 16 | 1.15 |
| 25 | 0.727 |

## Common design limits
Commonly used guidelines limit voltage drop to about 3% on a final circuit and about 5% in total from the supply point to the furthest load (for example the NEC informational notes, and IEC 60364-5-52 suggests 3% for lighting and 5% for other uses on a public LV supply). Always check the project specification and the local code that applies.

## Fixing excessive drop
Increase the conductor size, shorten the route, split the load across circuits, or raise the distribution voltage.
