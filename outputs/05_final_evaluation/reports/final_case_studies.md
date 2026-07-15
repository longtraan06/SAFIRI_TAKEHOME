# Final Case Studies

## S1 highest risk

- Shipment: `SHP-0069`; route: `HO_CHI_MINH-SYDNEY`; stage: `ORIGIN_DEPARTED`.
- Predicted ETA: `2026-05-06T14:06:21Z`; predicted delay: `24.11h`.
- Risk probability: `0.883`; alert at fixed 0.29: `True`.
- Waterfall: Not applicable at S1..
- Rule-based recommendation:
  - Escalate operational follow-up for the predicted material-delay risk.

Actual outcome is shown only after the frozen case-selection rule above: actual final delay `30.64h`.

## S2 highest structured deviation

- Shipment: `SHP-0039`; route: `SHENZHEN-BRISBANE`; stage: `PORT_ARRIVED`.
- Predicted ETA: `2026-07-04T00:19:45Z`; predicted delay: `24.33h`.
- Risk probability: `0.845`; alert at fixed 0.29: `True`.
- Waterfall: planned customs 23.13h + customs deviation 18.03h + planned post-customs 28.00h + post-customs deviation 6.30h.
- Rule-based recommendation:
  - Escalate operational follow-up for the predicted material-delay risk.
  - Prioritize document completion and pre-clearance review.

Actual outcome is shown only after the frozen case-selection rule above: actual final delay `23.80h`.

## S3 lowest risk

- Shipment: `SHP-0131`; route: `SINGAPORE-MELBOURNE`; stage: `CUSTOMS_CLEARED`.
- Predicted ETA: `2026-07-17T11:00:18Z`; predicted delay: `0.01h`.
- Risk probability: `0.104`; alert at fixed 0.29: `False`.
- Waterfall: planned inland 4.32h + inland deviation 0.01h.
- Rule-based recommendation:
  - Continue monitoring the next milestone update.

Actual outcome is shown only after the frozen case-selection rule above: actual final delay `-3.83h`.
