# EDA Summary

## Shipment Counts By Route

| Route | Shipments |
| --- | ---: |
| SHENZHEN-BRISBANE | 76 |
| SHANGHAI-SYDNEY | 64 |
| HO_CHI_MINH-SYDNEY | 62 |
| SINGAPORE-MELBOURNE | 48 |

## Shipment Counts By Carrier

| Carrier | Shipments |
| --- | ---: |
| BlueWave Logistics | 91 |
| Meridian Cargo | 83 |
| Pacific Bridge | 76 |

## Snapshot Counts By Stage

| Stage | Snapshots |
| --- | ---: |
| ORIGIN_DEPARTED | 240 |
| PORT_ARRIVED | 238 |
| CUSTOMS_CLEARED | 240 |

## Planned Duration By Route

| Route | Ocean | Customs | Inland |
| --- | ---: | ---: | ---: |
| HO_CHI_MINH-SYDNEY | 312.0 | 40.0 | 32.0 |
| SHANGHAI-SYDNEY | 360.0 | 36.0 | 30.0 |
| SHENZHEN-BRISBANE | 336.0 | 34.0 | 28.0 |
| SINGAPORE-MELBOURNE | 264.0 | 32.0 | 27.0 |

## Final Delay Distribution And Buckets

Mean=10.382h; median=9.910h; standard deviation=10.486h.

| Bucket | Shipments |
| --- | ---: |
| early_or_on_time | 43 |
| 1_to_6h | 49 |
| 6_to_12h | 50 |
| over_12h | 108 |

## Route Material Delay Rate And Median

| Route | Shipments | Material delay rate | Median final delay hours |
| --- | ---: | ---: | ---: |
| HO_CHI_MINH-SYDNEY | 62 | 0.532258064516129 | 12.721499999999999 |
| SHANGHAI-SYDNEY | 64 | 0.40625 | 10.315000000000001 |
| SHENZHEN-BRISBANE | 76 | 0.47368421052631576 | 9.9155 |
| SINGAPORE-MELBOURNE | 48 | 0.2708333333333333 | 5.5120000000000005 |

## Delay Propagation Relations

| Relation | Correlation |
| --- | ---: |
| port_delay_to_customs_increment | 0.22229968856458923 |
| customs_increment_to_inland_increment | 0.2369335680043613 |
| port_delay_to_final_delay | 0.7584043704862875 |

## Missing And Late Update Summary

| Milestone | Events | Missing rate | Late rate |
| --- | ---: | ---: | ---: |
| CUSTOMS_CLEARED | 250 | 0.04 | 0.12 |
| FINAL_DELIVERED | 250 | 0.044 | 0.176 |
| INLAND_DISPATCHED | 250 | 0.04 | 0.148 |
| ORIGIN_DEPARTED | 250 | 0.04 | 0.164 |
| PORT_ARRIVED | 250 | 0.048 | 0.124 |

## Feature Availability By Prediction Stage

Values are fractions available at S1/S2/S3.

| Stage | Departure delay | Port delay | Customs delay | Truck availability |
| --- | ---: | ---: | ---: | ---: |
| ORIGIN_DEPARTED | 1.0 | 0.0 | 0.0 | 0.0 |
| PORT_ARRIVED | 0.957983193277311 | 1.0 | 0.0 | 0.0 |
| CUSTOMS_CLEARED | 0.9583333333333334 | 0.95 | 1.0 | 1.0 |
