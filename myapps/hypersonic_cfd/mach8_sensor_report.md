# mach8_sensor comparison report

Mesh source: `exasim`
Reference: `../hdg_navierstokes/wall_data.csv`

| Stage | Overrides | PTC | Newton updates | Final residual | Damped steps | Minimum alpha |
|---:|---|:---:|---:|---:|---:|---:|
| 1 | `base` | on | 53 | 1.908342999797726e-07 | 12 | 0.03125 |
| 2 | `av.mode=tanh av.lambda=0.04 av.c=30.0` | on | 6 | 1.077094168474254e-11 | 0 | 1 |
| 3 | `av.mode=tanh av.lambda=0.025 av.c=30.0` | on | 6 | 1.379301555467384e-10 | 1 | 0.5 |
| 4 | `av.mode=tanh av.lambda=0.025 av.c=5.0` | on | 8 | 3.262155508151847e-09 | 3 | 0.125 |

| Quantity | Result | Gate |
|---|---:|---:|
| wall Cp maximum relative difference | 1.178024259247577e-05 | <= 0.01 |
| wall heat-flux maximum relative difference | 2.41384076177733e-05 | <= 0.02 |
| wall-coordinate maximum absolute difference | 5.551115123125783e-17 | geometry diagnostic |
| computed shock standoff | 0.002002002002001957 | report |
| reference shock standoff | 0.002002002002001957 | report |
| shock-standoff relative difference | 0 | <= 0.01 |
