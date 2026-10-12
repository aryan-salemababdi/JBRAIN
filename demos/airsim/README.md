# J-BRAIN AirSim Navigation Demos

This directory documents three autonomous navigation experiments using J-BRAIN in the Microsoft AirSim simulation environment.

The demos investigate goal-directed navigation, obstacle-rich environments, and the relationship between model-proposed actions and safety-filtered actions.

**These experiments are simulation demonstrations, not evidence of real-world flight readiness or safety.**

## Overview

| Demo | Focus | Recorded outcome |
|---|---|---|
| Demo 01 | Basic autonomous navigation | Reached within 0.954 m of the goal |
| Demo 02 | Navigation in an obstacle-rich environment | Reached within 0.909 m of the goal |
| Demo 03 | Policy attribution and safety intervention | Reached within 0.886 m of the goal |

All three demonstrations used **Policy A**, checkpoint SHA-256:

`f496e18d96beb55eda6954516aa88b83947169e8168e85eb513a05e28517668c`

The Safety Layer was enabled in every run.

## Demo 01 — Basic Navigation

**Scenario:** `s2026_e00104`

This experiment demonstrates goal-directed navigation in an AirSim environment containing eight spawned obstacles.

| Metric | Recorded value |
|---|---:|
| Control steps | 137 |
| Simulated duration | 25.98 s |
| Initial goal distance | Not reported here |
| Final goal distance | 0.954 m |
| Path length | 44.70 m |
| Path efficiency | 0.87 |
| Safety overrides | 3 (2.2%) |
| Recorded collisions | 0 |
| Minimum reported LiDAR clearance | 1.496 m |

The run demonstrates that the configured system completed this particular simulated navigation scenario. The recorded outcome does not establish general obstacle-avoidance performance or physical flight safety.

## Demo 02 — Obstacle-Rich Navigation

**Scenario:** `s2026_e00323`

This experiment examines navigation in a more obstacle-dense environment containing three spawned obstacles and approximately ten static obstacles.

| Metric | Recorded value |
|---|---:|
| Control steps | 310 |
| Simulated duration | 58.75 s |
| Initial straight-line goal distance | Approximately 44 m |
| Final goal distance | 0.909 m |
| Path length | 96.4 m |
| Path efficiency | Approximately 0.45 |
| Safety overrides | 10 (3.2%) |
| Recorded collisions | 0 |
| Minimum reported LiDAR clearance | 1.74 m |
| Median inference latency | 563 ms |
| P95 inference latency | 583 ms |

The trajectory was substantially longer than the direct distance to the goal and exhibited hesitant, wall-hugging behavior. These observations indicate that reaching the goal in this scenario did not necessarily correspond to efficient navigation.

The reported latency was measured in a laptop-based, simulation-coupled execution setup. It should not be interpreted as evidence of onboard real-time performance.

## Demo 03 — Policy Attribution

**Scenario:** `s2026_e00086`

This experiment focuses on distinguishing the model's proposed actions from the actions ultimately executed after safety filtering.

| Metric | Recorded value |
|---|---:|
| Control steps | 190 |
| Simulated duration | 35.854 s |
| Initial goal distance | 44.816 m |
| Final goal distance | 0.886 m |
| Path length | 60.455 m |
| Path efficiency | 0.7248 |
| Safety overrides | 6 (3.2%) |
| Recorded collisions | 0 |
| Minimum reported horizontal LiDAR range | 1.704 m |
| Safety replay agreement | 190/190 decisions |
| Video SHA-256 | `73ef681dd4163cd61d78efc7e34f21986854ef145dd5dcfcaf8629bd5756e318` |

In 184 of 190 control steps (96.8%), the model's proposed action was executed without modification by the Safety Layer.

**This percentage measures action agreement with the safety filter, not action correctness, navigation success probability, or safety.**

The trajectory also contained prolonged, indecisive `FORWARD_LEFT` behavior, including movement that was not consistently aligned with the goal. The cause has not yet been established.

The 190/190 safety replay agreement indicates that the recorded safety decisions were reproduced in the replay procedure; it does not independently validate the safety policy itself.

## Interpretation and Limitations

These experiments provide initial evidence of J-BRAIN's integration into a simulated navigation pipeline. They should be interpreted within the following limits:

- **Simulation only:** All reported results come from AirSim, not real-world drone flights.
- **Safety Layer enabled:** The executed trajectory reflects both the model's proposals and the safety filter's interventions.
- **Limited scenarios:** Three demonstrations cannot establish generalization across environments, obstacle configurations, weather conditions, or hardware platforms.
- **Collision reporting:** Zero recorded collisions means that no collisions were recorded by the simulator during these runs. It is not a guarantee of collision-free operation in physical environments.
- **LiDAR measurements:** Reported ranges are sensor-derived measurements and should not be treated as independent geometric proof of minimum physical clearance.
- **Efficiency:** Reaching a goal does not necessarily imply an optimal or robust path.
- **Latency:** The reported inference latency does not establish that the current pipeline meets a real-time onboard control budget.
- **Recurrent memory:** These demonstrations do not, by themselves, establish that recurrent state was carried across navigation decisions.

## Reproducibility

The following information should accompany future published runs:

1. Exact checkpoint and its SHA-256 hash.
2. AirSim version and environment configuration.
3. Scenario identifiers and obstacle configurations.
4. Model inference settings and action-selection procedure.
5. Safety Layer configuration and override logs.
6. Definitions of path efficiency, clearance, and other reported metrics.
7. Raw measurement provenance and scripts used to calculate the summary statistics.

The scenario identifiers, recorded metrics, and checkpoint hash above identify the reported runs. Complete independent reproduction requires the associated configuration, code, and measurement artifacts.

## Video Artifacts

Video links can be added here once the files have been published:

- **Demo 01:** Basic Navigation — `JBRAIN_Autonomous_Drone_Control.mp4`
- **Demo 02:** Obstacle-Rich Navigation — `JBRAIN_Obstacle_Rich_Demo.mp4`
- **Demo 03:** Policy Attribution — `JBRAIN_Demo03_Policy_Attribution.mp4`

For large video files, consider attaching them to a GitHub Release and linking to that release rather than committing large binary files directly to the repository.

## Conclusion

The three demonstrations document an initial AirSim navigation integration for J-BRAIN, covering basic goal-directed movement, a more obstacle-dense scenario, and explicit analysis of model proposals versus safety-filtered actions.

The results are preliminary. Further evaluation should include repeated trials across diverse scenarios, paired comparisons of inference configurations, analysis of safety interventions and failure cases, and testing on the intended deployment hardware before making claims about real-time or real-world autonomous flight.