# Матриця тестів Chong-Fly

Дата: 2026-10-22. Збір через `venv/Scripts/python.exe -m pytest tests --collect-only -q -p no:cacheprovider`: **725 cases, 63 test files, 12.22 с**. Повний suite у цьому аудиті не запускався.

66 Python-файлів у tests/: крім 63 файлів із cases, два helpers і одна порожня legacy заглушка. Кількість враховує parametrization, а не лише кількість функцій.

Групи — рекомендація для майбутньої організації, не вже застосовані markers і не дозвіл видалити тести. Повний список test functions та dependencies — в [інвентарі](CODEBASE_INVENTORY_UK.md); node IDs — в [JSON](codebase_audit_inventory.json).

| Файл | Cases | Група | Що перевіряється |
|---|---:|---|---|
| `tests/conftest.py` | 0 | DATA_TRAINING_EVAL | Спільна pytest fixture tiny_model_meta; 0 окремих тестів — очікувано. |
| `tests/synthetic_graphs.py` | 0 | DATA_TRAINING_EVAL | Tiny synthetic graph fixtures/positive controls; 0 окремих тестів — очікувано. |
| `tests/test_altitude_control.py` | 15 | CORE_CONTRACT | Контур висоти, range/tilt, fault/reset, bounds, ownership |
| `tests/test_artifact_entrypoints.py` | 3 | DATA_TRAINING_EVAL | Явні шляхи артефактів у ручних CLI |
| `tests/test_behavior_cloning.py` | 8 | DATA_TRAINING_EVAL | Teacher targets, dataset generation, convergence behavioral cloning |
| `tests/test_brake_before_turn.py` | 6 | DATA_TRAINING_EVAL | Гальмування до повороту й регресія зіткнення |
| `tests/test_candidate_diagnostics.py` | 6 | DATA_TRAINING_EVAL | Research opt-in, метрики padding, native precision |
| `tests/test_candidate_pathway.py` | 17 | DATA_TRAINING_EVAL | CAVE projection, directional counts, rejection/provenance |
| `tests/test_candidate_selection.py` | 6 | DATA_TRAINING_EVAL | Registry verification, admission і study contract |
| `tests/test_cave_access_probe.py` | 6 | DATA_TRAINING_EVAL | CAVE доступ/версії та редагування помилок без витоку секретів |
| `tests/test_cave_auth_setup.py` | 9 | DATA_TRAINING_EVAL | Безпечне локальне введення/зберігання credentials |
| `tests/test_cave_client_integration.py` | 2 | DATA_TRAINING_EVAL | Реальний CAVEclient serializer/decoder на підміненому transport, не live API |
| `tests/test_cfc_dynamics.py` | 0 | DATA_TRAINING_EVAL | Порожній файл, 0 тестів. Реальні dynamics tests знаходяться у test_neural_contract.py. |
| `tests/test_command_gate_timing.py` | 6 | CORE_CONTRACT | Receiver TTL, replay, late replies та порядок викликів |
| `tests/test_control_ownership.py` | 14 | CORE_CONTRACT | Throttle не навчається, altitude owns applied throttle, несумісні contracts відхиляються |
| `tests/test_coordinate_contract.py` | 10 | CORE_CONTRACT | FLU/RC знаки, ToF орієнтація, flow/memory координати |
| `tests/test_demonstration_replay.py` | 3 | DATA_TRAINING_EVAL | Точний replay епізодів і відмова при підміні даних |
| `tests/test_downward_raycast.py` | 4 | DATA_TRAINING_EVAL | Downward cylinder caps та box face normals |
| `tests/test_drone_env.py` | 18 | DATA_TRAINING_EVAL | PT1/PID/mixer, physics, sensors і середовище |
| `tests/test_edge_runtime.py` | 15 | CORE_CONTRACT | Export/hash/parity, reset/fault, standalone без Torch |
| `tests/test_encoding_audit.py` | 4 | DATA_TRAINING_EVAL | Вимірювання encoding і projected input без мутації policy |
| `tests/test_evaluation_contract.py` | 13 | CORE_CONTRACT | Control/physics clocks, termination, метрики, NaN |
| `tests/test_exploration_teacher.py` | 17 | OPTIONAL_RESEARCH | Research scan supervisor, preemption/brake/resume |
| `tests/test_filtered_candidates.py` | 33 | DATA_TRAINING_EVAL | Filtered views, source cleaning, signs, bridges та candidate builds |
| `tests/test_flight_bridge.py` | 12 | CORE_CONTRACT | Admission/authority, handover, fault, freshness, receiver reference |
| `tests/test_functional_probe.py` | 14 | DATA_TRAINING_EVAL | Hooks/clamps, напрямки стимулів, recovery та незмінність model |
| `tests/test_levy_factorial.py` | 12 | OPTIONAL_RESEARCH | Пауза/target-half/pulse preemption і точність report метрик |
| `tests/test_memory.py` | 9 | OPTIONAL_RESEARCH | Rotation/decay/ring behavior; не доказ підключення memory до k306 |
| `tests/test_mvp_continuation.py` | 25 | DATA_TRAINING_EVAL | Frozen constraints, post-step, checkpoint restore і practical admission |
| `tests/test_navigation_completion.py` | 7 | DATA_TRAINING_EVAL | Proximity/neutral-origin, loss balance, physical yaw metrics |
| `tests/test_navigation_learning.py` | 13 | DATA_TRAINING_EVAL | Threat encoding/neutral origin, learned navigation та scene checks |
| `tests/test_navigation_scenes.py` | 4 | DATA_TRAINING_EVAL | Сценарії фізичних navigation challenges |
| `tests/test_neural_contract.py` | 31 | CORE_CONTRACT | Rate equations, matrix orientation, solver/time/reset/contracts |
| `tests/test_optical_flow_exploration.py` | 8 | OPTIONAL_RESEARCH | OU wrapper/units/seed/reset і limits |
| `tests/test_policy.py` | 7 | CORE_CONTRACT | Sensor/head/PWM композиція та step/reset |
| `tests/test_pretrain_contract.py` | 17 | DATA_TRAINING_EVAL | Версії даних, valid-prefix, split/weights/contracts |
| `tests/test_published_annotations.py` | 28 | DATA_TRAINING_EVAL | Published IDs/labels, hashes, exact source selection |
| `tests/test_recurrent_dynamics_probe.py` | 11 | DATA_TRAINING_EVAL | Copy-only tau/W scaling та bounded search restrictions |
| `tests/test_reduction_contract.py` | 6 | DATA_TRAINING_EVAL | Directed aggregation/normalization, legacy migration |
| `tests/test_reflex_training_bundle.py` | 2 | DATA_TRAINING_EVAL | Labels/provenance цілісність training bundle |
| `tests/test_role_reduction.py` | 19 | DATA_TRAINING_EVAL | Ролі/сторони/знаки, input-output nodes, reproducible reduction |
| `tests/test_route_audit.py` | 4 | DATA_TRAINING_EVAL | Source gap проти reduction loss/invented route |
| `tests/test_routing_contract.py` | 30 | CORE_CONTRACT | Mask enforcement, no hidden dense paths, disconnected memory |
| `tests/test_scan_probe.py` | 8 | OPTIONAL_RESEARCH | Scripted maneuver/metrics, brake prerequisite і phase logs |
| `tests/test_search_pipeline.py` | 11 | DATA_TRAINING_EVAL | Registry study/run/resume/checkpoints і research gate |
| `tests/test_simulation_contract.py` | 12 | CORE_CONTRACT | Swept/body collisions, pure sensor reads, physics validity |
| `tests/test_source_contract.py` | 30 | DATA_TRAINING_EVAL | IDs, query completeness, immutable bundles/provenance |
| `tests/test_spatial_calibration.py` | 8 | OPTIONAL_RESEARCH | Experimental mean-proximity encoding і mirrored temporal stimuli |
| `tests/test_study_contract.py` | 17 | DATA_TRAINING_EVAL | Feasible Pareto, experiment identity, explicit CUDA та registry |
| `tests/test_synapse_sign_constraint.py` | 5 | CORE_CONTRACT | Sign preservation після updates і checkpoint restore |
| `tests/test_synthetic_examples.py` | 3 | DATA_TRAINING_EVAL | Малі явно synthetic graphs та route positive controls |
| `tests/test_teacher_braking.py` | 12 | DATA_TRAINING_EVAL | Teacher braking hysteresis, release, historical version replay |
| `tests/test_teacher_comparison.py` | 14 | DATA_TRAINING_EVAL | Matched teacher comparisons і recorded regression cases |
| `tests/test_teacher_contract.py` | 12 | DATA_TRAINING_EVAL | Валідність/версії teacher observations/labels та sensor timebase |
| `tests/test_teacher_v5.py` | 5 | DATA_TRAINING_EVAL | Versioned v5 selection, replay, rejection старої aggregation |
| `tests/test_tof_geometry.py` | 11 | DATA_TRAINING_EVAL | Front ToF raycast caps/slabs/occlusion і thin targets |
| `tests/test_tof_surfaces.py` | 15 | OPTIONAL_RESEARCH | Analytic planes/spheres intersections та invalid geometry |
| `tests/test_tof_threat_contract.py` | 15 | DATA_TRAINING_EVAL | Continuous aggregation, input boundaries і monotonic threat response |
| `tests/test_training_coverage.py` | 4 | DATA_TRAINING_EVAL | Mask/split correctness, absent-data semantics |
| `tests/test_watch_mvp.py` | 6 | DATA_TRAINING_EVAL | Trace/physics timing, applied throttle, checkpoint identity і no-training viewer |
| `tests/test_xy_behaviors.py` | 16 | CORE_CONTRACT | Levy/STOP/preemption, seed/reset і priority |
| `tests/test_xy_coordinator.py` | 13 | CORE_CONTRACT | Anti-rollback/latch, disabled identity, near-zero preset |
| `tests/test_xy_passages.py` | 2 | DATA_TRAINING_EVAL | Interpolated crossing і відокремлення STOP від описового passage |
| `tests/test_xy_starts.py` | 5 | DATA_TRAINING_EVAL | Frozen starts/noise, hazard durations і admission |
| `tests/test_yaw_recovery.py` | 7 | DATA_TRAINING_EVAL | Causal recovery labels, loss balance, carryover/reset та constraints |
| `tests/test_zonal_tof.py` | 20 | OPTIONAL_RESEARCH | Aggregator semantics, valid samples, mirror/rotation і research-only contract |

**Разом: 725 cases.**

Літерально однакових AST тіл test-функцій не знайдено. Семантичне дублювання не доведено і не спростовано цією перевіркою. Geometry/downward raycast, controller/runtime/bridge перевіряють різні boundaries; схожі сюжети самі по собі не є підставою для видалення.
