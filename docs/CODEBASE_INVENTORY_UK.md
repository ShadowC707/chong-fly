# Повна інвентаризація Python-файлів

Дата: 2026-10-22. Основний висновок і правила класифікації: [CODEBASE_AUDIT_UK.md](CODEBASE_AUDIT_UK.md).

159 Python-файлів у 9 запитаних каталогах; додатково 6 кореневих/benchmark файлів. Інвентаризація зроблена через AST без import-everything. Повні docstrings, methods, імпорти з рядками, SHA-256 кожного файла та 725 collected node IDs містяться в [JSON](codebase_audit_inventory.json).

**Статус стосується поточного k306 MVP.** `DORMANT_USEFUL` не означає перевірену готовність до ввімкнення. `ONE_OFF_HISTORY` означає фіксований експеримент, не дозвіл видалити його helpers. Статичний importer може бути lazy/optional; його наявність не доводить виконання гілки. CLI без importer не є мертвим кодом.

## core

### `core/contracts.py`

**ACTIVE_CORE** — Версії рівнянь, орієнтація матриці, мінімальна tau й сумісні назви інтегратора.

Символи: `canonical_solver (L9)`.

Вхідні імпорти з перелічених production/CLI файлів: `core/models.py`, `generator/graph_reducer.py`.

### `core/models.py`

**ACTIVE_CORE** — PyTorch rate-ядро, маски/знаки W, інтегрування та побудова з метаданих; назви CfC збережені для сумісності.

Символи: `class _LeCunTanh (L40)`; `_make_activation (L46)`; `class BiologicalCfCCell (L65)`; `class BiologicalCfCNetwork (L390)`; `build_network_from_meta (L489)`.

Локальні залежності (AST): `configs/flight_config.py`, `core/contracts.py`, `core/routing.py`, `generator/graph_reducer.py`.

Вхідні імпорти з перелічених production/CLI файлів: `optimizer/probe_recurrent_dynamics.py`, `simulation/policy.py`.

### `core/routing.py`

**ACTIVE_CORE** — Вхідні/вихідні маски, RoutedLinear, структурний ранг та спрямована досяжність.

Символи: `_indices (L14)`; `sensor_mask (L22)`; `motor_mask (L54)`; `structural_rank (L74)`; `sensor_motor_paths (L90)`; `class RoutedLinear (L132)`.

Вхідні імпорти з перелічених production/CLI файлів: `core/models.py`, `generator/audit_routes.py`, `generator/role_reducer.py`, `optimizer/diagnose_candidate.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/search_pipeline.py`, `simulation/policy.py`.

## configs

### `configs/__init__.py`

**ACTIVE_CORE** — Сумісний реекспорт конфігурації; порожній перелік функцій тут нормальний.

Символи: немає top-level класів/функцій; див. призначення..

Локальні залежності (AST): `configs/flight_config.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `configs/flight_config.py`

**ACTIVE_CORE** — Спільні розміри, координати, частоти, RC, фізичні й сенсорні константи host-коду; не є єдиним джерелом усіх edge-порогів.

Символи: `class SensorsConfig (L22)`; `class ActuatorConfig (L58)`; `class PhysicsConfig (L90)`; `class ArenaConfig (L126)`; `class APFConfig (L152)`; `class SpatialMemoryConfig (L170)`; `class MetricsConfig (L181)`.

Вхідні імпорти з перелічених production/CLI файлів: `config.py`, `configs/__init__.py`, `core/models.py`, `evaluate/validate_xy_starts.py`, `evaluate/watch_champion.py`, `evaluate/watch_mvp.py`, `generator/exploration_teacher.py`, `generator/generate_reflex_dataset.py`, `generator/prepare_flywire_candidates.py`, `generator/validate_flywire_candidates.py`, `optimizer/benchmark_candidate.py`, `optimizer/candidate_selection.py`, `optimizer/compare_teachers.py`, `optimizer/diagnose_candidate.py`, `optimizer/evaluate.py`, `optimizer/finalize_mvp_candidate.py`, `optimizer/functional_probe.py`, `optimizer/navigation_loss.py`, `optimizer/pretrain.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/replay_demonstrations.py`, `optimizer/rollout.py`, `optimizer/search_pipeline.py`, `optimizer/validate_altitude.py`, `scripts/audit_sensor_encoding.py`, `scripts/audit_tof_threats.py`, `scripts/probe_exploration_teacher.py`, `scripts/probe_zonal_tof.py`, `scripts/scan_probe.py`, `simulation/avionics_filter.py`, `simulation/benchmark_scenes.py`, `simulation/drone_env.py`, `simulation/drone_interface.py`, `simulation/memory.py`, `simulation/metrics.py`, `simulation/pmw3901_emulator.py`, `simulation/policy.py`, `simulation/policy_diagnostics.py`, `simulation/xy_scenes.py`.

## generator

### `generator/acquire_public_connectome.py`

**DORMANT_USEFUL** — Ранній pinned CAVE/annotation acquisition; зберегти для відтворення джерел, поточні filtered кандидати готуються іншим CLI.

Символи: `prepare_selection (L16)`; `main (L39)`.

Локальні залежності (AST): `generator/cave_source.py`, `generator/circuit_extractor.py`, `generator/published_annotations.py`, `generator/source_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `generator/audit_candidate_pathway.py`

**DORMANT_USEFUL** — Спрямований анатомічний аудит вибраних коренів; не створює автоматично придатну до навчання модель.

Символи: `select_pathway (L19)`; `audit_candidate_pathway (L54)`; `projection_summary (L97)`; `main (L129)`.

Локальні залежності (AST): `generator/cave_source.py`, `generator/published_annotations.py`, `generator/source_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `generator/audit_routes.py`

**ACTIVE_EVAL** — Порівняння sensor→motor шляхів у джерелі й редукції, без додавання зв'язків.

Символи: `audit_routes (L20)`; `main (L64)`.

Локальні залежності (AST): `core/routing.py`, `generator/graph_reducer.py`, `generator/reflex_contract.py`, `generator/role_reducer.py`.

Вхідні імпорти з перелічених production/CLI файлів: `generator/prepare_flywire_candidates.py`, `generator/validate_flywire_candidates.py`.

### `generator/cave_source.py`

**DORMANT_USEFUL** — Завантаження pinned CAVE даних із перевіркою кількості, розбиттям запитів і без synthetic fallback.

Символи: `_count (L14)`; `_fetch_complete (L24)`; `fetch_cave_graph (L53)`; `fetch_cave_connectivity (L88)`.

Локальні залежності (AST): `generator/source_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: `generator/acquire_public_connectome.py`, `generator/audit_candidate_pathway.py`, `generator/circuit_extractor.py`, `generator/filtered_candidates.py`.

### `generator/circuit_extractor.py`

**DORMANT_USEFUL** — Відбір клітин і CLI джерела; synthetic helpers потрібні тестам, не є джерелом активних ваг.

Символи: `_project_root (L26)`; `load_cell_mapping (L30)`; `build_target_classes (L38)`; `build_exclusion_sets (L87)`; `filter_nodes (L97)`; `_build_synthetic_nodes (L128)`; `_build_synthetic_edges (L233)`; `main (L404)`.

Локальні залежності (AST): `generator/cave_source.py`, `generator/source_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: `generator/acquire_public_connectome.py`.

### `generator/continuous_reflex_teacher.py`

**DORMANT_USEFUL** — Неперервна агрегація загроз і виправлений teacher v5; потрібен відтворенню навчання/діагностиці.

Символи: `class ContinuousReflexTeacher (L13)`; `class CorrectedExpertReflexPolicy (L33)`.

Локальні залежності (AST): `generator/generate_reflex_dataset.py`, `generator/reflex_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: `generator/generate_reflex_dataset.py`, `optimizer/summarize_training_pilot.py`, `scripts/audit_tof_threats.py`, `scripts/probe_zonal_tof.py`.

### `generator/exploration_teacher.py`

**DORMANT_USEFUL** — Експериментальний teacher зі скануванням; не керує активною k306 на edge.

Символи: `class ScanBlocked (L16)`; `class ScanConfig (L21)`; `class ExplorationTeacher (L40)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/generate_reflex_dataset.py`, `generator/reflex_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: `scripts/probe_exploration_teacher.py`.

### `generator/filtered_candidates.py`

**DORMANT_USEFUL** — Офіційні filtered views, очищення, припущення знаків, bridge selection і reachability.

Символи: `fetch_view (L20)`; `clean_synapses (L75)`; `assign_polarities (L100)`; `select_bridges (L121)`; `compare_reachability (L133)`.

Локальні залежності (AST): `generator/cave_source.py`, `generator/source_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: `generator/prepare_flywire_candidates.py`, `generator/validate_flywire_candidates.py`.

### `generator/generate_reflex_dataset.py`

**DORMANT_USEFUL** — Геометричні демонстрації, версійна фабрика вчителів і історичні політики; factory також викликається probe-ами.

Символи: `class LegacyExpertReflexPolicy (L36)`; `class BrakingExpertReflexPolicy (L116)`; `class ExpertReflexPolicy (L166)`; `make_teacher (L190)`; `_make_scenario (L207)`; `_behavior (L237)`; `generate_reflex_dataset (L246)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/continuous_reflex_teacher.py`, `generator/reflex_contract.py`, `simulation/altitude_control.py`, `simulation/control_contract.py`, `simulation/drone_env.py`, `simulation/memory.py`.

Вхідні імпорти з перелічених production/CLI файлів: `evaluate/watch_champion.py`, `generator/continuous_reflex_teacher.py`, `generator/exploration_teacher.py`, `generator/prepare_reflex_training.py`, `optimizer/compare_teachers.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/replay_demonstrations.py`, `optimizer/validate_altitude.py`, `scripts/audit_tof_threats.py`, `scripts/audit_training_coverage.py`, `simulation/policy_diagnostics.py`.

### `generator/graph_reducer.py`

**ACTIVE_CORE** — ReducedModel та читання графа потрібні host-завантаженню; spectral/centrality/magnitude редуктори — окрема дослідницька гілка.

Символи: `class ReducedModel (L69)`; `benchmark_matrix (L207)`; `_normalize_rows_jit (L269)`; `_condense_synapses_jit (L285)`; `class BaseReducer (L318)`; `class SpectralReducer (L460)`; `class CentralityReducer (L539)`; `class MagnitudePruner (L609)`; `_project_root (L688)`; `load_graph (L692)`; `run_reduction (L749)`; `_print_header (L865)`; `_print_row (L871)`; `_append_manifest (L879)`.

Локальні залежності (AST): `core/contracts.py`, `generator/source_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: `core/models.py`, `generator/audit_routes.py`, `generator/prepare_flywire_candidates.py`, `generator/role_reducer.py`, `generator/validate_flywire_candidates.py`.

### `generator/prepare_flywire_candidates.py`

**ACTIVE_CORE** — Змішаний файл: acquisition/build CLI відкладені, але publication/write_json імпортуються активними звітами й loader-модулями.

Символи: `validate_config (L28)`; `write_json (L51)`; `publication (L56)`; `select_seeds (L67)`; `acquire (L78)`; `read_source (L157)`; `smoke (L182)`; `build (L210)`; `main (L290)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/audit_routes.py`, `generator/filtered_candidates.py`, `generator/graph_reducer.py`, `generator/published_annotations.py`, `generator/role_reducer.py`, `generator/source_contract.py`, `generator/validate_flywire_candidates.py`, `simulation/control_contract.py`, `simulation/policy.py`.

Вхідні імпорти з перелічених production/CLI файлів: `generator/prepare_reflex_training.py`, `generator/validate_flywire_candidates.py`, `optimizer/benchmark_candidate.py`, `optimizer/compare_teachers.py`, `optimizer/diagnose_candidate.py`, `optimizer/finalize_mvp_candidate.py`, `optimizer/functional_probe.py`, `optimizer/plot_yaw_recovery.py`, `optimizer/probe_recurrent_dynamics.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/replay_demonstrations.py`, `optimizer/search_pipeline.py`, `optimizer/search_recurrent_dynamics.py`, `optimizer/summarize_training_pilot.py`, `scripts/audit_sensor_encoding.py`, `scripts/audit_tof_threats.py`, `scripts/audit_training_coverage.py`, `scripts/probe_exploration_teacher.py`, `scripts/probe_tof_geometry.py`, `scripts/probe_tof_surfaces.py`, `scripts/probe_zonal_tof.py`, `scripts/scan_probe.py`.

### `generator/prepare_reflex_training.py`

**DORMANT_USEFUL** — Атомарна публікація перевіреного v5 training bundle; заново запускати для цього аудиту не потрібно.

Символи: `audit_labels (L18)`; `prepare (L39)`; `main (L75)`.

Локальні залежності (AST): `generator/generate_reflex_dataset.py`, `generator/prepare_flywire_candidates.py`, `generator/reflex_contract.py`, `generator/source_contract.py`, `optimizer/search_pipeline.py`.

Вхідні імпорти з перелічених production/CLI файлів: `optimizer/summarize_training_pilot.py`.

### `generator/probe_cave.py`

**DORMANT_USEFUL** — Обмежена перевірка доступу/версій CAVE без друку секретів.

Символи: `probe_access (L18)`; `main (L98)`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `generator/published_annotations.py`

**DORMANT_USEFUL** — Відбір точних опублікованих labels/root IDs із pinned provenance.

Символи: `annotation_source (L19)`; `select_annotations (L27)`.

Локальні залежності (AST): `generator/source_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: `generator/acquire_public_connectome.py`, `generator/audit_candidate_pathway.py`, `generator/prepare_flywire_candidates.py`.

### `generator/reflex_contract.py`

**ACTIVE_CORE** — Версії вчителя/датасету, сценарії, поведінки та required paths; також використовується діагностикою.

Символи: немає top-level класів/функцій; див. призначення..

Вхідні імпорти з перелічених production/CLI файлів: `generator/audit_routes.py`, `generator/continuous_reflex_teacher.py`, `generator/exploration_teacher.py`, `generator/generate_reflex_dataset.py`, `generator/prepare_reflex_training.py`, `optimizer/compare_teachers.py`, `optimizer/evaluate.py`, `optimizer/finalize_mvp_candidate.py`, `optimizer/optuna_tuner.py`, `optimizer/pretrain.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/search_pipeline.py`, `optimizer/summarize_training_pilot.py`, `optimizer/validate_altitude.py`, `scripts/audit_tof_threats.py`, `simulation/policy_diagnostics.py`.

### `generator/role_reducer.py`

**DORMANT_USEFUL** — Рольова редукція зі збереженням напрямків, сторін, знаків та інтерфейсних нейронів.

Символи: `class RolePreservingReducer (L26)`; `resolve_source_kind (L164)`; `main (L175)`.

Локальні залежності (AST): `core/routing.py`, `generator/graph_reducer.py`, `generator/source_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: `generator/audit_routes.py`, `generator/prepare_flywire_candidates.py`, `generator/validate_flywire_candidates.py`, `optimizer/evaluate.py`.

### `generator/setup_cave_auth.py`

**DORMANT_USEFUL** — Інтерактивне налаштування credential store; не частина inference.

Символи: `setup_auth (L12)`; `main (L39)`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `generator/source_contract.py`

**ACTIVE_CORE** — Цілісність/походження raw bundles, canonical IDs, SHA-256 та незмінна публікація; hash helpers повторно використовуються всюди.

Символи: `canonical_ids (L24)`; `validate_graph (L37)`; `file_sha256 (L64)`; `manifest_sha256 (L72)`; `validate_source (L76)`; `publish_directory (L103)`; `write_bundle (L121)`; `read_bundle (L150)`.

Вхідні імпорти з перелічених production/CLI файлів: `evaluate/audit_xy_behaviors.py`, `evaluate/compare_levy_factorial.py`, `evaluate/compare_xy_behaviors.py`, `evaluate/compare_xy_coordinator.py`, `evaluate/validate_xy_starts.py`, `evaluate/watch_mvp.py`, `generator/acquire_public_connectome.py`, `generator/audit_candidate_pathway.py`, `generator/cave_source.py`, `generator/circuit_extractor.py`, `generator/filtered_candidates.py`, `generator/graph_reducer.py`, `generator/prepare_flywire_candidates.py`, `generator/prepare_reflex_training.py`, `generator/published_annotations.py`, `generator/role_reducer.py`, `generator/validate_flywire_candidates.py`, `optimizer/benchmark_candidate.py`, `optimizer/candidate_selection.py`, `optimizer/compare_teachers.py`, `optimizer/diagnose_candidate.py`, `optimizer/finalize_mvp_candidate.py`, `optimizer/functional_probe.py`, `optimizer/plot_yaw_recovery.py`, `optimizer/probe_recurrent_dynamics.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/replay_demonstrations.py`, `optimizer/search_pipeline.py`, `optimizer/search_recurrent_dynamics.py`, `optimizer/summarize_training_pilot.py`, `scripts/audit_sensor_encoding.py`, `scripts/audit_tof_threats.py`, `scripts/audit_training_coverage.py`, `scripts/probe_exploration_teacher.py`, `scripts/probe_tof_geometry.py`, `scripts/probe_tof_surfaces.py`, `scripts/probe_zonal_tof.py`, `scripts/scan_probe.py`.

### `generator/validate_flywire_candidates.py`

**ACTIVE_EVAL** — Offline quality/structural checks і smoke кандидатів; не надає льотного допуску.

Символи: `verify_files (L18)`; `quality_audit (L28)`; `make_reducer (L66)`; `directed_witness (L79)`; `check_structural_admission (L100)`; `functional_smoke (L129)`; `validate (L153)`; `main (L226)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/audit_routes.py`, `generator/filtered_candidates.py`, `generator/graph_reducer.py`, `generator/prepare_flywire_candidates.py`, `generator/role_reducer.py`, `generator/source_contract.py`, `simulation/policy.py`, `simulation/policy_diagnostics.py`.

Вхідні імпорти з перелічених production/CLI файлів: `generator/prepare_flywire_candidates.py`, `optimizer/candidate_selection.py`, `optimizer/diagnose_candidate.py`.

## optimizer

### `optimizer/benchmark_candidate.py`

**ACTIVE_CORE** — Змішаний loader точного checkpoint + CLI фізичних benchmark сцен; loader потрібен export/watch.

Символи: `class ToFAblation (L20)`; `load_research_run (L39)`; `benchmark_run (L77)`; `main (L121)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/prepare_flywire_candidates.py`, `generator/source_contract.py`, `optimizer/mvp_constraints.py`, `optimizer/rollout.py`, `simulation/altitude_control.py`, `simulation/benchmark_scenes.py`, `simulation/control_contract.py`, `simulation/policy.py`, `simulation/policy_diagnostics.py`.

Вхідні імпорти з перелічених production/CLI файлів: `edge/export_numpy.py`, `edge/validate_numpy.py`, `evaluate/watch_mvp.py`, `optimizer/finalize_mvp_candidate.py`, `optimizer/functional_probe.py`, `optimizer/probe_recurrent_dynamics.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/search_recurrent_dynamics.py`, `optimizer/summarize_training_pilot.py`, `scripts/audit_sensor_encoding.py`.

### `optimizer/candidate_selection.py`

**DORMANT_USEFUL** — Верифікований registry pool, хеш контракту й відбір дозволених research кандидатів.

Символи: `contract_sha256 (L10)`; `class CandidatePool (L15)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/source_contract.py`, `generator/validate_flywire_candidates.py`, `simulation/policy.py`.

Вхідні імпорти з перелічених production/CLI файлів: `optimizer/finalize_mvp_candidate.py`, `optimizer/optuna_tuner.py`, `optimizer/search_pipeline.py`, `optimizer/search_recurrent_dynamics.py`.

### `optimizer/cma_es_tuner.py`

**DEAD_LEGACY** — Порожній файл, 0 байтів: CMA-ES тут не реалізований.

Символи: немає top-level класів/функцій; див. призначення..

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `optimizer/compare_teachers.py`

**DORMANT_USEFUL** — Парне порівняння версій вчителя на збережених початкових умовах.

Символи: `_rollout (L22)`; `compare_teachers (L83)`; `main (L137)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/generate_reflex_dataset.py`, `generator/prepare_flywire_candidates.py`, `generator/reflex_contract.py`, `generator/source_contract.py`, `optimizer/search_pipeline.py`, `simulation/altitude_control.py`, `simulation/memory.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `optimizer/diagnose_candidate.py`

**DORMANT_USEFUL** — Research-only навчальна діагностика, heldout predictions і короткі rollout-и; helpers використовуються recovery experiments.

Символи: `prediction_metrics (L20)`; `heldout_predictions (L54)`; `short_rollouts (L62)`; `run_diagnostic (L73)`; `main (L240)`.

Локальні залежності (AST): `configs/flight_config.py`, `core/routing.py`, `generator/prepare_flywire_candidates.py`, `generator/source_contract.py`, `generator/validate_flywire_candidates.py`, `optimizer/pretrain.py`, `optimizer/rollout.py`, `simulation/altitude_control.py`, `simulation/control_contract.py`, `simulation/policy.py`, `simulation/policy_diagnostics.py`.

Вхідні імпорти з перелічених production/CLI файлів: `optimizer/probe_yaw_recovery.py`, `optimizer/search_pipeline.py`, `optimizer/summarize_training_pilot.py`.

### `optimizer/evaluate.py`

**DORMANT_USEFUL** — Фабрика моделей і старший objective multi-seed оцінки; не основний шлях відтворення замороженого MVP.

Символи: `class DefaultFlightPolicy (L21)`; `create_model (L85)`; `objective (L181)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/reflex_contract.py`, `generator/role_reducer.py`, `optimizer/pretrain.py`, `optimizer/rollout.py`, `simulation/altitude_control.py`, `simulation/control_contract.py`, `simulation/policy.py`.

Вхідні імпорти з перелічених production/CLI файлів: `evaluate/watch_champion.py`.

### `optimizer/finalize_mvp_candidate.py`

**ONE_OFF_HISTORY** — Відтворювана фіксована серія MVP continuation: baseline/after, заморожування, окремий practical admission; зберегти як рецепт походження ваг.

Символи: `trial_settings (L38)`; `assess_mvp (L48)`; `evaluate_phase (L105)`; `save_phase (L112)`; `comparison_row (L135)`; `write_comparison (L147)`; `run_search (L169)`; `main (L299)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/prepare_flywire_candidates.py`, `generator/reflex_contract.py`, `generator/source_contract.py`, `optimizer/benchmark_candidate.py`, `optimizer/candidate_selection.py`, `optimizer/mvp_constraints.py`, `optimizer/optuna_tuner.py`, `optimizer/pretrain.py`, `optimizer/probe_recurrent_dynamics.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/rollout.py`, `optimizer/search_pipeline.py`, `simulation/benchmark_scenes.py`, `simulation/control_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `optimizer/functional_probe.py`

**ACTIVE_EVAL** — Контрольовані стимули, trace та тимчасове вимкнення груп для функціонального розбору.

Символи: `make_stimuli (L24)`; `trace_response (L43)`; `resolve_groups (L82)`; `response_metrics (L139)`; `plot_results (L166)`; `run_probe (L197)`; `main (L272)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/prepare_flywire_candidates.py`, `generator/source_contract.py`, `optimizer/benchmark_candidate.py`.

Вхідні імпорти з перелічених production/CLI файлів: `scripts/audit_sensor_encoding.py`.

### `optimizer/mvp_constraints.py`

**ACTIVE_CORE** — Заморожування ядра й обмеження норм; metadata restore викликається при завантаженні поточного checkpoint.

Символи: `class RecurrentConstraints (L10)`; `decoder_norms (L134)`; `verify_clear_drive (L142)`.

Локальні залежності (AST): `optimizer/probe_yaw_recovery.py`.

Вхідні імпорти з перелічених production/CLI файлів: `optimizer/benchmark_candidate.py`, `optimizer/finalize_mvp_candidate.py`.

### `optimizer/navigation_loss.py`

**DORMANT_USEFUL** — Версійні channel losses і causal recovery mask; потрібні наступному навчальному циклу.

Символи: `_balanced (L9)`; `yaw_recovery_mask (L17)`; `navigation_weights (L38)`; `navigation_loss (L74)`.

Локальні залежності (AST): `configs/flight_config.py`, `simulation/control_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: `optimizer/pretrain.py`, `optimizer/probe_yaw_recovery.py`.

### `optimizer/optuna_tuner.py`

**DORMANT_USEFUL** — Study contract/Pareto feasibility та делегування registry search; не слід видаляти через паузу навчання.

Символи: `safety_constraints (L36)`; `feasible_pareto_trials (L52)`; `create_study (L74)`; `run_optuna_study (L128)`.

Локальні залежності (AST): `generator/reflex_contract.py`, `optimizer/candidate_selection.py`, `optimizer/rollout.py`, `optimizer/search_pipeline.py`, `simulation/control_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: `optimizer/finalize_mvp_candidate.py`, `optimizer/search_pipeline.py`, `optimizer/search_recurrent_dynamics.py`.

### `optimizer/plot_yaw_recovery.py`

**ONE_OFF_HISTORY** — Побудова графіків зі збереженої recovery серії без нового навчання.

Символи: `plot_report (L10)`; `main (L49)`.

Локальні залежності (AST): `generator/prepare_flywire_candidates.py`, `generator/source_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `optimizer/pretrain.py`

**DORMANT_USEFUL** — Behavior cloning: перевірка контрактів, розбиття епізодів, маски padding, balanced loss і post-step constraints.

Символи: `pretrain_policy (L32)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/reflex_contract.py`, `optimizer/navigation_loss.py`, `simulation/control_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: `bc_test.py`, `evaluate/watch_champion.py`, `optimizer/diagnose_candidate.py`, `optimizer/evaluate.py`, `optimizer/finalize_mvp_candidate.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/search_pipeline.py`, `optimizer/search_recurrent_dynamics.py`.

### `optimizer/probe_recurrent_dynamics.py`

**DORMANT_USEFUL** — Copy-only масштабування tau/W та кінцева сітка probe-ів; scaled_policy також використовується continuation.

Символи: `scaled_policy (L26)`; `summarize_probe (L52)`; `run_grid (L77)`; `plot_grid (L141)`; `main (L165)`.

Локальні залежності (AST): `core/models.py`, `generator/prepare_flywire_candidates.py`, `generator/source_contract.py`, `optimizer/benchmark_candidate.py`, `optimizer/search_pipeline.py`, `simulation/policy_diagnostics.py`.

Вхідні імпорти з перелічених production/CLI файлів: `optimizer/finalize_mvp_candidate.py`, `optimizer/search_recurrent_dynamics.py`.

### `optimizer/probe_yaw_recovery.py`

**ACTIVE_CORE** — Змішаний файл: recovery experiment відкладено, але verify_recurrent_contract потрібен mvp_constraints/loader.

Символи: `recovery_probe (L32)`; `evaluate (L72)`; `verify_recurrent_contract (L103)`; `run_experiment (L112)`; `main (L196)`.

Локальні залежності (AST): `configs/flight_config.py`, `core/routing.py`, `generator/generate_reflex_dataset.py`, `generator/prepare_flywire_candidates.py`, `generator/reflex_contract.py`, `generator/source_contract.py`, `optimizer/benchmark_candidate.py`, `optimizer/diagnose_candidate.py`, `optimizer/navigation_loss.py`, `optimizer/pretrain.py`, `optimizer/rollout.py`, `optimizer/search_pipeline.py`, `simulation/altitude_control.py`, `simulation/benchmark_scenes.py`, `simulation/policy_diagnostics.py`.

Вхідні імпорти з перелічених production/CLI файлів: `optimizer/finalize_mvp_candidate.py`, `optimizer/mvp_constraints.py`, `optimizer/search_recurrent_dynamics.py`.

### `optimizer/replay_demonstrations.py`

**DORMANT_USEFUL** — Точний replay збережених демонстрацій, зокрема їхніх невдач; без виправлення labels заднім числом.

Символи: `audit_failures (L16)`; `main (L93)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/generate_reflex_dataset.py`, `generator/prepare_flywire_candidates.py`, `generator/source_contract.py`, `optimizer/search_pipeline.py`, `simulation/altitude_control.py`, `simulation/memory.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `optimizer/rollout.py`

**ACTIVE_EVAL** — Основний host closed-loop runner: 50 Гц control/250 Гц physics, XY, altitude, traces і метрики; не включає FlightBridge/Gate.

Символи: `_observation (L25)`; `_state (L44)`; `_policy_action (L59)`; `simulate_policy_rollout (L77)`.

Локальні залежності (AST): `configs/flight_config.py`, `simulation/altitude_control.py`, `simulation/control_contract.py`, `simulation/drone_env.py`, `simulation/memory.py`, `simulation/metrics.py`.

Вхідні імпорти з перелічених production/CLI файлів: `evaluate/watch_mvp.py`, `optimizer/benchmark_candidate.py`, `optimizer/diagnose_candidate.py`, `optimizer/evaluate.py`, `optimizer/finalize_mvp_candidate.py`, `optimizer/optuna_tuner.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/search_pipeline.py`, `optimizer/search_recurrent_dynamics.py`, `optimizer/summarize_training_pilot.py`, `optimizer/validate_altitude.py`, `scripts/audit_tof_threats.py`, `scripts/probe_exploration_teacher.py`, `scripts/probe_tof_geometry.py`, `scripts/probe_zonal_tof.py`, `scripts/scan_probe.py`.

### `optimizer/search_pipeline.py`

**DORMANT_USEFUL** — Registry training/search orchestration, gate, resumable study, точні checkpoints; не тотожний frozen-core рецепту.

Символи: `validate_search_config (L38)`; `assess_navigation (L71)`; `validate_demonstration_context (L129)`; `export_trial (L151)`; `candidate_objective (L187)`; `run_search (L251)`; `main (L317)`.

Локальні залежності (AST): `configs/flight_config.py`, `core/routing.py`, `generator/prepare_flywire_candidates.py`, `generator/reflex_contract.py`, `generator/source_contract.py`, `optimizer/candidate_selection.py`, `optimizer/diagnose_candidate.py`, `optimizer/optuna_tuner.py`, `optimizer/pretrain.py`, `optimizer/rollout.py`, `simulation/altitude_control.py`, `simulation/benchmark_scenes.py`, `simulation/control_contract.py`, `simulation/policy.py`, `simulation/policy_diagnostics.py`.

Вхідні імпорти з перелічених production/CLI файлів: `generator/prepare_reflex_training.py`, `optimizer/compare_teachers.py`, `optimizer/finalize_mvp_candidate.py`, `optimizer/optuna_tuner.py`, `optimizer/probe_recurrent_dynamics.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/replay_demonstrations.py`, `optimizer/search_recurrent_dynamics.py`.

### `optimizer/search_recurrent_dynamics.py`

**ONE_OFF_HISTORY** — Фіксована серія донавчання після screen масштабувань; відтворення історичної гіпотези.

Символи: `trial_settings (L27)`; `run_search (L32)`; `main (L150)`.

Локальні залежності (AST): `generator/prepare_flywire_candidates.py`, `generator/source_contract.py`, `optimizer/benchmark_candidate.py`, `optimizer/candidate_selection.py`, `optimizer/optuna_tuner.py`, `optimizer/pretrain.py`, `optimizer/probe_recurrent_dynamics.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/rollout.py`, `optimizer/search_pipeline.py`, `simulation/control_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `optimizer/summarize_training_pilot.py`

**ONE_OFF_HISTORY** — Агрегація результатів попереднього training pilot.

Символи: `summarize (L25)`; `main (L109)`.

Локальні залежності (AST): `generator/continuous_reflex_teacher.py`, `generator/prepare_flywire_candidates.py`, `generator/prepare_reflex_training.py`, `generator/reflex_contract.py`, `generator/source_contract.py`, `optimizer/benchmark_candidate.py`, `optimizer/diagnose_candidate.py`, `optimizer/rollout.py`, `scripts/audit_tof_threats.py`, `simulation/altitude_control.py`, `simulation/benchmark_scenes.py`, `simulation/policy.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `optimizer/validate_altitude.py`

**ACTIVE_EVAL** — Окрема перевірка зовнішнього контуру висоти в симуляції.

Символи: `class TeacherPolicy (L19)`; `validate (L31)`; `main (L49)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/generate_reflex_dataset.py`, `generator/reflex_contract.py`, `optimizer/rollout.py`, `simulation/altitude_control.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

## simulation

### `simulation/altitude_control.py`

**ACTIVE_CORE** — Канонічний NumPy altitude controller: валідність/вік телеметрії, height/vertical speed, PI-D, limits, latch fault.

Символи: `class AltitudeTelemetryError (L15)`; `class AltitudeSample (L20)`; `class AltitudeConfig (L29)`; `class AltitudeCommand (L66)`; `sample_from_observation (L73)`; `class AltitudeHold (L86)`.

Вхідні імпорти з перелічених production/CLI файлів: `edge/altitude_control.py`, `edge/validate_numpy.py`, `evaluate/validate_flight_bridge.py`, `evaluate/watch_mvp.py`, `generator/generate_reflex_dataset.py`, `optimizer/benchmark_candidate.py`, `optimizer/compare_teachers.py`, `optimizer/diagnose_candidate.py`, `optimizer/evaluate.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/replay_demonstrations.py`, `optimizer/rollout.py`, `optimizer/search_pipeline.py`, `optimizer/summarize_training_pilot.py`, `optimizer/validate_altitude.py`, `scripts/audit_tof_threats.py`, `scripts/probe_exploration_teacher.py`, `scripts/probe_tof_geometry.py`, `scripts/probe_zonal_tof.py`, `scripts/scan_probe.py`, `simulation/control_contract.py`.

### `simulation/avionics_filter.py`

**ACTIVE_EVAL** — PT1, каскад attitude/rate PID і mixer реально використовуються standalone фізикою; не зайвий hardware PID.

Символи: `class PT1Filter (L40)`; `class PIDConstants (L66)`; `class BetaflightAxisPID (L74)`; `class BetaflightCascadedPID (L112)`.

Локальні залежності (AST): `configs/flight_config.py`.

Вхідні імпорти з перелічених production/CLI файлів: `simulation/drone_env.py`.

### `simulation/benchmark_scenes.py`

**ACTIVE_EVAL** — Сім детермінованих NavigationScene для фізичного benchmark.

Символи: `class NavigationScene (L12)`.

Локальні залежності (AST): `configs/flight_config.py`, `simulation/drone_env.py`.

Вхідні імпорти з перелічених production/CLI файлів: `evaluate/validate_flight_bridge.py`, `evaluate/watch_mvp.py`, `optimizer/benchmark_candidate.py`, `optimizer/finalize_mvp_candidate.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/search_pipeline.py`, `optimizer/summarize_training_pilot.py`, `scripts/audit_tof_threats.py`, `scripts/probe_exploration_teacher.py`, `simulation/tof_challenge_scenes.py`.

### `simulation/collisions.py`

**ACTIVE_EVAL** — Swept collision і clearance оболонки корпусу для боксів/циліндрів/кімнати.

Символи: `_slab_interval (L12)`; `contact_and_clearance (L26)`.

Вхідні імпорти з перелічених production/CLI файлів: `simulation/drone_env.py`, `simulation/xy_scenes.py`.

### `simulation/control_contract.py`

**ACTIVE_CORE** — Ownership: learned Roll/Pitch/Yaw, applied Throttle належить AltitudeHold; версійний контракт.

Символи: `control_contract (L14)`.

Локальні залежності (AST): `simulation/altitude_control.py`.

Вхідні імпорти з перелічених production/CLI файлів: `edge/export_numpy.py`, `generator/generate_reflex_dataset.py`, `generator/prepare_flywire_candidates.py`, `optimizer/benchmark_candidate.py`, `optimizer/diagnose_candidate.py`, `optimizer/evaluate.py`, `optimizer/finalize_mvp_candidate.py`, `optimizer/navigation_loss.py`, `optimizer/optuna_tuner.py`, `optimizer/pretrain.py`, `optimizer/rollout.py`, `optimizer/search_pipeline.py`, `optimizer/search_recurrent_dynamics.py`.

### `simulation/drone_env.py`

**ACTIVE_EVAL** — Змішаний великий файл: standalone physics, raycasting, sensors, env, OU wrapper та optional Isaac/interactive UI.

Символи: `class IsaacGymDroneSim (L196)`; `class RoomBoundaries (L599)`; `class CylinderObstacle (L610)`; `class BoxObstacle (L619)`; `class ToFRaycaster (L633)`; `class DroneDynamicsParams (L819)`; `class QuadcopterDynamics (L841)`; `class DroneSimulationEnv (L1046)`; `_make_bar (L1521)`; `run_flight_simulation (L1527)`; `interactive_menu (L1936)`; `main (L1998)`; `class OpticalFlowOUWrapper (L2069)`.

Локальні залежності (AST): `configs/flight_config.py`, `simulation/avionics_filter.py`, `simulation/collisions.py`, `simulation/drone_interface.py`, `simulation/isaac_hud.py`, `simulation/pmw3901_emulator.py`, `simulation/policy.py`.

Вхідні імпорти з перелічених production/CLI файлів: `evaluate/watch_champion.py`, `evaluate/watch_mvp.py`, `generator/generate_reflex_dataset.py`, `optimizer/rollout.py`, `scripts/probe_tof_surfaces.py`, `scripts/probe_zonal_tof.py`, `scripts/scan_probe.py`, `simulation/benchmark_scenes.py`, `simulation/tof_challenge_scenes.py`, `simulation/xy_scenes.py`, `simulation/zonal_tof.py`.

### `simulation/drone_interface.py`

**ACTIVE_EVAL** — Змішаний файл: downward laser і sensor snapshot потрібні standalone; GUI/model adapters — відкладена гілка.

Символи: `class LaserHitResult (L32)`; `class DownwardLaserSensor (L42)`; `class IsaacObservation (L187)`; `class IsaacAction (L295)`; `class BaseIsaacModel (L366)`; `class ConnectomeModelAdapter (L379)`; `class AutonomousLaserNavigatorModel (L417)`.

Локальні залежності (AST): `configs/flight_config.py`, `simulation/policy.py`.

Вхідні імпорти з перелічених production/CLI файлів: `simulation/drone_env.py`.

### `simulation/isaac_hud.py`

**DORMANT_USEFUL** — Optional Isaac viewer/HUD; імпортується drone_env, тож просто видалити файл не можна.

Символи: `class IsaacGymHUD (L103)`.

Вхідні імпорти з перелічених production/CLI файлів: `simulation/drone_env.py`.

### `simulation/memory.py`

**DORMANT_USEFUL** — Кільцева пам'ять обчислюється rollout/collector, але її маршрути в active k306 відключені; поведінково неактивна, структурно викликається.

Символи: `class EgocentricMemoryWrapper (L54)`.

Локальні залежності (AST): `configs/flight_config.py`.

Вхідні імпорти з перелічених production/CLI файлів: `evaluate/watch_champion.py`, `generator/generate_reflex_dataset.py`, `optimizer/compare_teachers.py`, `optimizer/replay_demonstrations.py`, `optimizer/rollout.py`.

### `simulation/metrics.py`

**ACTIVE_EVAL** — Jitter, фізичні yaw bursts, coverage, energy/clearance trackers; старі PWM jump метрики окремі.

Символи: `calculate_jitter_pr (L15)`; `calculate_saccades_yaw (L33)`; `measure_yaw_bursts (L50)`; `class VoxelTracker (L86)`; `calculate_area_coverage (L134)`; `calculate_forward_ratio (L149)`; `calculate_impact_energy (L184)`; `calculate_mean_clearance (L221)`; `class PhysicsTelemetryTracker (L235)`.

Локальні залежності (AST): `configs/flight_config.py`.

Вхідні імпорти з перелічених production/CLI файлів: `optimizer/rollout.py`.

### `simulation/pmw3901_emulator.py`

**ACTIVE_EVAL** — Емуляція нормалізованого оптичного потоку, висотних/кутових ефектів і шуму; не hardware driver.

Символи: `class PMW3901FlowSensor (L28)`.

Локальні залежності (AST): `configs/flight_config.py`.

Вхідні імпорти з перелічених production/CLI файлів: `simulation/drone_env.py`.

### `simulation/policy.py`

**ACTIVE_CORE** — Host SensorInputLayer → rate core → DN head → PWM; reset/step/контракти; назва CfC у docstring історична.

Символи: `class SensorInputLayer (L107)`; `class DNProjectionHead (L199)`; `class PWMOutputLayer (L234)`; `class ChongFlyMSPPolicy (L274)`.

Локальні залежності (AST): `configs/flight_config.py`, `core/models.py`, `core/routing.py`.

Вхідні імпорти з перелічених production/CLI файлів: `bc_test.py`, `generator/prepare_flywire_candidates.py`, `generator/validate_flywire_candidates.py`, `optimizer/benchmark_candidate.py`, `optimizer/candidate_selection.py`, `optimizer/diagnose_candidate.py`, `optimizer/evaluate.py`, `optimizer/search_pipeline.py`, `optimizer/summarize_training_pilot.py`, `scripts/audit_sensor_encoding.py`, `scripts/audit_tof_threats.py`, `simulation/drone_env.py`, `simulation/drone_interface.py`.

### `simulation/policy_diagnostics.py`

**ACTIVE_EVAL** — Відкриті sensor contrasts і 12-case temporal response/recovery вимірювання.

Символи: `temporal_response_probe (L10)`; `response_probe (L69)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/generate_reflex_dataset.py`, `generator/reflex_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: `generator/validate_flywire_candidates.py`, `optimizer/benchmark_candidate.py`, `optimizer/diagnose_candidate.py`, `optimizer/probe_recurrent_dynamics.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/search_pipeline.py`.

### `simulation/tof_challenge_scenes.py`

**DORMANT_USEFUL** — Малі development сцени для геометрії ToF; не незалежний holdout.

Символи: `class ToFChallengeScene (L14)`.

Локальні залежності (AST): `simulation/benchmark_scenes.py`, `simulation/drone_env.py`.

Вхідні імпорти з перелічених production/CLI файлів: `scripts/probe_tof_geometry.py`, `scripts/probe_zonal_tof.py`.

### `simulation/tof_surfaces.py`

**DORMANT_USEFUL** — Аналітичні сфери/площини для статичних сенсорних експериментів; не collision mesh літаючої сцени.

Символи: `_vector (L13)`; `_positive (L20)`; `class Sphere (L26)`; `class PlanePatch (L36)`; `cast_surfaces (L55)`.

Вхідні імпорти з перелічених production/CLI файлів: `scripts/probe_tof_surfaces.py`.

### `simulation/xy_coordinator.py`

**ACTIVE_CORE** — Канонічний NumPy behavioral XY: anti-rollback/latch, opt-in Levy/STOP; не арбітр прав керування.

Символи: `class XYConfig (L13)`; `class BehaviorConfig (L31)`; `class XYCoordinator (L65)`.

Вхідні імпорти з перелічених production/CLI файлів: `edge/export_numpy.py`, `edge/validate_numpy.py`, `edge/xy_coordinator.py`, `evaluate/audit_xy_behaviors.py`, `evaluate/compare_levy_factorial.py`, `evaluate/compare_xy_behaviors.py`, `evaluate/compare_xy_coordinator.py`, `evaluate/validate_xy_starts.py`, `evaluate/watch_mvp.py`.

### `simulation/xy_scenes.py`

**ACTIVE_EVAL** — Парні кути й три density арени, фіксовані стартові стани.

Символи: `class DensityArena (L14)`; `class DensityStartArena (L65)`; `reflect_geometry_y (L100)`; `class CornerApproachScene (L106)`.

Локальні залежності (AST): `configs/flight_config.py`, `simulation/collisions.py`, `simulation/drone_env.py`.

Вхідні імпорти з перелічених production/CLI файлів: `evaluate/compare_levy_factorial.py`, `evaluate/compare_xy_behaviors.py`, `evaluate/validate_xy_starts.py`, `evaluate/watch_mvp.py`.

### `simulation/zonal_tof.py`

**DORMANT_USEFUL** — Експериментальні кілька променів/зона та агрегація; production ToFRaycaster не замінює.

Символи: `class ZonalConfig (L15)`; `aggregate_ranges (L34)`; `class ZonalToF (L65)`.

Локальні залежності (AST): `simulation/drone_env.py`.

Вхідні імпорти з перелічених production/CLI файлів: `scripts/probe_tof_surfaces.py`, `scripts/probe_zonal_tof.py`.

## evaluate

### `evaluate/__init__.py`

**ACTIVE_EVAL** — Маркер пакета evaluate.

Символи: немає top-level класів/функцій; див. призначення..

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `evaluate/audit_xy_behaviors.py`

**ONE_OFF_HISTORY** — Розбір збережених behavior traces, імпульсів і пристінних ділянок; повторюваний аудит серії.

Символи: `pulse_measurements (L15)`; `wall_diagnostics (L32)`; `audit (L60)`; `main (L112)`.

Локальні залежності (AST): `evaluate/compare_xy_behaviors.py`, `evaluate/watch_mvp.py`, `generator/source_contract.py`, `simulation/xy_coordinator.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `evaluate/compare_levy_factorial.py`

**ONE_OFF_HISTORY** — Заздалегідь задана Levy v2 сітка 3×4; не production координатор.

Символи: `variant_configs (L17)`; `interior_metrics (L24)`; `compare_old_control (L53)`; `run_experiment (L68)`; `main (L125)`.

Локальні залежності (AST): `evaluate/compare_xy_behaviors.py`, `evaluate/watch_mvp.py`, `generator/source_contract.py`, `simulation/xy_coordinator.py`, `simulation/xy_scenes.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `evaluate/compare_xy_behaviors.py`

**ONE_OFF_HISTORY** — Серія 3 density × 2 behaviors; summarize/plot_grid також використовуються новішими валідаторами.

Символи: `summarize (L15)`; `plot_grid (L63)`; `main (L122)`.

Локальні залежності (AST): `evaluate/watch_mvp.py`, `generator/source_contract.py`, `simulation/xy_coordinator.py`, `simulation/xy_scenes.py`.

Вхідні імпорти з перелічених production/CLI файлів: `evaluate/audit_xy_behaviors.py`, `evaluate/compare_levy_factorial.py`, `evaluate/validate_xy_starts.py`.

### `evaluate/compare_xy_coordinator.py`

**ONE_OFF_HISTORY** — Факторний експеримент 2 кути × 4 XY варіанти; рецепт і порівняльні графіки.

Символи: `summarize (L17)`; `plot_comparison (L58)`; `main (L93)`.

Локальні залежності (AST): `evaluate/watch_mvp.py`, `generator/source_contract.py`, `simulation/xy_coordinator.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `evaluate/describe_xy_passages.py`

**ACTIVE_EVAL** — Описове перетинання заданих воріт, не критерій navigation admission.

Символи: `crossing_events (L9)`; `describe (L27)`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `evaluate/validate_flight_bridge.py`

**ACTIVE_EVAL** — Чотири короткі handover сценарії; пряме застосування lease без CommandGate/мережевого транспорту.

Символи: `run (L11)`.

Локальні залежності (AST): `edge/flight_bridge.py`, `edge/runtime.py`, `simulation/altitude_control.py`, `simulation/benchmark_scenes.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `evaluate/validate_xy_starts.py`

**ACTIVE_EVAL** — Заморожені 8 стартів × 2 noise seeds, clearance/STOP admission і графіки.

Символи: `seed_probe (L35)`; `proximity_metrics (L52)`; `admission (L74)`; `plot_starts (L82)`; `run_experiment (L114)`; `main (L182)`.

Локальні залежності (AST): `configs/flight_config.py`, `evaluate/compare_xy_behaviors.py`, `evaluate/watch_mvp.py`, `generator/source_contract.py`, `simulation/xy_coordinator.py`, `simulation/xy_scenes.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `evaluate/watch_champion.py`

**DEAD_LEGACY** — Застарілий viewer: імпорт відсутнього DroneSimulationEnv з optimizer.evaluate; перебудова/донавчання замість точного checkpoint. Замінений watch_mvp.

Символи: `format_distance_bar (L79)`; `load_champion_params (L95)`; `run_light_simulation (L145)`; `run_3d_simulation (L303)`; `print_summary_report (L395)`; `main (L427)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/generate_reflex_dataset.py`, `optimizer/evaluate.py`, `optimizer/pretrain.py`, `simulation/drone_env.py`, `simulation/memory.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `evaluate/watch_mvp.py`

**ACTIVE_EVAL** — Точний checkpoint → standalone run → JSON/PNG, без прихованого донавчання.

Символи: `make_environment (L34)`; `geometry_snapshot (L53)`; `burst_positions (L59)`; `plot_summary (L70)`; `run_view (L163)`; `main (L238)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/source_contract.py`, `optimizer/benchmark_candidate.py`, `optimizer/rollout.py`, `simulation/altitude_control.py`, `simulation/benchmark_scenes.py`, `simulation/drone_env.py`, `simulation/xy_coordinator.py`, `simulation/xy_scenes.py`.

Вхідні імпорти з перелічених production/CLI файлів: `evaluate/audit_xy_behaviors.py`, `evaluate/compare_levy_factorial.py`, `evaluate/compare_xy_behaviors.py`, `evaluate/compare_xy_coordinator.py`, `evaluate/validate_xy_starts.py`.

## edge

### `edge/__init__.py`

**ACTIVE_CORE** — Маркер пакета інструментів NumPy deployment.

Символи: немає top-level класів/функцій; див. призначення..

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `edge/altitude_control.py`

**ACTIVE_CORE** — Реекспорт canonical simulation controller у репозиторії; export копіює canonical source у bundle.

Символи: немає top-level класів/функцій; див. призначення..

Локальні залежності (AST): `simulation/altitude_control.py`.

Вхідні імпорти з перелічених production/CLI файлів: `edge/runtime.py`.

### `edge/benchmark.py`

**ACTIVE_EVAL** — Час policy/runtime та пам'ять на конкретному host; не benchmark transport/Bridge/Gate.

Символи: `process_memory (L26)`; `measure (L50)`; `main (L67)`.

Локальні залежності (AST): `edge/runtime.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `edge/export_numpy.py`

**ACTIVE_CORE** — Host export frozen weights + standalone code + contracts/hash; не виконується на Pi.

Символи: `digest (L19)`; `export_policy (L23)`; `export_run (L78)`; `main (L89)`.

Локальні залежності (AST): `optimizer/benchmark_candidate.py`, `simulation/control_contract.py`, `simulation/xy_coordinator.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `edge/exporter/codegen_c.py`

**DEAD_LEGACY** — Порожній файл, 0 байтів; генератор C не реалізований.

Символи: немає top-level класів/функцій; див. призначення..

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `edge/exporter/export_onnx.py`

**DEAD_LEGACY** — Порожній файл, 0 байтів; ONNX export не реалізований.

Символи: немає top-level класів/функцій; див. призначення..

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `edge/exporter/quantizer.py`

**DEAD_LEGACY** — Порожній файл, 0 байтів; quantization не реалізована.

Символи: немає top-level класів/функцій; див. призначення..

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `edge/flight_bridge.py`

**ACTIVE_CORE** — SensorFrame/CommandLease, MANUAL/SHADOW/AUTO/FAULT admission, slew; окремий receiver CommandGate як executable specification.

Символи: `class SensorFrame (L12)`; `class CommandLease (L21)`; `class BridgeConfig (L28)`; `pwm_array (L51)`; `class FlightBridge (L58)`; `class CommandGate (L146)`.

Вхідні імпорти з перелічених production/CLI файлів: `evaluate/validate_flight_bridge.py`.

### `edge/runtime.py`

**ACTIVE_CORE** — Bundle loader/hash checks, exact float32 NumpyPolicy, FlightRuntime composition/fault/reset.

Символи: `sha256 (L25)`; `load_bundle (L29)`; `class NumpyPolicy (L69)`; `class FlightRuntime (L120)`.

Локальні залежності (AST): `edge/altitude_control.py`, `edge/xy_coordinator.py`.

Вхідні імпорти з перелічених production/CLI файлів: `edge/benchmark.py`, `edge/validate_numpy.py`, `evaluate/validate_flight_bridge.py`.

### `edge/validate_numpy.py`

**ACTIVE_EVAL** — Багатокрокове PyTorch/NumPy порівняння на однакових входах і порівняння причин XY втручань.

Символи: `validate (L19)`; `main (L76)`.

Локальні залежності (AST): `edge/runtime.py`, `optimizer/benchmark_candidate.py`, `simulation/altitude_control.py`, `simulation/xy_coordinator.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `edge/xy_coordinator.py`

**ACTIVE_CORE** — Реекспорт canonical XY у репозиторії; не друга незалежно редагована реалізація.

Символи: немає top-level класів/функцій; див. призначення..

Локальні залежності (AST): `simulation/xy_coordinator.py`.

Вхідні імпорти з перелічених production/CLI файлів: `edge/runtime.py`.

## tests

### `tests/conftest.py`

**ACTIVE_EVAL** — Спільна pytest fixture tiny_model_meta; 0 окремих тестів — очікувано.

Зібрано pytest: **0**. Група: `DATA_TRAINING_EVAL`.

Символи: `tiny_model_meta (L7)`.

Локальні залежності (AST): `tests/synthetic_graphs.py`.

### `tests/synthetic_graphs.py`

**ACTIVE_EVAL** — Tiny synthetic graph fixtures/positive controls; 0 окремих тестів — очікувано.

Зібрано pytest: **0**. Група: `DATA_TRAINING_EVAL`.

Символи: `write_tiny_model (L14)`.

Локальні залежності (AST): `generator/graph_reducer.py`.

### `tests/test_altitude_control.py`

**ACTIVE_EVAL** — Контур висоти, range/tilt, fault/reset, bounds, ownership

Зібрано pytest: **15**. Група: `CORE_CONTRACT`.

Символи: `sample (L9)`; `test_height_error_and_vertical_motion_have_opposing_feedback`; `test_mount_and_tilt_compensation_do_not_invent_height_change`; `test_bad_telemetry_revokes_autonomous_output_instead_of_guessing_throttle`; `test_duplicate_timestamp_and_abrupt_surface_jump_are_rejected`; `test_composition_changes_only_throttle_and_reset_discards_old_history`; `test_controller_is_bounded_and_recovers_after_sustained_error`; `test_adapter_uses_only_sensor_fields`; `test_fault_remains_latched_until_explicit_reset`; `test_rollout_reports_separate_control_contract_and_sensor_fault_stops_before_step`; `test_real_altitude_loop_holds_height_during_sustained_tilt_without_world_state_feedback`.

Локальні залежності (AST): `optimizer/rollout.py`, `simulation/altitude_control.py`, `simulation/drone_env.py`, `tests/test_evaluation_contract.py`.

### `tests/test_artifact_entrypoints.py`

**ACTIVE_EVAL** — Явні шляхи артефактів у ручних CLI

Зібрано pytest: **3**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_manual_entrypoints_require_explicit_artifact_paths`.

### `tests/test_behavior_cloning.py`

**ACTIVE_EVAL** — Teacher targets, dataset generation, convergence behavioral cloning

Зібрано pytest: **8**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_expert_policy_straight_flight`; `test_expert_policy_apf_smoothness`; `test_expert_policy_apf_directional_repulsion`; `test_expert_policy_apf_symmetric_braking`; `test_generate_dataset_structure_and_bounds`; `test_pretrain_policy_loss_decreases`; `test_pretrain_policy_bagging_subsets`; `test_optuna_objective_end_to_end`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/generate_reflex_dataset.py`, `optimizer/evaluate.py`, `optimizer/pretrain.py`, `simulation/policy.py`.

### `tests/test_brake_before_turn.py`

**ACTIVE_EVAL** — Гальмування до повороту й регресія зіткнення

Зібрано pytest: **6**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_fast_approach_reduces_yaw_and_strengthens_initial_braking`; `test_slowing_down_restores_escape_turn_without_changing_its_direction`; `test_turn_gating_requires_ordered_finite_flow_thresholds`; `test_last_recorded_collision_now_completes_and_recovers_motion`.

Локальні залежності (AST): `generator/generate_reflex_dataset.py`, `generator/reflex_contract.py`, `optimizer/compare_teachers.py`, `tests/test_teacher_contract.py`.

### `tests/test_candidate_diagnostics.py`

**ACTIVE_EVAL** — Research opt-in, метрики padding, native precision

Зібрано pytest: **6**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_metrics_ignore_padding_and_separate_left_and_right_turns`; `test_response_probe_uses_control_dt_and_records_actual_float32_contrast`; `test_research_probe_requires_explicit_opt_in_before_accessing_files`; `test_unapproved_artifact_cannot_enter_pretrain_without_research_opt_in`; `test_standard_factory_rejects_explicitly_unapproved_artifact`; `test_validator_smoke_reports_real_dt_and_native_precision`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/role_reducer.py`, `generator/validate_flywire_candidates.py`, `optimizer/diagnose_candidate.py`, `optimizer/evaluate.py`, `optimizer/pretrain.py`, `simulation/policy.py`, `tests/test_pretrain_contract.py`, `tests/test_role_reduction.py`.

### `tests/test_candidate_pathway.py`

**ACTIVE_EVAL** — CAVE projection, directional counts, rejection/provenance

Зібрано pytest: **17**. Група: `DATA_TRAINING_EVAL`.

Символи: `candidate_fixture (L8)`; `test_projection_is_directed_count_checked_and_thresholds_apply_to_pairs`; `test_zero_connections_is_a_valid_negative_result_with_all_targets_reported`; `test_same_root_rows_do_not_inflate_inter_neuron_connectivity`; `test_invalid_registry_or_query_cannot_start_network_access`; `test_invalid_synapse_response_is_never_published_as_evidence`; `test_cli_restores_request_handler_and_publishes_only_success`.

Локальні залежності (AST): `generator/audit_candidate_pathway.py`, `generator/published_annotations.py`, `generator/source_contract.py`.

### `tests/test_candidate_selection.py`

**ACTIVE_EVAL** — Registry verification, admission і study contract

Зібрано pytest: **6**. Група: `DATA_TRAINING_EVAL`.

Символи: `registry_fixture (L8)`; `test_pool_requires_explicit_research_mode_for_unapproved_registry`; `test_pool_uses_recomputed_admission_not_registry_claim`; `test_factory_selects_real_registry_names_and_propagates_policy_contract`; `test_search_contract_cannot_be_changed_when_resuming`; `test_old_optuna_grid_requires_a_registry`; `test_registry_artifact_changes_are_detected_before_constructing_policy`.

Локальні залежності (AST): `generator/source_contract.py`, `generator/validate_flywire_candidates.py`, `optimizer/candidate_selection.py`, `optimizer/evaluate.py`, `optimizer/optuna_tuner.py`.

### `tests/test_cave_access_probe.py`

**ACTIVE_EVAL** — CAVE доступ/версії та редагування помилок без витоку секретів

Зібрано pytest: **6**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_probe_lists_pinned_metadata_without_fetching_cells_or_saving_credentials`; `test_auth_failure_reports_stage_without_leaking_exception_contents`; `test_requested_unavailable_version_does_not_silently_switch_to_latest`; `test_discovery_does_not_require_access_to_production_or_fetch_cells`; `test_forbidden_is_distinct_from_missing_auth_and_never_leaks_error`; `test_discovery_rejects_snapshot_options`.

Локальні залежності (AST): `generator/probe_cave.py`.

### `tests/test_cave_auth_setup.py`

**ACTIVE_EVAL** — Безпечне локальне введення/зберігання credentials

Зібрано pytest: **9**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_setup_preserves_existing_token_without_prompting`; `test_setup_saves_hidden_input_to_one_native_location_without_echo`; `test_setup_rejects_empty_or_multiline_input`; `test_setup_replacement_requires_explicit_flag`; `test_hidden_input_failure_does_not_fall_back_to_visible_input`; `test_cli_refuses_noninteractive_token_input`; `test_native_auth_save_and_reload_stays_in_explicit_temporary_file`.

Локальні залежності (AST): `generator/setup_cave_auth.py`.

### `tests/test_cave_client_integration.py`

**ACTIVE_EVAL** — Реальний CAVEclient serializer/decoder на підміненому transport, не live API

Зібрано pytest: **2**. Група: `DATA_TRAINING_EVAL`.

Символи: `class ArrowPeer (L22)`; `class OfflineMaterialization (L59)`; `extract (L80)`; `test_actual_client_encodes_pinned_queries_and_decodes_arrow_counts_and_large_ids`; `test_truncation_survives_actual_client_decoding_and_is_rejected`.

Локальні залежності (AST): `generator/cave_source.py`, `generator/source_contract.py`.

### `tests/test_cfc_dynamics.py`

**DEAD_LEGACY** — Порожній файл, 0 тестів. Реальні dynamics tests знаходяться у test_neural_contract.py.

Зібрано pytest: **0**. Група: `DATA_TRAINING_EVAL`.

Символи: немає top-level класів/функцій; див. призначення..

### `tests/test_command_gate_timing.py`

**ACTIVE_EVAL** — Receiver TTL, replay, late replies та порядок викликів

Зібрано pytest: **6**. Група: `CORE_CONTRACT`.

Символи: `accept (L7)`; `test_continuous_50hz_stream_with_bounded_latency_does_not_trip`; `test_fresh_reply_after_authority_expiry_requires_reset_in_either_call_order`; `test_old_response_cannot_replace_newer_sensor_frame_even_inside_ttl`; `test_late_new_packet_does_not_hide_a_missed_watchdog_poll`.

Локальні залежності (AST): `edge/flight_bridge.py`.

### `tests/test_control_ownership.py`

**ACTIVE_EVAL** — Throttle не навчається, altitude owns applied throttle, несумісні contracts відхиляються

Зібрано pytest: **14**. Група: `CORE_CONTRACT`.

Символи: `class ConstantPolicy (L13)`; `test_throttle_targets_do_not_change_training_or_validation`; `test_validation_mse_averages_only_the_three_navigation_channels`; `test_incompatible_ownership_is_rejected_before_training`; `test_collector_records_supervised_actions_separately_from_clean_labels`; `test_sensor_fault_aborts_collection_without_writing_dataset`; `test_objective_uses_altitude_controller_and_refuses_raw_results`; `test_rollout_labels_raw_standard_and_custom_controllers_separately`; `test_applied_action_record_is_required_even_though_not_a_learning_target`.

Локальні залежності (AST): `generator/generate_reflex_dataset.py`, `optimizer/evaluate.py`, `optimizer/pretrain.py`, `optimizer/rollout.py`, `simulation/altitude_control.py`, `tests/test_evaluation_contract.py`.

### `tests/test_coordinate_contract.py`

**ACTIVE_EVAL** — FLU/RC знаки, ToF орієнтація, flow/memory координати

Зібрано pytest: **10**. Група: `CORE_CONTRACT`.

Символи: `level_env (L14)`; `test_positive_rc_roll_pitch_yaw_moves_right_forward_turns_right`; `test_native_positive_yaw_setpoint_remains_left_positive`; `test_tof_columns_are_left_to_right_and_rows_top_to_bottom`; `test_left_obstacle_hits_left_image_half_and_mirror_hits_right`; `test_flow_tracks_body_forward_and_left_at_rotated_heading`; `test_memory_adapter_moves_front_obstacle_left_after_physical_right_turn`; `test_navigation_flow_feedback_brakes_leftward_drift`; `test_isaac_state_adapter_converts_body_angular_velocity_to_world_frame`.

Локальні залежності (AST): `simulation/drone_env.py`, `simulation/drone_interface.py`, `simulation/memory.py`, `simulation/pmw3901_emulator.py`.

### `tests/test_demonstration_replay.py`

**ACTIVE_EVAL** — Точний replay епізодів і відмова при підміні даних

Зібрано pytest: **3**. Група: `DATA_TRAINING_EVAL`.

Символи: `interrupted_demonstrations (L5)`; `test_failure_replay_checks_recorded_prefix_and_retains_fault`; `test_failure_replay_refuses_changed_observations`; `test_failure_replay_refuses_unrecorded_execution_noise`.

Локальні залежності (AST): `generator/generate_reflex_dataset.py`, `optimizer/replay_demonstrations.py`, `simulation/altitude_control.py`.

### `tests/test_downward_raycast.py`

**ACTIVE_EVAL** — Downward cylinder caps та box face normals

Зібрано pytest: **4**. Група: `DATA_TRAINING_EVAL`.

Символи: `sensor (L10)`; `pitch_rotation (L14)`; `test_vertical_beam_hits_cylinder_top_before_floor`; `test_tilted_beam_crossing_top_reports_cap_not_farther_side`; `test_box_wall_hit_reports_wall_normal_not_horizontal_surface`; `test_parallel_ray_outside_box_is_not_a_hit`.

Локальні залежності (AST): `simulation/drone_env.py`, `simulation/drone_interface.py`.

### `tests/test_drone_env.py`

**ACTIVE_EVAL** — PT1/PID/mixer, physics, sensors і середовище

Зібрано pytest: **18**. Група: `DATA_TRAINING_EVAL`.

Символи: `class TestBetaflightPID (L37)`; `class TestQuadcopterDynamics (L101)`; `class TestToFRaycaster (L137)`; `class TestPMW3901FlowSensor (L179)`; `class TestDroneSimulationEnv (L209)`; `test_pt1_filter`; `test_outer_loop_angle_mode`; `test_rate_pid_anti_windup`; `test_quad_x_motor_mixer_bounds`; `test_hover_equilibrium`; `test_roll_torque_generation`; `test_quaternion_euler_roundtrip`; `test_ray_directions_count`; `test_wall_detection_straight_ahead`; `test_cylinder_obstacle_detection`; `test_stationary_flow`; `test_forward_velocity_flow`; `test_altitude_scaling`; `test_reset_shape`; `test_chong_fly_obs_helper`; `test_step_with_setpoints`; `test_step_with_pwm`; `test_closed_loop_policy_integration`.

Локальні залежності (AST): `simulation/avionics_filter.py`, `simulation/drone_env.py`, `simulation/pmw3901_emulator.py`, `simulation/policy.py`, `tests/synthetic_graphs.py`.

### `tests/test_edge_runtime.py`

**ACTIVE_EVAL** — Export/hash/parity, reset/fault, standalone без Torch

Зібрано pytest: **15**. Група: `CORE_CONTRACT`.

Символи: `exported (L20)`; `test_multistep_policy_matches_hidden_state_and_pwm`; `test_pipeline_matches_coordinator_then_altitude_and_latches_fault`; `test_bundle_detects_corruption`; `test_contract_rejects_unsupported_solver_even_with_updated_checksum`; `test_invalid_inputs_do_not_advance_neural_state`; `test_export_refuses_unsupported_encoding_and_overwrite`; `test_bundle_runs_outside_repository_without_torch_or_scipy`; `test_exported_behavior_module_matches_source_through_search_preemption_stop`; `test_edge_runtime_stop_request_is_forwarded_and_disabled_request_rejected`.

Локальні залежності (AST): `edge/export_numpy.py`, `edge/runtime.py`, `simulation/altitude_control.py`, `simulation/policy.py`, `simulation/xy_coordinator.py`, `tests/synthetic_graphs.py`.

### `tests/test_encoding_audit.py`

**ACTIVE_EVAL** — Вимірювання encoding і projected input без мутації policy

Зібрано pytest: **4**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_narrow_and_broad_encoding_matches_declared_formula`; `test_audit_measures_actual_forward_without_modifying_policy`; `test_boundary_probe_records_support_discontinuity`; `test_projection_summary_uses_clear_subtraction`.

Локальні залежності (AST): `optimizer/functional_probe.py`, `scripts/audit_sensor_encoding.py`, `simulation/policy.py`.

### `tests/test_evaluation_contract.py`

**ACTIVE_EVAL** — Control/physics clocks, termination, метрики, NaN

Зібрано pytest: **13**. Група: `CORE_CONTRACT`.

Символи: `class ScriptedEnv (L13)`; `class RecordingPolicy (L40)`; `test_control_period_holds_action_for_five_physics_steps`; `test_crash_in_middle_of_control_period_stops_immediately`; `test_hover_is_not_falsely_marked_as_a_collision`; `test_sideways_flight_is_reported_truthfully`; `test_invalid_or_nonintegral_control_period_is_rejected`; `test_nan_policy_never_reaches_physics`; `test_actual_model_storage_includes_buffers_and_active_heads`; `test_objective_aggregates_all_seeds_not_just_last`; `test_external_termination_on_last_tick_is_not_a_success`; `test_policy_programming_error_is_not_retried_as_another_call_signature`.

Локальні залежності (AST): `optimizer/evaluate.py`, `optimizer/rollout.py`, `simulation/control_contract.py`.

### `tests/test_exploration_teacher.py`

**ACTIVE_EVAL** — Research scan supervisor, preemption/brake/resume

Зібрано pytest: **17**. Група: `OPTIONAL_RESEARCH`.

Символи: `observation (L7)`; `teacher (L15)`; `advance (L20)`; `test_periodic_scan_requires_braking_then_resumes_with_ramp`; `test_due_timer_cannot_start_scan_while_flow_is_high`; `test_new_threat_immediately_preempts_scan`; `test_avoidance_requests_scan_without_waiting_for_period`; `test_obstacle_has_priority_over_periodic_scan_for_entire_episode`; `test_brake_timeout_is_explicit_instead_of_forced_scan`; `test_drift_during_scan_returns_to_braking_without_yaw`; `test_reset_replays_same_labels_and_events`; `test_invalid_configuration_fails`; `test_roughness_is_diagnostic_not_an_independent_scan_trigger`; `test_invalid_observation_does_not_advance_clock`; `test_old_teacher_factory_default_is_not_replaced`; `test_teacher_uses_shared_rollout_interface_and_altitude_controller`; `test_repeated_scans_are_measured_separately_with_endpoints_and_yaw_unwrap`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/exploration_teacher.py`, `generator/generate_reflex_dataset.py`, `generator/reflex_contract.py`, `optimizer/rollout.py`, `scripts/probe_exploration_teacher.py`, `simulation/altitude_control.py`, `simulation/benchmark_scenes.py`.

### `tests/test_filtered_candidates.py`

**ACTIVE_EVAL** — Filtered views, source cleaning, signs, bridges та candidate builds

Зібрано pytest: **33**. Група: `DATA_TRAINING_EVAL`.

Символи: `class View (L11)`; `synapses (L25)`; `test_view_count_split_is_pinned_and_exact`; `test_truncation_duplicate_ids_and_float_roots_fail`; `test_clean_is_separate_and_threshold_after_aggregation`; `test_official_filtered_view_contract_is_checked`; `test_nt_signs_are_explicit_assumptions_unknowns_fail`; `test_bridges_require_both_directed_legs_and_deterministic_budget`; `test_reduction_cannot_invent_route_by_merging_disconnected_intermediates`; `test_failed_publication_does_not_leave_ready_directory`; `test_config_cannot_mislabel_ignored_quality_or_sensor_settings`; `test_filtered_source_loader_rejects_missing_evidence_even_if_complete`; `test_candidate_hash_check_fails_before_loading_modified_weights`; `test_required_route_loss_is_reported`; `test_unique_rows_from_same_pair_remain_distinct_contacts`; `test_single_root_over_budget_cannot_be_published_as_complete`; `test_count_schema_cannot_be_a_random_id`; `test_empty_view_has_explicit_zero_count`; `test_bridge_unknown_or_unannotated_neuron_cannot_enter_by_ranking`; `test_build_real_pipeline_with_tiny_explicitly_mocked_acquisition`; `test_publication_retries_transient_windows_lock`; `test_publication_permanent_lock_is_bounded_and_leaves_no_ready_output`; `test_validator_recomputes_admission_instead_of_trusting_registry`.

Локальні залежності (AST): `generator/audit_routes.py`, `generator/filtered_candidates.py`, `generator/prepare_flywire_candidates.py`, `generator/role_reducer.py`, `generator/source_contract.py`, `generator/validate_flywire_candidates.py`, `tests/test_role_reduction.py`.

### `tests/test_flight_bridge.py`

**ACTIVE_EVAL** — Admission/authority, handover, fault, freshness, receiver reference

Зібрано pytest: **12**. Група: `CORE_CONTRACT`.

Символи: `class Runtime (L9)`; `frame (L19)`; `tick (L25)`; `ready (L29)`; `test_shadow_has_no_authority_and_auto_requires_observed_dwell`; `test_moving_entry_is_denied_without_claiming_authority`; `test_manual_override_does_not_require_healthy_sensors`; `test_fault_revokes_authority_and_requires_explicit_reset`; `test_throttle_and_attitude_command_disagreement_blocks_engagement`; `test_receiver_lease_expires_without_pi_and_rejects_replay_after_revocation`; `test_uart_buffering_does_not_extend_lease_from_old_sensor_frame`; `test_real_runtime_invalid_altitude_is_caught_and_fault_is_latched`; `test_output_throttle_slew_and_first_command_are_bounded`.

Локальні залежності (AST): `edge/flight_bridge.py`, `edge/runtime.py`, `simulation/altitude_control.py`, `tests/test_edge_runtime.py`.

### `tests/test_functional_probe.py`

**ACTIVE_EVAL** — Hooks/clamps, напрямки стимулів, recovery та незмінність model

Зібрано pytest: **14**. Група: `DATA_TRAINING_EVAL`.

Символи: `group_fixture (L107)`; `test_trace_matches_normal_policy_and_does_not_modify_it`; `test_clamp_blocks_known_relay_and_is_removed_afterwards`; `test_failed_trace_also_removes_hook`; `test_invalid_clamp_is_rejected`; `test_stimuli_are_mirrored_with_identical_clear_history`; `test_metrics_subtract_each_interventions_own_clear_response`; `test_explicit_research_opt_in_before_file_access`; `test_controls_are_reproducible_and_match_side_sign_size_without_interfaces`; `test_missing_annotations_do_not_silently_produce_noop_intervention`; `test_controls_fail_if_matching_is_impossible`; `test_no_response_is_not_reported_as_zero_latency`.

Локальні залежності (AST): `configs/flight_config.py`, `optimizer/functional_probe.py`, `simulation/policy.py`.

### `tests/test_levy_factorial.py`

**ACTIVE_EVAL** — Пауза/target-half/pulse preemption і точність report метрик

Зібрано pytest: **12**. Група: `OPTIONAL_RESEARCH`.

Символи: `step (L10)`; `test_pause_preserves_wait_but_never_launches_in_grey_zone`; `test_directed_gate_checks_minimum_as_well_as_mean_without_consuming_randomness`; `test_directed_gate_rejects_shallow_mean_and_chooses_open_left`; `test_large_pulse_retains_same_step_preemption`; `test_factor_configs_only_change_intended_parameters`; `test_rollout_records_exact_earliest_minimum_at_physics_time`; `test_interior_time_is_time_weighted_and_excludes_stop`; `test_invalid_new_configuration_rejected`.

Локальні залежності (AST): `evaluate/compare_levy_factorial.py`, `optimizer/rollout.py`, `simulation/xy_coordinator.py`, `tests/test_evaluation_contract.py`.

### `tests/test_memory.py`

**ACTIVE_EVAL** — Rotation/decay/ring behavior; не доказ підключення memory до k306

Зібрано pytest: **9**. Група: `OPTIONAL_RESEARCH`.

Символи: `test_memory_initialization`; `test_memory_wall_shift_90_deg_right`; `test_memory_wall_shift_45_deg_right`; `test_memory_wall_shift_90_deg_left`; `test_memory_decay_mechanism`; `test_memory_accumulated_yaw_sub_threshold`; `test_memory_center_tof_extraction`; `test_memory_reset`; `test_policy_sensor_input_layer_with_memory`.

Локальні залежності (AST): `simulation/memory.py`, `simulation/policy.py`.

### `tests/test_mvp_continuation.py`

**ACTIVE_EVAL** — Frozen constraints, post-step, checkpoint restore і practical admission

Зібрано pytest: **25**. Група: `DATA_TRAINING_EVAL`.

Символи: `valid_metrics (L90)`; `test_training_callback_runs_after_native_projection_on_every_step`; `test_constraints_freeze_only_requested_parameters_and_bound_actual_updates`; `test_constraints_reject_invalid_saved_bound`; `test_mvp_admission_accepts_exact_narrow_floor_without_old_gate_pass`; `test_mvp_admission_fails_closed`; `test_mvp_search_does_not_overwrite_existing_results`; `test_four_trial_pipeline_saves_baselines_frozen_checkpoints_and_separate_gates`; `test_full_core_training_keeps_zero_drive_and_reports_readout_gain`; `test_full_core_rejects_mutation_and_checkpoint_with_wrong_reference`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/reflex_contract.py`, `generator/source_contract.py`, `optimizer/benchmark_candidate.py`, `optimizer/finalize_mvp_candidate.py`, `optimizer/mvp_constraints.py`, `optimizer/pretrain.py`, `optimizer/probe_recurrent_dynamics.py`, `optimizer/rollout.py`, `simulation/benchmark_scenes.py`, `simulation/control_contract.py`, `simulation/policy.py`, `tests/synthetic_graphs.py`, `tests/test_pretrain_contract.py`.

### `tests/test_navigation_completion.py`

**ACTIVE_EVAL** — Proximity/neutral-origin, loss balance, physical yaw metrics

Зібрано pytest: **7**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_proximity_encoding_resolves_distance_and_suppresses_far_background`; `test_proximity_can_use_neutral_origin_and_has_distinct_checkpoint`; `test_amplitude_balancing_does_not_ignore_rare_strong_turns`; `test_varied_physical_demonstrations_record_scene_profile_and_remain_reproducible`; `test_physical_yaw_bursts_ignore_command_jitter_and_sustained_rotation`; `test_physical_yaw_burst_contract_rejects_bad_timebase`; `test_varied_collection_records_telemetry_failure_as_prefix_not_new_labels`.

Локальні залежності (AST): `generator/generate_reflex_dataset.py`, `optimizer/navigation_loss.py`, `simulation/altitude_control.py`, `simulation/metrics.py`, `simulation/policy.py`.

### `tests/test_navigation_learning.py`

**ACTIVE_EVAL** — Threat encoding/neutral origin, learned navigation та scene checks

Зібрано pytest: **13**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_threat_encoding_has_zero_clear_origin_and_preserves_flow_and_locations`; `test_sensor_encoding_is_guarded_by_checkpoint_contract`; `test_yaw_balancing_uses_its_targets_not_other_channels_behavior_labels`; `test_navigation_loss_ignores_padding_and_throttle_and_keeps_global_normalizer`; `test_invalid_balance_is_rejected`; `test_temporal_probe_detects_a_turn_that_never_releases`; `test_pretrain_records_explicit_channel_objective_and_train_only_group_counts`; `test_invalid_loss_configuration_fails_before_updating_parameters`; `test_neutral_origin_stays_neutral_after_updates_without_gating_recurrent_state`; `test_neutral_origin_requires_explicit_threat_encoding_and_guards_checkpoints`; `test_legacy_sequence_padding_matches_explicit_clear_memory`.

Локальні залежності (AST): `configs/flight_config.py`, `optimizer/navigation_loss.py`, `optimizer/pretrain.py`, `simulation/policy.py`, `simulation/policy_diagnostics.py`, `tests/test_pretrain_contract.py`.

### `tests/test_navigation_scenes.py`

**ACTIVE_EVAL** — Сценарії фізичних navigation challenges

Зібрано pytest: **4**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_obstacle_scenes_are_mirrored_and_survive_benchmark_reset`; `test_drift_and_fast_approach_keep_distinct_initial_conditions`; `test_checkpoint_benchmark_refuses_modified_checkpoint_before_loading`; `test_tof_ablation_preserves_real_flow_and_memory_and_resets_state`.

Локальні залежності (AST): `optimizer/benchmark_candidate.py`, `simulation/benchmark_scenes.py`.

### `tests/test_neural_contract.py`

**ACTIVE_EVAL** — Rate equations, matrix orientation, solver/time/reset/contracts

Зібрано pytest: **31**. Група: `CORE_CONTRACT`.

Символи: `isolated_cell (L11)`; `test_one_way_edge_carries_activity_from_source_to_target`; `test_chain_propagates_one_edge_per_explicit_step`; `test_inhibitory_edge_retains_sign`; `test_zero_elapsed_time_is_identity_including_state_gradient`; `test_solvers_implement_same_leaky_rate_equation`; `test_exponential_solver_is_exact_for_constant_drive_across_substeps`; `test_nonlinear_recurrent_solution_converges_as_timestep_shrinks`; `test_invalid_time_is_rejected`; `test_unknown_solver_is_not_silently_treated_as_cfc`; `test_pruning_does_not_create_trainable_edges_at_existing_zeros`; `test_checkpoint_without_dynamics_contract_is_rejected_even_non_strict`; `test_checkpoint_restores_masked_topology_and_identical_outputs`; `test_exponential_step_stays_in_state_target_envelope_for_large_dt`; `test_state_and_input_gradients_match_numerical_derivatives`; `test_versioned_checkpoint_roundtrip_through_weights_only_loader`; `test_solver_alias_does_not_override_explicit_clock`; `test_invalid_tau_is_rejected`; `test_same_shape_checkpoint_with_different_activation_is_rejected`.

Локальні залежності (AST): `core/models.py`, `generator/graph_reducer.py`.

### `tests/test_optical_flow_exploration.py`

**ACTIVE_EVAL** — OU wrapper/units/seed/reset і limits

Зібрано pytest: **8**. Група: `OPTIONAL_RESEARCH`.

Символи: `class SensorEnv (L9)`; `observation (L20)`; `test_distance_ramp_uses_metres_and_preserves_real_sensor_input`; `test_occlusion_jump_does_not_restore_stale_noise_in_one_tick`; `test_reset_replays_seed_and_does_not_advance_noise_clock`; `test_tuple_and_flat_observations_have_identical_perturbations`; `test_invalid_noise_configuration_is_rejected`.

Локальні залежності (AST): `simulation/drone_env.py`.

### `tests/test_policy.py`

**ACTIVE_EVAL** — Sensor/head/PWM композиція та step/reset

Зібрано pytest: **7**. Група: `CORE_CONTRACT`.

Символи: `policy (L13)`; `assert_pwm_bounds (L19)`; `test_pwm_ranges_and_decode`; `test_sensor_packing_and_learnable_scale`; `test_motor_head_masks_and_explicit_dense_fallback`; `test_policy_single_step_and_sequence`; `test_numpy_interface_state_carry_and_reset`; `test_post_optimizer_step_restores_forbidden_edges`; `test_gradient_flow_through_active_policy_parameters`.

Локальні залежності (AST): `simulation/policy.py`.

### `tests/test_pretrain_contract.py`

**ACTIVE_EVAL** — Версії даних, valid-prefix, split/weights/contracts

Зібрано pytest: **17**. Група: `DATA_TRAINING_EVAL`.

Символи: `class TimedPolicy (L10)`; `dataset (L21)`; `test_pretrain_uses_dataset_timebase`; `test_missing_dataset_does_not_silently_train_on_another_file`; `test_old_unversioned_demonstrations_are_not_accepted_silently`; `test_padded_targets_never_contribute_to_training`; `test_split_and_subset_keep_scenarios_and_holdout_episodes_separate`; `test_rare_behavior_has_equal_total_training_weight`; `test_impossible_teacher_route_is_rejected_before_parameter_update`; `test_invalid_new_dataset_contract_fails_before_training`; `test_invalid_dataset_fails_before_training`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/reflex_contract.py`, `optimizer/pretrain.py`, `simulation/control_contract.py`.

### `tests/test_published_annotations.py`

**ACTIVE_EVAL** — Published IDs/labels, hashes, exact source selection

Зібрано pytest: **28**. Група: `DATA_TRAINING_EVAL`.

Символи: `fixture_file (L8)`; `root_selector (L157)`; `test_published_labels_match_exactly_preserving_both_names_and_large_ids`; `test_conflicting_requested_aliases_are_not_silently_assigned`; `test_same_label_in_different_namespaces_requires_explicit_diagnostic_union`; `test_snapshot_and_bytes_are_verified_before_selection`; `test_selection_rejects_missing_side_and_duplicate_roots`; `test_hybrid_source_retains_annotation_origin_and_counted_synapses`; `test_explicit_namespace_avoids_alias_union_and_records_exact_selection`; `test_explicit_namespace_resolves_collision_without_disabling_other_guards`; `test_explicit_selectors_fail_closed_on_invalid_or_changed_identity`; `test_unused_selector_and_diagnostic_union_cannot_hide_configuration_error`; `test_author_root_selection_preserves_paper_name_and_checks_snapshot_identity`; `test_author_root_selection_rejects_unverified_or_changed_annotations`; `test_author_roots_do_not_duplicate_one_neuron_under_two_names`.

Локальні залежності (AST): `generator/cave_source.py`, `generator/published_annotations.py`, `generator/source_contract.py`.

### `tests/test_recurrent_dynamics_probe.py`

**ACTIVE_EVAL** — Copy-only tau/W scaling та bounded search restrictions

Зібрано pytest: **11**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_scaling_is_copy_only_and_preserves_graph_and_neural_time_constants`; `test_invalid_or_below_floor_scaling_is_rejected`; `test_selection_rejects_fast_but_weak_or_wrong_turns`; `test_search_refuses_overwriting_results_before_reading_inputs`; `test_search_requires_matching_unchanged_screen_before_loading_data`.

Локальні залежності (AST): `optimizer/probe_recurrent_dynamics.py`, `optimizer/search_pipeline.py`, `optimizer/search_recurrent_dynamics.py`, `simulation/policy.py`.

### `tests/test_reduction_contract.py`

**ACTIVE_EVAL** — Directed aggregation/normalization, legacy migration

Зібрано pytest: **6**. Група: `DATA_TRAINING_EVAL`.

Символи: `write_legacy (L38)`; `test_reduction_equals_lift_propagate_then_average_target_population`; `test_saved_graph_declares_orientation_and_normalization`; `test_known_legacy_graph_converts_without_rewriting_original`; `test_unknown_orientation_is_not_guessed`.

Локальні залежності (AST): `generator/graph_reducer.py`.

### `tests/test_reflex_training_bundle.py`

**ACTIVE_EVAL** — Labels/provenance цілісність training bundle

Зібрано pytest: **2**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_bundle_audit_checks_every_label_and_rejects_corruption`; `test_current_gate_rejects_historical_teacher_reference`.

Локальні залежності (AST): `generator/generate_reflex_dataset.py`, `generator/prepare_reflex_training.py`, `optimizer/search_pipeline.py`.

### `tests/test_role_reduction.py`

**ACTIVE_EVAL** — Ролі/сторони/знаки, input-output nodes, reproducible reduction

Зібрано pytest: **19**. Група: `DATA_TRAINING_EVAL`.

Символи: `graph (L11)`; `test_every_budget_preserves_roles_sides_signs_and_individual_interface_neurons`; `test_impossible_budget_is_rejected_instead_of_merging_protected_roles`; `test_raw_aggregation_equals_lift_propagate_and_average_with_signed_directed_edges`; `test_order_of_dataframe_rows_does_not_relabel_neurons`; `test_normalization_is_recorded_and_does_not_change_edge_directions_or_ratios`; `test_malformed_source_graph_is_rejected_before_aggregation`; `test_graph_fingerprint_changes_when_wiring_or_mapping_changes`; `test_policy_factory_can_load_role_preserving_graph_and_report_its_source`; `test_default_optuna_candidates_do_not_return_to_collapsed_spectral_models`; `test_default_missing_artifacts_explain_regeneration`.

Локальні залежності (AST): `core/routing.py`, `generator/graph_reducer.py`, `generator/role_reducer.py`, `optimizer/evaluate.py`.

### `tests/test_route_audit.py`

**ACTIVE_EVAL** — Source gap проти reduction loss/invented route

Зібрано pytest: **4**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_audit_distinguishes_source_gap_from_reduction_loss_and_invented_route`; `test_audit_refuses_to_compare_artifacts_from_a_different_source`; `test_missing_route_yaw_is_invariant_to_tof_with_identical_flow_history`.

Локальні залежності (AST): `generator/audit_routes.py`, `generator/role_reducer.py`, `simulation/policy.py`, `tests/synthetic_graphs.py`, `tests/test_role_reduction.py`.

### `tests/test_routing_contract.py`

**ACTIVE_EVAL** — Mask enforcement, no hidden dense paths, disconnected memory

Зібрано pytest: **30**. Група: `CORE_CONTRACT`.

Символи: `cell (L10)`; `test_sensor_perturbation_reaches_only_declared_population`; `test_no_edges_means_no_cross_neuron_influence_without_disabling_backbone`; `test_unmapped_memory_cannot_become_a_hidden_control_path`; `test_memory_route_requires_explicit_receivers`; `test_unreachable_neuron_stays_unaffected_after_optimizer_updates`; `test_bad_sensor_mapping_fails_closed`; `test_free_connectivity_requires_explicit_unconstrained_mode`; `test_missing_or_invalid_motor_mapping_never_enables_dense_fallback`; `test_motor_masks_block_unauthorized_state_influence`; `test_generic_network_motor_head_obeys_same_mapping`; `test_policy_does_not_store_unused_dense_motor_head`; `test_routing_change_rejects_same_shape_checkpoint`; `test_motor_routing_change_rejects_same_shape_checkpoint`; `test_legacy_pitch_roll_mapping_reports_lost_output_dimensions`; `test_dense_baseline_remains_explicitly_available`; `test_routed_policy_roundtrip_restores_outputs_without_learning_new_routes`; `test_sensor_signal_reaches_motor_side_only_along_directed_path`; `test_reachability_distinguishes_missing_paths_from_full_motor_rank`; `test_policy_reports_paths_after_removing_an_edge`; `test_sparse_path_diagnostics_match_dense_and_ignore_stored_zeros`; `test_checkpoint_with_previous_coordinate_contract_is_rejected`.

Локальні залежності (AST): `configs/flight_config.py`, `core/models.py`, `core/routing.py`, `simulation/policy.py`.

### `tests/test_scan_probe.py`

**ACTIVE_EVAL** — Scripted maneuver/metrics, brake prerequisite і phase logs

Зібрано pytest: **8**. Група: `OPTIONAL_RESEARCH`.

Символи: `test_scan_request_is_rejected_if_braking_did_not_settle_motion`; `test_settled_scan_retains_flow_damping_and_resumes_afterwards`; `test_metrics_measure_max_excursion_not_only_final_displacement_and_unwrap_yaw`; `test_absent_scan_metrics_are_missing_not_zero_success`; `test_invalid_probe_settings_are_rejected`; `test_real_probe_logs_applied_throttle_and_all_phases`.

Локальні залежності (AST): `scripts/scan_probe.py`.

### `tests/test_search_pipeline.py`

**ACTIVE_EVAL** — Registry study/run/resume/checkpoints і research gate

Зібрано pytest: **11**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_search_rejects_unverifiable_demonstration_timebase_and_seeds`; `test_functional_gate_accepts_complete_current_engineering_evidence`; `test_functional_admission_rejects_constant_turn_and_hover_even_without_crash`; `test_functional_admission_requires_all_scene_evidence_and_current_benchmark`; `test_search_configuration_rejects_unknown_fields_and_impossible_device`; `test_registry_search_exports_reloadable_weights_and_resumes_local_database`; `test_checkpoint_rejects_changed_altitude_controller_even_with_matching_file_hash`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/generate_reflex_dataset.py`, `generator/reflex_contract.py`, `generator/source_contract.py`, `optimizer/benchmark_candidate.py`, `optimizer/candidate_selection.py`, `optimizer/rollout.py`, `optimizer/search_pipeline.py`, `simulation/benchmark_scenes.py`, `simulation/control_contract.py`, `tests/test_candidate_selection.py`.

### `tests/test_simulation_contract.py`

**ACTIVE_EVAL** — Swept/body collisions, pure sensor reads, physics validity

Зібрано pytest: **12**. Група: `CORE_CONTRACT`.

Символи: `make_env (L8)`; `test_inside_obstacle_is_a_crash`; `test_swept_collision_catches_crossing_with_both_endpoints_clear`; `test_propeller_envelope_hits_without_center_penetration`; `test_safe_flight_above_short_obstacle_does_not_collide`; `test_room_lower_bounds_are_not_assumed_symmetric`; `test_observation_reads_do_not_advance_sensors_or_odometry`; `test_bad_action_is_rejected_before_state_mutation`; `test_reset_repeats_sensor_stream_even_after_another_episode`.

Локальні залежності (AST): `simulation/drone_env.py`.

### `tests/test_source_contract.py`

**ACTIVE_EVAL** — IDs, query completeness, immutable bundles/provenance

Зібрано pytest: **30**. Група: `DATA_TRAINING_EVAL`.

Символи: `tables (L10)`; `class FakeMaterialize (L39)`; `cave_result (L63)`; `csv_parquet_stub (L189)`; `test_source_rejects_ambiguous_ids_and_incomplete_graphs`; `test_large_ids_are_preserved_exactly_and_input_is_not_mutated`; `test_cave_pins_every_query_splits_by_filter_and_applies_exclusions`; `test_truncated_cave_response_fails_instead_of_publishing_a_partial_graph`; `test_single_filter_over_budget_fails_without_unbounded_download`; `test_count_probe_is_bounded_when_remote_ignores_count_flag`; `test_count_frame_must_have_count_column_not_an_arbitrary_single_integer`; `test_query_order_does_not_change_neuron_indexing_or_edges`; `test_duplicate_synapses_are_rejected_before_count_aggregation`; `test_cave_error_never_runs_synthetic_generator_or_touches_existing_output`; `test_extractor_requires_explicit_source`; `test_missing_cave_version_fails_before_network_access`; `test_synthetic_generator_is_seeded_without_changing_global_random_state`; `test_complete_bundle_roundtrip_hashes_and_large_ids`; `test_parquet_failure_never_publishes_half_of_a_bundle`; `test_bundle_refuses_to_overwrite_existing_data`; `test_tampered_source_rejected_even_with_allow_legacy`; `test_loader_requires_explicit_legacy_optin_and_never_drops_bad_edges`; `test_reducer_inherits_acquisition_and_refuses_a_source_kind_override`; `test_unknown_cell_type_cannot_silently_become_excitatory`; `test_invalid_polarity_override_is_rejected`; `test_publish_retries_transient_windows_rename_lock_without_copying`.

Локальні залежності (AST): `generator/cave_source.py`, `generator/circuit_extractor.py`, `generator/graph_reducer.py`, `generator/role_reducer.py`, `generator/source_contract.py`, `tests/test_role_reduction.py`.

### `tests/test_spatial_calibration.py`

**ACTIVE_EVAL** — Experimental mean-proximity encoding і mirrored temporal stimuli

Зібрано pytest: **8**. Група: `OPTIONAL_RESEARCH`.

Символи: `test_mean_proximity_keeps_distance_mass_across_obstacle_coverage`; `test_mean_proximity_has_finite_clear_and_single_pixel_response`; `test_mean_encoding_keeps_neutral_origin_and_cannot_load_other_encoding`; `test_temporal_probe_covers_mirrored_widths_without_changing_default`; `test_temporal_probe_rejects_ambiguous_coverage`.

Локальні залежності (AST): `simulation/policy.py`, `simulation/policy_diagnostics.py`.

### `tests/test_study_contract.py`

**ACTIVE_EVAL** — Feasible Pareto, experiment identity, explicit CUDA та registry

Зібрано pytest: **17**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_safe_frontier_excludes_smaller_smooth_crash`; `test_no_feasible_candidate_means_empty_frontier`; `test_legacy_study_is_rejected_without_relabelling`; `test_public_tuner_delegates_to_verified_registry_pipeline`; `test_registry_frontier_rejects_missing_or_different_experiment_evidence`; `test_explicit_cuda_request_cannot_silently_train_on_cpu`; `test_objective_publishes_constraints_to_optuna`; `test_resume_with_different_seed_is_rejected`; `test_previous_neural_dynamics_cannot_resume_in_new_benchmark`; `test_dense_bypass_benchmark_cannot_resume_after_routing_change`; `test_collapsed_motor_benchmark_cannot_resume_after_reduction_change`; `test_previous_coordinate_benchmark_is_rejected`; `test_previous_raw_and_assisted_scores_cannot_enter_navigation_frontier`; `test_navigation_frontier_requires_matching_controller`; `test_study_with_changed_controller_is_rejected_on_resume`.

Локальні залежності (AST): `optimizer/evaluate.py`, `optimizer/optuna_tuner.py`, `optimizer/rollout.py`, `optimizer/search_pipeline.py`, `simulation/control_contract.py`.

### `tests/test_synapse_sign_constraint.py`

**ACTIVE_EVAL** — Sign preservation після updates і checkpoint restore

Зібрано pytest: **5**. Група: `CORE_CONTRACT`.

Символи: `cell (L9)`; `test_optimizer_cannot_reverse_initial_signs_and_zero_edges_can_regrow`; `test_sign_policy_is_an_explicit_checkpoint_contract`; `test_policy_factory_preserves_declared_signs`; `test_free_connectivity_cannot_claim_preserved_signs`; `test_signed_checkpoint_cannot_change_the_initial_sign_reference`.

Локальні залежності (AST): `core/models.py`, `simulation/policy.py`.

### `tests/test_synthetic_examples.py`

**ACTIVE_EVAL** — Малі явно synthetic graphs та route positive controls

Зібрано pytest: **3**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_tiny_example_roundtrip_and_directed_routes`; `test_missing_yaw_route_has_a_working_positive_control`.

Локальні залежності (AST): `generator/graph_reducer.py`, `simulation/policy.py`, `tests/synthetic_graphs.py`.

### `tests/test_teacher_braking.py`

**ACTIVE_EVAL** — Teacher braking hysteresis, release, historical version replay

Зібрано pytest: **12**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_losing_obstacle_does_not_release_brake_while_motion_continues`; `test_brake_release_requires_sustained_low_motion_and_recovers_cruise`; `test_braking_does_not_push_backwards_after_forward_motion_stops`; `test_brake_settling_uses_elapsed_time`; `test_versioned_factory_retains_historical_teacher_for_exact_replay`; `test_invalid_braking_parameters_are_rejected`; `test_historical_demonstrations_cannot_silently_train_current_teacher_contract`.

Локальні залежності (AST): `generator/generate_reflex_dataset.py`, `generator/reflex_contract.py`, `optimizer/pretrain.py`, `tests/test_teacher_contract.py`.

### `tests/test_teacher_comparison.py`

**ACTIVE_EVAL** — Matched teacher comparisons і recorded regression cases

Зібрано pytest: **14**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_comparison_reuses_recorded_scene_seeds_and_runs_both_versions`; `test_comparison_rejects_invalid_horizon`; `test_subset_comparison_retains_original_episode_indices_and_previous_version`; `test_invalid_comparison_subsets_are_rejected`; `test_recorded_collision_is_prevented_after_obstacle_leaves_view`.

Локальні залежності (AST): `generator/generate_reflex_dataset.py`, `generator/reflex_contract.py`, `optimizer/compare_teachers.py`.

### `tests/test_teacher_contract.py`

**ACTIVE_EVAL** — Валідність/версії teacher observations/labels та sensor timebase

Зібрано pytest: **12**. Група: `DATA_TRAINING_EVAL`.

Символи: `observation (L9)`; `test_teacher_default_range_matches_simulator`; `test_symmetric_obstacle_produces_repeatable_escape_without_random_labels`; `test_latched_turn_survives_a_single_opposite_or_clear_frame_then_releases`; `test_flow_feedback_opposes_forward_and_lateral_drift`; `test_invalid_observations_are_not_replaced_with_clear_space`; `test_dataset_has_balanced_geometric_scenarios_clean_labels_and_timebase`; `test_generator_reproducible_and_rejects_nonintegral_control_period`; `test_generator_stops_at_termination_instead_of_resetting_inside_sequence`; `test_execution_noise_does_not_contaminate_teacher_labels`; `test_episode_scene_seeds_are_independent_of_execution_horizon`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/generate_reflex_dataset.py`.

### `tests/test_teacher_v5.py`

**ACTIVE_EVAL** — Versioned v5 selection, replay, rejection старої aggregation

Зібрано pytest: **5**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_factory_preserves_v4_and_selects_repaired_v5`; `test_default_collection_labels_replay_with_v5_and_differ_from_v4`; `test_explicit_historical_collection_retains_its_version`; `test_pretrain_rejects_old_teacher_and_wrong_aggregation_before_updates`; `test_temporal_probe_records_and_uses_v5_reference`.

Локальні залежності (AST): `generator/continuous_reflex_teacher.py`, `generator/generate_reflex_dataset.py`, `generator/reflex_contract.py`, `optimizer/pretrain.py`, `simulation/policy_diagnostics.py`, `tests/test_pretrain_contract.py`.

### `tests/test_tof_geometry.py`

**ACTIVE_EVAL** — Front ToF raycast caps/slabs/occlusion і thin targets

Зібрано pytest: **11**. Група: `DATA_TRAINING_EVAL`.

Символи: `ray (L8)`; `test_vertical_ray_hits_cylinder_top_before_floor`; `test_oblique_ray_hits_cylinder_top_before_far_side`; `test_parallel_box_ray_inside_or_on_slab_has_a_finite_hit`; `test_parallel_box_ray_outside_slab_does_not_create_a_phantom_hit`; `test_nearer_box_occludes_cylinder_cap`; `test_point_rays_can_miss_a_thin_centered_object`; `test_challenge_scenes_have_reproducible_mirrored_visibility`; `test_rollout_records_new_geometry_and_excludes_old_benchmark_version`; `test_demonstrations_record_sensor_geometry_version`.

Локальні залежності (AST): `generator/generate_reflex_dataset.py`, `optimizer/rollout.py`, `scripts/audit_tof_threats.py`, `scripts/probe_tof_geometry.py`, `simulation/altitude_control.py`, `simulation/drone_env.py`, `simulation/tof_challenge_scenes.py`.

### `tests/test_tof_surfaces.py`

**ACTIVE_EVAL** — Analytic planes/spheres intersections та invalid geometry

Зібрано pytest: **15**. Група: `OPTIONAL_RESEARCH`.

Символи: `test_sphere_near_root_tangent_miss_and_inside_exit`; `test_patch_is_finite_two_sided_and_parallel_safe`; `test_tilted_plane_matches_plane_equation`; `test_occlusion_order_and_range_cap`; `test_rejects_invalid_sphere_radius`; `test_rejects_invalid_patch_basis`; `test_rejects_nonunit_or_nonfinite_rays`; `test_front_patch_matches_existing_box_front_and_rotation_is_invariant`; `test_small_patch_can_be_sampled_then_rejected_by_support`; `test_probe_distinguishes_missing_samples_from_discarded_near_ranges`; `test_generated_scenes_have_a_real_near_surface_point_in_field_of_view`.

Локальні залежності (AST): `scripts/probe_tof_surfaces.py`, `simulation/drone_env.py`, `simulation/tof_surfaces.py`, `simulation/zonal_tof.py`.

### `tests/test_tof_threat_contract.py`

**ACTIVE_EVAL** — Continuous aggregation, input boundaries і monotonic threat response

Зібрано pytest: **15**. Група: `DATA_TRAINING_EVAL`.

Символи: `observation (L11)`; `crossing_pair (L21)`; `current_action (L27)`; `continuous_action (L31)`; `test_weak_extra_ray_does_not_weaken_braking`; `test_force_summary_is_continuous_when_support_changes`; `test_uniform_single_obstacle_retains_existing_teacher_behavior`; `test_adding_closer_obstacle_cannot_reduce_initial_braking`; `test_new_teacher_preserves_high_flow_turn_suppression_and_reset`; `test_local_encoding_does_not_rescale_existing_pixels_at_threshold`; `test_local_encoding_is_monotone_bounded_and_keeps_spatial_coverage`; `test_local_encoder_cannot_silently_load_mean_checkpoint`; `test_old_teacher_is_numerically_unchanged_at_audited_boundary`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/continuous_reflex_teacher.py`, `generator/generate_reflex_dataset.py`, `simulation/policy.py`.

### `tests/test_training_coverage.py`

**ACTIVE_EVAL** — Mask/split correctness, absent-data semantics

Зібрано pytest: **4**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_padding_is_excluded_and_disconnected_frames_are_not_a_long_episode`; `test_coarse_column_coverage_does_not_imply_full_height_probe_pattern`; `test_no_matching_frames_produces_no_fake_zero_target`; `test_duplicate_or_overlapping_split_is_rejected`.

Локальні залежності (AST): `scripts/audit_training_coverage.py`.

### `tests/test_watch_mvp.py`

**ACTIVE_EVAL** — Trace/physics timing, applied throttle, checkpoint identity і no-training viewer

Зібрано pytest: **6**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_trace_preserves_rollout_results_and_both_timebases`; `test_trace_stops_at_actual_collision_tick`; `test_trace_records_altitude_applied_throttle_at_control_time`; `test_viewer_environment_is_standalone_and_ou_window_uses_metres`; `test_burst_marker_uses_physical_event_time_not_pwm_jump`; `test_viewer_exports_exact_loaded_model_without_training`.

Локальні залежності (AST): `evaluate/watch_mvp.py`, `optimizer/rollout.py`, `simulation/altitude_control.py`, `simulation/benchmark_scenes.py`, `simulation/policy.py`, `tests/synthetic_graphs.py`, `tests/test_evaluation_contract.py`.

### `tests/test_xy_behaviors.py`

**ACTIVE_EVAL** — Levy/STOP/preemption, seed/reset і priority

Зібрано pytest: **16**. Група: `CORE_CONTRACT`.

Символи: `step (L8)`; `active_search (L14)`; `test_levy_is_seeded_bounded_and_reset_replays`; `test_search_is_preempted_in_same_step_by_range_or_network`; `test_search_waits_for_clear_space_and_chooses_correct_image_side`; `test_stop_cancels_search_damps_both_axes_and_yields_to_critical_threat`; `test_stop_is_opt_in_and_invalid_request_does_not_advance_state`; `test_latch_blocks_search_until_its_clear_dwell_expires`; `test_search_diagnostics_distinguish_wait_reset_from_pulse_preemption`; `test_search_diagnostics_report_simultaneous_latch_range_and_reflex_blocks`; `test_stop_metrics_distinguish_path_from_displacement_and_initial_acceleration`; `test_stop_schedule_uses_control_time_and_keeps_five_physics_substeps`; `test_stop_metric_cannot_pass_on_a_lucky_final_sample_or_early_termination`; `test_density_scenes_have_measured_clear_passages_and_clear_start`.

Локальні залежності (AST): `evaluate/compare_xy_behaviors.py`, `optimizer/rollout.py`, `simulation/xy_coordinator.py`, `simulation/xy_scenes.py`, `tests/test_evaluation_contract.py`.

### `tests/test_xy_coordinator.py`

**ACTIVE_EVAL** — Anti-rollback/latch, disabled identity, near-zero preset

Зібрано pytest: **13**. Група: `CORE_CONTRACT`.

Символи: `command (L7)`; `apply (L11)`; `test_anti_rollback_preserves_braking_at_speed_and_unrelated_channels`; `test_anti_rollback_is_smooth_between_stop_and_forward_flow`; `test_turn_latch_preserves_entire_pulse_envelope_and_neutral`; `test_latch_requires_threat_and_clear_dwell_before_choosing_new_sign`; `test_disabled_coordinator_is_identity`; `test_near_zero_window_keeps_forward_braking_and_ignores_small_yaw_for_acquisition`; `test_rollout_records_network_and_applied_commands_separately`; `test_arena_mirror_reflects_geometry_and_preserves_sensor_frame`; `test_corner_pair_mirrors_pose_velocity_and_actual_tof_columns`; `test_factorial_metrics_measure_reverse_travel_and_sustained_corner_exit`; `test_invalid_dt_rejected`.

Локальні залежності (AST): `evaluate/compare_xy_coordinator.py`, `evaluate/watch_mvp.py`, `optimizer/rollout.py`, `simulation/xy_coordinator.py`, `tests/test_evaluation_contract.py`.

### `tests/test_xy_passages.py`

**ACTIVE_EVAL** — Interpolated crossing і відокремлення STOP від описового passage

Зібрано pytest: **2**. Група: `DATA_TRAINING_EVAL`.

Символи: `rows (L5)`; `test_gate_crossing_interpolates_time_and_does_not_count_twice_at_vertex`; `test_infinite_line_crossing_outside_gate_or_during_stop_is_not_passage`.

Локальні залежності (AST): `evaluate/describe_xy_passages.py`.

### `tests/test_xy_starts.py`

**ACTIVE_EVAL** — Frozen starts/noise, hazard durations і admission

Зібрано pytest: **5**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_starts_are_fixed_valid_and_seed_changes_sensors_not_start_state`; `test_hazard_episodes_count_duration_and_longest_without_merging_gaps`; `test_admission_does_not_pass_collision_truncated_stop_or_long_close_approach`; `test_custom_start_rejected_for_non_density_scene`; `test_original_density_start_remains_unchanged`.

Локальні залежності (AST): `evaluate/validate_xy_starts.py`, `evaluate/watch_mvp.py`, `simulation/xy_scenes.py`.

### `tests/test_yaw_recovery.py`

**ACTIVE_EVAL** — Causal recovery labels, loss balance, carryover/reset та constraints

Зібрано pytest: **7**. Група: `DATA_TRAINING_EVAL`.

Символи: `test_recovery_labels_are_causal_episode_local_and_ignore_padding`; `test_recovery_weighting_preserves_mass_and_only_changes_yaw`; `test_probe_separates_carried_state_from_clear_input_and_teacher_release`; `test_recovery_mode_preserves_split_and_records_time_contract`; `test_continuation_checks_anatomical_mask_not_previous_zero_weights`; `test_absent_recovery_group_does_not_change_existing_weights`.

Локальні залежності (AST): `optimizer/navigation_loss.py`, `optimizer/pretrain.py`, `optimizer/probe_yaw_recovery.py`, `optimizer/search_pipeline.py`, `simulation/policy.py`, `tests/test_pretrain_contract.py`.

### `tests/test_zonal_tof.py`

**ACTIVE_EVAL** — Aggregator semantics, valid samples, mirror/rotation і research-only contract

Зібрано pytest: **20**. Група: `OPTIONAL_RESEARCH`.

Символи: `test_mixed_zone_mean_minimum_and_supported_group_are_distinct`; `test_group_diameter_is_anchored_not_chained`; `test_no_hit_stays_max_range_without_invented_support`; `test_no_return_counts_against_coverage_and_mean_includes_its_range_cap`; `test_invalid_samples_are_not_converted_to_free_space`; `test_invalid_configuration_fails`; `test_sampling_shape_and_mirror_symmetry_and_legacy_is_unchanged`; `test_minimum_sees_thin_center_with_sufficient_sampling_but_mean_dilutes_it`; `test_supported_rule_has_an_explicit_coverage_boundary`; `test_zonal_rotation_uses_the_shared_world_transform`; `test_experiment_pairs_physics_but_labels_custom_sensor_results_separately`.

Локальні залежності (AST): `optimizer/rollout.py`, `scripts/probe_zonal_tof.py`, `simulation/drone_env.py`, `simulation/zonal_tof.py`.

## scripts

### `scripts/audit_sensor_encoding.py`

**ACTIVE_EVAL** — Hooks на реальному forward, перевірка encoding/input projection без зміни checkpoint.

Символи: `boundary_inputs (L23)`; `inspect_pipeline (L33)`; `run_audit (L91)`; `main (L175)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/prepare_flywire_candidates.py`, `generator/source_contract.py`, `optimizer/benchmark_candidate.py`, `optimizer/functional_probe.py`, `simulation/policy.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `scripts/audit_tof_threats.py`

**ONE_OFF_HISTORY** — Арифметика зональної агрегації, teacher v4/experimental comparison; передумова v5.

Символи: `band (L31)`; `boundary_observations (L40)`; `arithmetic_audit (L48)`; `class TeacherAdapter (L100)`; `physical_comparison (L117)`; `plot_audit (L149)`; `main (L178)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/continuous_reflex_teacher.py`, `generator/generate_reflex_dataset.py`, `generator/prepare_flywire_candidates.py`, `generator/reflex_contract.py`, `generator/source_contract.py`, `optimizer/rollout.py`, `simulation/altitude_control.py`, `simulation/benchmark_scenes.py`, `simulation/policy.py`.

Вхідні імпорти з перелічених production/CLI файлів: `optimizer/summarize_training_pilot.py`, `scripts/probe_tof_geometry.py`.

### `scripts/audit_training_coverage.py`

**ACTIVE_EVAL** — Coverage exact dataset/split, без підміни labels і без уявного покриття відсутніх кадрів.

Символи: `validate_split (L19)`; `summarize_mask (L26)`; `condition_masks (L43)`; `run_audit (L76)`; `main (L197)`.

Локальні залежності (AST): `generator/generate_reflex_dataset.py`, `generator/prepare_flywire_candidates.py`, `generator/source_contract.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `scripts/probe_exploration_teacher.py`

**DORMANT_USEFUL** — Фізичний probe scan teacher; додаткова поведінка поки не ввімкнена в навчання MVP.

Символи: `class RecordingScene (L20)`; `scan_segments (L46)`; `run_probe (L70)`; `plot_trace (L100)`; `main (L126)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/exploration_teacher.py`, `generator/prepare_flywire_candidates.py`, `generator/source_contract.py`, `optimizer/rollout.py`, `simulation/altitude_control.py`, `simulation/benchmark_scenes.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `scripts/probe_filtered783.py`

**ONE_OFF_HISTORY** — Bounded probe schema офіційних filtered views.

Символи: `main (L10)`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `scripts/probe_tof_geometry.py`

**DORMANT_USEFUL** — Видимість/геометрія і поведінка в development ToF сценах.

Символи: `class VisibilityAdapter (L21)`; `visibility (L34)`; `run_case (L45)`; `plot_visibility (L75)`; `main (L90)`.

Локальні залежності (AST): `generator/prepare_flywire_candidates.py`, `generator/source_contract.py`, `optimizer/rollout.py`, `scripts/audit_tof_threats.py`, `simulation/altitude_control.py`, `simulation/tof_challenge_scenes.py`.

Вхідні імпорти з перелічених production/CLI файлів: `scripts/probe_zonal_tof.py`.

### `scripts/probe_tof_surfaces.py`

**DORMANT_USEFUL** — Статичний експеримент поверхонь/кривизни, без flight policy.

Символи: `scenes (L28)`; `observation_metrics (L52)`; `evaluate (L62)`; `summarize (L80)`; `plots (L109)`; `main (L160)`.

Локальні залежності (AST): `generator/prepare_flywire_candidates.py`, `generator/source_contract.py`, `simulation/drone_env.py`, `simulation/tof_surfaces.py`, `simulation/zonal_tof.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `scripts/probe_zonal_tof.py`

**DORMANT_USEFUL** — Порівняння subray aggregators, static та physical cases, без зміни production sensor.

Символи: `make_sensor (L36)`; `sensor_contract (L48)`; `static_sweep (L56)`; `physical_case (L81)`; `plot_results (L119)`; `main (L148)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/continuous_reflex_teacher.py`, `generator/prepare_flywire_candidates.py`, `generator/source_contract.py`, `optimizer/rollout.py`, `scripts/probe_tof_geometry.py`, `simulation/altitude_control.py`, `simulation/drone_env.py`, `simulation/tof_challenge_scenes.py`, `simulation/zonal_tof.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `scripts/scan_probe.py`

**ONE_OFF_HISTORY** — Scripted move/brake/scan/resume reference; це не learned policy і не replay k306.

Символи: `class ProbeConfig (L26)`; `class ScanProbePolicy (L42)`; `class RecordingEnv (L78)`; `phase_metrics (L109)`; `run_probe (L129)`; `plot_trace (L165)`; `main (L203)`.

Локальні залежності (AST): `configs/flight_config.py`, `generator/prepare_flywire_candidates.py`, `generator/source_contract.py`, `optimizer/rollout.py`, `simulation/altitude_control.py`, `simulation/drone_env.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

## additional

### `config.py`

**DORMANT_USEFUL** — Кореневий compatibility реекспорт configs.flight_config; не друга конфігурація.

Символи: немає top-level класів/функцій; див. призначення..

Локальні залежності (AST): `configs/flight_config.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `bc_test.py`

**DORMANT_USEFUL** — Ручний pretrain CLI з явними meta/dataset; імпорт сам не запускає навчання.

Символи: `main (L5)`.

Локальні залежності (AST): `optimizer/pretrain.py`, `simulation/policy.py`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `test_imports.py`

**DEAD_LEGACY** — Небезпечний import-everything на рівні модуля; може імпортувати scratch mutation script. Не запускати root pytest до карантину цього механізму.

Символи: немає top-level класів/функцій; див. призначення..

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `scratch_apply_gpu.py`

**ONE_OFF_HISTORY** — Одноразовий regex patcher, що записує код при імпорті; кандидат на карантин/видалення після архівації.

Символи: `patch_file (L4)`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `benchmarks/check_spectral.py`

**DORMANT_USEFUL** — Спектральний радіус файлів старого каталогу; корисна операція, CLI потребує явних шляхів і контролю densification.

Символи: `compute_spectral_radius (L6)`; `main (L14)`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.

### `benchmarks/hardware_benchmark.py`

**DEAD_LEGACY** — Proxy старого CfC/MLP з випадковими вагами й незавершеною функцією; не benchmark поточної exported rate-model.

Символи: `cfc_step_numpy (L7)`; `class ProxyBenchmark (L24)`; `run_benchmark (L60)`.

Вхідні імпорти з перелічених production/CLI файлів: не знайдено статично; можливий CLI/export/динамічний імпорт.
