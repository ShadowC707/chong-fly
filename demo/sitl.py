"""Interactive session over the same fixed-step loop as the SITL batch trial."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np

from edge.runtime import sha256 as digest
from simulation.sitl_flight import RADIUS, flight_steps, long_scene
from simulation.sitl_process import BIN
from .body import visual_contract
from .manual_input import ManualInput
from .recording import STAGES, plain
from .runner import DemoRunner
from .scenes import ROOM_SCENES
from .profiles import CANDIDATE_A_CONFIG

REFERENCE = 'long_arena'
DT = .00025
CONTROL_DT = .02
SOURCE = Path('output/sitl_bringup_v1/lockstep_v1')
CALIBRATION = SOURCE.parent/'twr_sensitivity_v1_run/calibration/report.json'
PROTOCOL = SOURCE.parent/'long_flight_v1/protocol.json'
PINNED_PROTOCOL = 'f818f68c4f49bf3cdc0d8b73da56740ffd106e6974600ce225f94ae68118ffaf'
REITERATION_SOURCE = CANDIDATE_A_CONFIG.parents[1]/'data/scenarios/reiteration_v1/sitl'
REITERATION_PROFILE = 'reiteration-sitl-control-v1'
REITERATION_MANIFEST_SHA = '63005e7dc1a702d0c794e2eec48bdbfc08613afdb885d3815062ce7cf31e4e78'


def pinned_profile():
    if not BIN.is_file():
        raise ValueError('Betaflight SITL доступний у Docker. Запустіть scripts/sitl/demo.ps1.')
    if digest(PROTOCOL) != PINNED_PROTOCOL:
        raise ValueError('SITL reference protocol checksum mismatch')
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    hashes = {'firmware': digest(BIN)}
    if hashes['firmware'] != protocol['firmware_sha256']:
        raise ValueError('SITL firmware checksum mismatch')
    for label, path, key in (
        ('config', Path('configs/demo_candidate_a.json'), 'configs/demo_candidate_a.json'),
        ('dynamics', Path('configs/ready_sim_reference_v1.json'), 'configs/ready_sim_reference_v1.json'),
        ('virtual_fc', Path('configs/sitl_virtual_v1.cli'), 'configs/sitl_virtual_v1.cli'),
        ('thrust_adapter', Path('simulation/sitl_twr.py'), 'simulation/sitl_twr.py'),
        ('eeprom', SOURCE/'template/eeprom.bin', '/results/lockstep_v1/template/eeprom.bin'),
        ('source_report', SOURCE/'report.json', '/results/lockstep_v1/report.json'),
        ('calibration', CALIBRATION, '/results/twr_sensitivity_v1_run/calibration/report.json'),
    ):
        hashes[label] = digest(path)
        if hashes[label] != protocol['input_sha256'][key]:
            raise ValueError(f'SITL reference checksum mismatch: {label}')
    return json.loads(CALIBRATION.read_text())['rows'], hashes


def reiteration_profile():
    manifest_path = REITERATION_SOURCE/'manifest.json'
    if digest(manifest_path) != REITERATION_MANIFEST_SHA:
        raise ValueError('Reiteration SITL manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    hashes = {'firmware': digest(BIN), 'manifest': REITERATION_MANIFEST_SHA}
    if hashes['firmware'] != manifest['firmware_sha256']:
        raise ValueError('Reiteration SITL firmware checksum mismatch')
    for base, members in ((REITERATION_SOURCE, manifest['files']),
                          (CANDIDATE_A_CONFIG.parents[1], manifest['repo_inputs'])):
        for name, expected in members.items():
            hashes[name] = digest(base/name)
            if hashes[name] != expected:
                raise ValueError('Reiteration SITL input checksum mismatch: '+name)
    rows = json.loads((REITERATION_SOURCE/'calibration.json').read_text(encoding='utf-8'))['rows']
    return rows, hashes


def scene_for_run(scene, seed, start, yaw, options):
    from .scenes import room_geometry
    if scene == REFERENCE:
        if options or list(start) != [0., 0., 1.] or yaw != 0.:
            raise ValueError('Еталонна кімната потребує старту (0, 0, 1) та напряму 0°.')
        return long_scene(), None
    if scene not in ROOM_SCENES:
        raise ValueError('Оберіть кімнату для Betaflight SITL.')
    room, boxes, cylinders, details = room_geometry(
        scene, seed=seed, start=start, options=options, collision_radius_m=RADIUS)
    spec = dict(room=asdict(room), boxes=[asdict(box) for box in boxes],
                cylinders=[asdict(cylinder) for cylinder in cylinders],
                start=list(start), yaw=yaw, radius_m=RADIUS)
    return spec, details


class SitlRunner(DemoRunner):
    """Reuse demo authority events and snapshots. Physics belongs to flight_steps."""
    step_dt = CONTROL_DT

    def __init__(self, *, scene='random_room', seed=20261021, start=(-.7, 0., 1.),
                 yaw=0., run_id='sitl', scene_options=None, max_duration_s=180.,
                 archive_path=None, reference_profile='historical'):
        if type(seed) is not int or seed < 0:
            raise ValueError('Invalid seed')
        if len(start) != 3 or not all(math.isfinite(v) for v in (*start, yaw)) or start[2] != 1.:
            raise ValueError('SITL start requires finite X/Y/yaw and height 1 m')
        self._validate_duration(max_duration_s)
        self.session_id = run_id
        self.scene, self.seed, self.scene_options = scene, seed, deepcopy(scene_options or {})
        self.initial_max_duration_s = self.max_duration_s = max_duration_s
        self.manual = ManualInput()
        self.paused, self.terminal, self.revision = True, False, 1
        self.pause_reason, self.request, self.stop_requested = 'ready', 'MANUAL', False
        self.previous_source = 'manual'
        self.events, self.input_schedule, self.rows = [], [], []
        self.trace = self.frame = None
        self.wall_started, self.wall_active = None, 0.
        spec, details = scene_for_run(scene, seed, start, yaw, self.scene_options)
        self.reference_profile = reference_profile
        if reference_profile == 'historical':
            calibration, hashes = pinned_profile()
            source, runtime = SOURCE, {}
        elif reference_profile == REITERATION_PROFILE:
            calibration, hashes = reiteration_profile()
            source, runtime = REITERATION_SOURCE, {'runtime_config': CANDIDATE_A_CONFIG}
        else:
            raise ValueError('Unknown SITL reference profile')
        self.scratch = TemporaryDirectory(prefix='chong-fly-sitl-') if archive_path is None else None
        out = Path(self.scratch.name)/'flight' if self.scratch else archive_path
        self.steps = flight_steps(out, source, scene, 'sitl', 'reference_units',
            physics_dt_s=DT, integration_profile='multi_rate', yaw_shaping=True, scene_override=spec,
            target_twr=5., calibration_rows=calibration, horizon_s=600., stop_at_s=None,
            session_id=self.session_id, archive=archive_path is not None, **runtime)
        try:
            self.view = next(self.steps)
            self.system = self.view.system
            self.receiver = self.view.integrated.receiver
            self._manifest = self._sitl_manifest(spec, details, hashes)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _validate_duration(seconds):
        DemoRunner._validate_duration(seconds)
        if abs(seconds/CONTROL_DT-round(seconds/CONTROL_DT)) > 1e-8:
            raise ValueError('SITL duration must align to 20 ms')

    @property
    def time_s(self):
        return self.view.time_s

    @property
    def step_index(self):
        return self.view.tick

    def pause(self, reason='user_pause'):
        if self.terminal:
            return
        if self.wall_started is not None:
            self.wall_active += monotonic()-self.wall_started
            self.wall_started = None
        if reason == 'user_pause':
            # A local freeze preserves virtual time and leases. Focus/owner
            # loss still revokes AUTO through the receiver, as in standalone.
            self.manual.clear()
            self.paused, self.pause_reason = True, reason
            self.revision += 1
            self._event('pause', reason=reason)
            self.input_schedule.append(dict(step=self.step_index, time_s=self.time_s,
                                            event=dict(type='pause', reason=reason)))
        else:
            super().pause(reason)

    def resume(self):
        super().resume()
        if self.wall_started is None:
            self.wall_started = monotonic()

    def event(self, event):
        kind = event.get('type')
        if kind == 'end':
            if not self.terminal:
                self.input_schedule.append(dict(step=self.step_index, time_s=self.time_s, event=dict(event)))
                self._finish('user_end')
        elif kind == 'resume' and 'mode' in event:
            mode = event['mode']
            if mode not in ('MANUAL', 'SHADOW', 'AUTO'):
                raise ValueError('Invalid mode')
            self.resume()
            super().event(dict(type='mode', mode=mode))
        else:
            super().event(event)

    def advance(self, steps=1):
        if type(steps) is not int or steps < 0:
            raise ValueError('Invalid step count')
        for _ in range(steps):
            if self.paused or self.terminal:
                break
            before = self.time_s
            try:
                self.steps.send(dict(request=self.request, stop_requested=self.stop_requested,
                                     manual_input=self.manual))
            except StopIteration:
                pass
            if self.view.observed is not None and self.view.observed['trace'].tick_time_s == before:
                self._record_control()
            if self.view.error:
                self._finish('simulation_fault: '+self.view.error)
            elif self.view.collision:
                self._finish('collision')
            elif self.time_s >= self.max_duration_s-1e-9:
                self._finish('duration_limit')

    def _record_control(self):
        view = self.view
        observed, selected = view.observed, view.observed['selected']
        result, trace = observed['result'], observed['trace'].to_dict()
        self.frame, self.trace = observed['frame'], trace
        time_s = trace['tick_time_s']
        source = ('auto' if selected['source']=='auto' else
                  'fallback' if self.request=='AUTO' and selected['tripped'] else 'manual')
        if source != self.previous_source:
            self.events.append(dict(time_s=time_s, kind='plant_authority', source=source))
        self.previous_source = source
        if trace['fault']:
            self.events.append(dict(time_s=time_s, kind='bridge_fault', reason=trace['fault']))
        sensors = {}
        for name in ('flow', 'tof', 'altitude', 'attitude'):
            sample = getattr(self.frame, name)
            channel = trace['sensor_channels'].get(name, {})
            sensors[name] = dict(asdict(sample), age_s=max(0., time_s-sample.timestamp_s),
                                 state=channel.get('state', 'received'))
        self.rows.append(plain(dict(time_s=time_s, **self._pose(view.control_state),
            clearance_m=view.control_gap, collision=None, sensors=sensors,
            min_tof_m=float(min(self.frame.tof.value)*3.5), flow_xy=self.frame.flow.value,
            request=self.request, authority=source.upper(), model_mode=trace['authority'],
            reason=result['reason'], manual_pwm=observed['manual_pwm'],
            plant_input_pwm=selected['applied_pwm'] if view.fc_pwm is not None else None, plant_source=source,
            fc_rc_pwm=view.fc_pwm, trace=trace, receiver=selected, decision=view.decision,
            stop_requested=self.stop_requested,
            events=[e for e in self.events if e['time_s']==time_s])))

    @staticmethod
    def _pose(state):
        return dict(position_m=state[:3], velocity_mps=state[3:6],
                    attitude_rad=state[6:9], angular_velocity_rad_s=state[9:12])

    def _endpoint(self):
        if not self.rows or self.time_s <= self.rows[-1]['time_s'] or not np.isfinite(self.view.state).all():
            return None
        row = deepcopy(self.rows[-1])
        for sensor in row['sensors'].values():
            sensor.update(age_s=self.time_s-sensor['timestamp_s'], state='held')
        return plain(dict(row, time_s=self.time_s, **self._pose(self.view.state),
            clearance_m=self.view.clearance, collision=self.view.collision, plant_input_pwm=None,
            manual_pwm=None, fc_rc_pwm=None, plant_source='terminated' if self.terminal else 'sample',
            authority='NONE', trace=dict.fromkeys(STAGES), receiver=None,
            events=[e for e in self.events if e['time_s']==self.time_s]))

    def _finish(self, reason):
        self._event('termination', reason=reason, collision=self.view.collision)
        self.pause(reason)
        self.terminal = True
        try:
            end = self._endpoint()
            if end is not None:
                self.rows.append(end)
        finally:
            self.close()

    def close(self):
        try:
            self.steps.close()
        finally:
            if self.scratch is not None:
                self.scratch.cleanup()

    def _sitl_manifest(self, spec, details, hashes):
        physics = self.view.manifest
        source = {}
        for path in ('demo/sitl.py', 'demo/runner.py', 'demo/manual_input.py', 'demo/scenes.py',
                     'demo/contacts.py', 'simulation/sensor_schedule.py', 'simulation/drone_env.py',
                     'simulation/drone_interface.py', 'simulation/pmw3901_emulator.py',
                     'simulation/ready_sim_contract.py', 'simulation/sitl_contract.py',
                     'simulation/sitl_lockstep.py', 'simulation/sitl_rc_units.py', 'simulation/sitl_flight.py',
                     'simulation/sitl_plant.py', 'simulation/sitl_process.py', 'simulation/sitl_control.py'):
            source[path] = digest(path)
        return plain(dict(kind='live_recording', backend='sitl', scene=self.scene, seed=self.seed,
            scene_options=self.scene_options, geometry_spec=details,
            geometry=dict(room=spec['room'], boxes=spec['boxes'], cylinders=spec['cylinders'],
                collision_radius_m=RADIUS, drone_visual=visual_contract(RADIUS)),
            start=dict(position_m=spec['start'], yaw_rad=spec['yaw'], velocity_mps=[0, 0, 0]),
            contact_model='sphere-sweep-v1', controller_mode='candidate_a_smooth_phased',
            reference_profile=self.reference_profile,
            candidate=self.system.runtime.metadata['source']['candidate'], system=self.system.manifest,
            physics_dt_s=DT, control_dt_s=CONTROL_DT, max_duration_s=self.initial_max_duration_s,
            dynamics=physics['dynamics'], thrust_adapter=physics['thrust_adapter'],
            actuator_calibration_rows=physics['actuator_calibration_rows'], input=self.manual.manifest(),
            target_height_m=self.system.runtime.altitude.config.target_height_m,
            sensor_profile='multi_rate', sensor_rates_hz=dict(tof=15, altitude=30, attitude=50, flow=50),
            pinned_sha256=hashes, source_sha256=source, transport=physics['authority'],
            units=dict(plant_input_pwm='reference PWM after receiver', fc_rc_pwm='integer SITL RC PWM'),
            limitations=['Ideal sensors and immediate simulated transport.',
                         'Swept-sphere contact ends flight. STOP damps velocity.']))

    def yaw_bursts(self):
        from simulation.metrics import measure_yaw_bursts
        rates = [r['angular_velocity_rad_s'][2] for r in self.rows if r['plant_input_pwm'] is not None]
        return plain(measure_yaw_bursts(rates, dt=CONTROL_DT))

    def recording(self):
        data = super().recording()
        end = self._endpoint()
        if end is not None:
            data['frames'].append(end)
        return data

    def snapshot(self):
        snapshot = super().snapshot()
        wall = self.wall_active
        if self.wall_started is not None:
            wall += monotonic()-self.wall_started
        return dict(snapshot, backend='sitl', real_time_factor=self.time_s/wall if wall else None,
                    pose=plain(self._pose(self.view.state)) if np.isfinite(self.view.state).all() else None)


def recompute(recording):
    """Reapply recorded input events on the same simulation tick grid."""
    manifest = recording['manifest']
    start, frames = manifest['start'], recording['frames']
    runner = SitlRunner(scene=manifest['scene'], seed=manifest['seed'], start=start['position_m'],
        yaw=start['yaw_rad'], scene_options=manifest['scene_options'],
        max_duration_s=manifest['max_duration_s'], run_id=frames[0]['trace']['session_id'],
        reference_profile=manifest.get('reference_profile', 'historical'))
    try:
        for name in ('geometry', 'geometry_spec', 'pinned_sha256', 'source_sha256'):
            if runner.manifest()[name] != manifest[name]:
                raise ValueError('SITL recording differs from current '+name)
        end = round(recording['end_time_s']/DT)
        if not 0 < end <= round(600/DT):
            raise ValueError('Invalid SITL end time')
        for item in recording['input_schedule']:
            tick = item['step']
            if type(tick) is not int or not runner.step_index <= tick <= end:
                raise ValueError('Invalid SITL input tick')
            while runner.step_index < tick and not (runner.paused or runner.terminal):
                runner.advance()
            if runner.step_index != tick:
                raise ValueError('Could not reach recorded SITL input tick')
            if not runner.terminal:
                runner.event(item['event'])
        while runner.step_index < end and not (runner.paused or runner.terminal):
            runner.advance()
        if runner.step_index != end:
            raise ValueError('Could not reach recorded SITL end time')
        return runner.recording()
    finally:
        runner.close()
