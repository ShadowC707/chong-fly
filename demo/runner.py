"""Small synchronous step adapter over the pinned system and actual receiver.

Zero simulated compute/link latency is explicit. Physics always advances 4 ms;
controller and sensor snapshot creation always occur on every fifth step.
No state or control law from the existing integration runner is copied here.
"""
from dataclasses import asdict, replace
from copy import deepcopy
import math
from pathlib import Path
import numpy as np

from edge.runtime import sha256
from edge.system import RunConfig, observe_receiver
from edge.pinned_system import build_pinned_system as build_system
from edge.session_receiver import SessionReceiver, CommandPacket
from edge.sensors import SensorTracker
from simulation.altitude_control import AltitudeHold, AltitudeSample, sample_from_observation
from simulation.collisions import contact_and_clearance as legacy_contact
from simulation.sensor_schedule import SampleSchedule
from .manual_input import ManualInput
from .recording import SCHEMA, plain
from .scenes import SCENES, make_environment
from .body import visual_contract
from .controllers import OVERLAY_PROFILES, install
from .contacts import contact_and_clearance,step_with_contact
from .profiles import PROFILE_ROLES, profile_config

ROOT = Path(__file__).resolve().parents[1]
DT = .004


class DemoRunner:
    step_dt = DT

    def __init__(self, *, config=None, scene='clear', seed=20261021,
                 start=(-.7,0.,1.), yaw=0., run_id='demo',scene_options=None,
                 collision_radius_m=.2,max_duration_s=180.,contact_model='sphere-sweep-v1',
                 controller_mode='original', recorded_config_sha256=None):
        if controller_mode not in ('original', 'candidate_a', *OVERLAY_PROFILES):
            raise ValueError('Unknown simulator controller')
        if controller_mode!='original' and config is not None:
            raise ValueError('Named simulator controller pins its own config')
        self.controller_mode=controller_mode
        self.config_path = Path(config or profile_config(controller_mode, recorded_config_sha256))
        self.base_config = RunConfig.from_json(self.config_path)
        if scene not in SCENES: raise ValueError('Unknown demo scene')
        if type(seed) is not int or seed < 0: raise ValueError('Invalid seed')
        if len(start) != 3 or not all(math.isfinite(x) for x in (*start,yaw)):
            raise ValueError('Invalid start pose')
        self.scene, self.seed, self.start, self.yaw = scene, seed, tuple(start), yaw
        self._validate_duration(max_duration_s)
        if type(collision_radius_m) not in (int,float) or not math.isfinite(collision_radius_m) or not 0<collision_radius_m<=.5:
            raise ValueError('Invalid demo collision radius')
        self.collision_radius_m=float(collision_radius_m)
        if contact_model not in ('sphere-sweep-v1','expanded-primitives-v1'):
            raise ValueError('Invalid demo contact model')
        self.contact_model=contact_model
        self.contact=contact_and_clearance if contact_model=='sphere-sweep-v1' else legacy_contact
        self.initial_max_duration_s=float(max_duration_s)
        self.scene_options=deepcopy(scene_options or {})
        self.run_id, self.generation = run_id, -1
        self.reset()

    @property
    def time_s(self): return self.step_index*DT

    def close(self):
        self.env.close()

    @staticmethod
    def _validate_duration(seconds):
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 1<=seconds<=600:
            raise ValueError('Live duration must be from 1 to 600 seconds')

    def reset(self):
        if hasattr(self, 'env'): self.env.close()
        self.generation += 1
        # Deterministic headless runs; the server supplies a random run_id.
        self.session_id = f'{self.run_id}-{self.generation}'
        self.system = build_system(replace(self.base_config, session_id=self.session_id, seed=self.seed))
        if self.controller_mode in OVERLAY_PROFILES:
            self.system=install(self.system, profile=self.controller_mode)
        self.receiver = SessionReceiver()
        self.receiver.reset(session_id=self.session_id, now_s=0.)
        self.env = make_environment(self.scene,seed=self.seed,start=self.start,options=self.scene_options,
                                    collision_radius_m=self.collision_radius_m)
        self.env.reset(initial_pos=self.start, seed=self.seed)
        self.env.physics.vel[:] = 0
        self.env.physics.quat[:] = self.env.physics.euler_to_quaternion(0,0,self.yaw)
        self.obs = self.env._sample_observation(0.)
        hit, _ = self.contact(self.start,self.start,self.env.room,
            self.env.boxes,self.env.cylinders,self.env.collision_radius_m)
        if hit: raise ValueError('Start intersects scene geometry')
        self.manual, self.manual_altitude = ManualInput(), AltitudeHold(self.system.runtime.altitude.config)
        self.schedule, self.sensor_tracker = SampleSchedule('multi'), SensorTracker(self.base_config.sensor_budgets)
        self.step_index = 0
        self.max_duration_s=self.initial_max_duration_s
        self.paused, self.terminal = True, False
        self.pause_reason, self.request, self.stop_requested = 'ready', 'MANUAL', False
        self.previous_pwm = (1213.,1500.,1500.,1500.)
        self.previous_source = 'manual'
        self.rows, self.events, self.input_schedule = [], [], []
        self.frame = self.trace = None
        self.last_reason, self.admission = 'ready', []
        self.revision = getattr(self, 'revision', 0)+1
        self._manifest = plain(self._build_manifest())

    def _event(self, kind, **fields):
        self.events.append(dict(time_s=self.time_s, kind=kind, **fields))

    def _request(self, mode):
        if mode not in ('MANUAL','SHADOW','AUTO'): raise ValueError('Invalid mode')
        if mode != self.request:
            self._event('mode_request', mode=mode)
        self.request = mode
        if mode != 'AUTO':
            self.receiver.set_mode(mode, now_s=self.time_s)
            self.stop_requested = False

    def pause(self, reason='user_pause'):
        self.manual.clear()
        self._request('MANUAL')
        self.paused, self.pause_reason = True, reason
        self.revision += 1
        self._event('pause', reason=reason)
        self.input_schedule.append(dict(step=self.step_index,time_s=self.time_s,
                                        event=dict(type='pause',reason=reason)))

    def resume(self):
        if self.terminal: raise ValueError('Run ended; reset required')
        self.paused, self.pause_reason = False, None
        self.revision += 1
        self._event('resume')
        self.input_schedule.append(dict(step=self.step_index,time_s=self.time_s,event=dict(type='resume')))

    def event(self, event):
        kind = event.get('type')
        if kind == 'key':
            if self.paused: return
            self.manual.key(event.get('code'), event.get('down'))
            if event['down'] and self.request == 'AUTO': self._request('MANUAL')
        elif kind == 'mode':
            if self.paused and event.get('mode') != 'MANUAL':
                raise ValueError('Resume explicitly before requesting SHADOW/AUTO')
            self._request(event.get('mode'))
        elif kind == 'stop':
            if self.previous_source != 'auto' or self.request != 'AUTO':
                self._event('stop_unavailable'); return
            self.stop_requested = not self.stop_requested
            self._event('stop', active=self.stop_requested)
        elif kind == 'pause': self.pause(event.get('reason','user_pause'))
        elif kind == 'resume': self.resume()
        elif kind == 'focus_lost': self.pause('focus_lost')
        elif kind == 'duration':
            seconds=event.get('seconds')
            self._validate_duration(seconds)
            if self.terminal or seconds<=self.time_s:
                raise ValueError('Live duration must exceed elapsed time; create a new run if ended')
            self.max_duration_s=float(seconds)
            self.revision+=1  # a pending snapshot must not restore the previous limit
            self._event('duration',seconds=self.max_duration_s)
        else: raise ValueError('Unknown demo event')
        if kind not in ('pause','resume','focus_lost'):
            self.input_schedule.append(dict(step=self.step_index, time_s=self.time_s, event=dict(event)))

    def _admission_details(self, result, manual):
        # Explain the existing predicate, never change its decision.
        if result['reason'] != 'engage_denied': return []
        b = self.system.bridge; c = b.config; a = b.runtime.altitude.last_command
        target = result['preview_pwm']; f = self.frame
        tests = [('flow', float(np.linalg.norm(b.filtered_flow)), c.max_flow),
            ('height_error_m', abs(a.height_m-b.target_height_m), c.height_tolerance_m),
            ('vertical_speed_mps', abs(a.vertical_speed_mps), c.max_vertical_speed_mps),
            ('tilt_rad', max(abs(x) for x in f.attitude.value), c.max_tilt_rad),
            ('throttle_difference_pwm',abs(target[0]-manual[0]),c.max_throttle_difference_pwm),
            ('attitude_difference_pwm',max(abs(target[i]-manual[i]) for i in (1,2)),c.max_attitude_difference_pwm),
            ('yaw_difference_pwm',abs(target[3]-manual[3]),c.max_yaw_difference_pwm)]
        failures = [dict(check=k,value=float(v),limit=lim) for k,v,lim in tests if v > lim]
        distance = min(f.tof.value)*b.tof_range_m
        if distance < c.clear_m: failures.append(dict(check='min_tof_m',value=float(distance),minimum=c.clear_m))
        if not failures: failures.append(dict(check='ready_dwell_s',required=c.ready_dwell_s,
            elapsed=0. if b.ready_since is None else self.time_s-b.ready_since))
        return failures

    def advance(self, steps=1):
        if type(steps) is not int or steps < 0: raise ValueError('Invalid step count')
        for _ in range(steps):
            if self.paused or self.terminal: break
            try: self._step()
            except (ValueError, FloatingPointError) as exc:
                self._event('simulation_fault', reason=str(exc))
                self.pause('simulation_fault: '+str(exc)); self.terminal = True
                break

    def _step(self):
        t = self.time_s
        before = self.receiver.output(now_s=t, fallback_pwm=self.previous_pwm)
        active_auto = before['source'] == 'auto' and self.request == 'AUTO'
        if self.step_index % 5 == 0:
            sample = sample_from_observation(self.env.get_isaac_obs())
            self.frame = self.schedule.frame(self.step_index//5, self.obs[:2], self.obs[2:], sample)
            self.sensor_tracker.resolve(self.frame, now_s=t)
        m = self.frame.altitude
        altitude = AltitudeSample(*m.value, m.timestamp_s, m.valid)
        throttle = self.manual_altitude.update(altitude, now_s=t, allow_hold=True,
                                              integrate=not active_auto).throttle_pwm
        if active_auto:
            self.manual_altitude.synchronize_output(self.previous_pwm[0], now_s=t)
            throttle = self.previous_pwm[0]
            self.manual.synchronize(self.previous_pwm)
            axes = np.asarray(self.previous_pwm[1:])
        else: axes = self.manual.advance(DT)
        manual = (float(throttle), *(float(x) for x in axes))
        if self.step_index % 5 == 0:
            self.receiver.announce_sensor(self.frame.sequence, now_s=t)
            requested = 'SHADOW' if self.request == 'AUTO' and self.receiver.gate.tripped else self.request
            self.prepare_controller()
            extra = ({'heading_sample': self.swath_adapter.capture(self, t)}
                     if hasattr(self.system, 'navigation') else {})
            result = self.system.tick(self.frame, now_s=t, request=requested,
                                      manual_pwm=manual, stop_requested=self.stop_requested, **extra)
            self.admission = self._admission_details(result, manual)
            self.last_reason = result['reason']
            trace = result['trace']
            self.receiver.set_mode('AUTO' if result['lease'] is not None else
                                   ('MANUAL' if requested == 'MANUAL' else 'SHADOW'), now_s=t)
            event = self.receiver.receive(CommandPacket(self.session_id, self.frame.sequence,
                                                        result['lease']), now_s=t)
            if trace.issued_pwm is not None:
                trace = observe_receiver(trace, session_id=self.session_id, sequence=self.frame.sequence,
                    accepted_pwm=event.accepted_pwm, receiver_time_s=t, reason=event.reason)
            self.trace = trace.to_dict()
            if result['fault']: self._event('bridge_fault', reason=result['fault'])
        selected = self.receiver.output(now_s=t, fallback_pwm=manual)
        source = ('auto' if selected['source']=='auto' else
                  'fallback' if self.request=='AUTO' and selected['tripped'] else 'manual')
        applied = selected['applied_pwm']
        # Track a newly accepted throttle at this same tick, without updating
        # either estimator twice. Next takeover starts at the actual plant RC.
        if source == 'auto':
            self.manual_altitude.synchronize_output(float(applied[0]), now_s=t)
            self.manual.synchronize(applied)
        if source != self.previous_source: self._event('plant_authority', source=source)
        kind, clearance = self.contact(self.env.physics.pos,self.env.physics.pos,
            self.env.room,self.env.boxes,self.env.cylinders,self.env.collision_radius_m)
        sensors = {}
        for name in ('flow','tof','altitude','attitude'):
            measurement = getattr(self.frame,name)
            state = self.sensor_tracker.last_report[name]['state']
            sensors[name] = dict(asdict(measurement), age_s=max(0.,t-measurement.timestamp_s),
                                 state=state if self.step_index % 5 == 0 else 'held')
        row = dict(time_s=t, position_m=self.env.physics.pos.copy(),
            attitude_rad=self.env.physics.quaternion_to_euler(self.env.physics.quat),
            velocity_mps=self.env.physics.vel.copy(), angular_velocity_rad_s=self.env.physics.omega.copy(),
            clearance_m=clearance, collision=kind, sensors=sensors,
            min_tof_m=float(min(self.frame.tof.value)*self.env.tof.max_range),
            flow_xy=self.frame.flow.value, request=self.request,
            authority='AUTO' if source=='auto' else ('FALLBACK' if source=='fallback' else 'MANUAL'),
            model_mode=self.system.bridge.state, reason=self.last_reason, admission=self.admission,
            manual_pwm=manual, plant_input_pwm=applied, plant_source=source,
            trace=self.trace, receiver=dict(selected, deadline_s=self.receiver.gate.deadline),
            stop_requested=self.stop_requested, manual_integral=self.manual_altitude._integral,
            events=[e for e in self.events if e['time_s']==t])
        self.rows.append(plain(row))
        self.previous_pwm, self.previous_source = applied, source
        self.obs, _, done, info = (step_with_contact(self.env,applied) if self.contact_model=='sphere-sweep-v1'
                                  else self.env.step(applied))
        self.step_index += 1
        if done:
            self._event('termination', collision=info['collision_kind'], invalid=info['invalid_state'])
            # Store the terminal physical sample, no invented next command.
            terminal = dict(self.rows[-1], time_s=self.time_s,
                position_m=plain(self.env.physics.pos), attitude_rad=plain(info['euler_rad']),
                velocity_mps=plain(self.env.physics.vel), collision=info['collision_kind'],
                clearance_m=info['clearance_m'], plant_input_pwm=None,
                plant_source='terminated',authority='NONE',
                events=[self.events[-1]])
            terminal['sensors']=deepcopy(terminal['sensors'])
            for measurement in terminal['sensors'].values():
                measurement['age_s']=self.time_s-measurement['timestamp_s']
                measurement['state']='held'
            if (not info['invalid_state'] and all(np.isfinite(terminal[k]).all()
                    for k in ('position_m','attitude_rad','velocity_mps'))):
                self.rows.append(terminal)
            self.pause('collision' if info['crashed'] else 'invalid_physics'); self.terminal=True
        elif self.step_index >= round(self.max_duration_s/DT):
            self.pause('duration_limit'); self.terminal=True

    def manifest(self):
        return deepcopy(self._manifest)

    def prepare_controller(self):
        if (hasattr(self.system, 'navigation')
                and getattr(getattr(self, 'swath_adapter', None), 'system', None) is not self.system):
            from .swath_adapter import SwathSimulationAdapter
            self.swath_adapter = SwathSimulationAdapter(self)

    def _build_manifest(self):
        self.prepare_controller()
        paths = ['demo/runner.py','demo/manual_input.py','demo/scenes.py','demo/body.py','demo/body.json','demo/contacts.py','simulation/collisions.py','edge/session_receiver.py',
                 'simulation/drone_env.py','simulation/sensor_schedule.py']
        profile={} if self.controller_mode=='original' else dict(controller_mode=self.controller_mode)
        if profile:
            profile['controller_role'] = PROFILE_ROLES.get(self.controller_mode, 'historical-experiment')
            paths+=['demo/controllers.py','demo/profiles.py','simulation/phased_coordinator.py']
        if hasattr(self.system, 'navigation'):
            profile['swath_inputs'] = self.swath_adapter.contract()
        return dict(**profile,kind='live_recording', scene=self.scene, seed=self.seed,
            contact_model=self.contact_model,
            scene_options=self.scene_options,geometry_spec=getattr(self.env,'demo_geometry_spec',None),
            start=dict(position_m=self.start,yaw_rad=self.yaw,velocity_mps=[0,0,0]),
            geometry=dict(room=asdict(self.env.room), boxes=[asdict(b) for b in self.env.boxes],
                cylinders=[asdict(c) for c in self.env.cylinders],collision_radius_m=self.env.collision_radius_m,
                drone_visual=visual_contract(self.env.collision_radius_m)),
            physics_dt_s=DT, control_dt_s=.02, sensor_profile='multi',
            max_duration_s=self.initial_max_duration_s,
            input=self.manual.manifest(), target_height_m=self.manual_altitude.config.target_height_m,
            config_sha256=sha256(self.config_path), system=self.system.manifest,
            candidate=self.system.runtime.metadata['source']['candidate'],
            source_sha256={p:sha256(ROOT/p) for p in paths},
            units=dict(position='m',attitude='rad',velocity='m/s',pwm='us',tof='normalized distance / 3.5 m',flow='normalized'),
            tof_orientation='rows top to bottom; columns left (+body Y) to right (-body Y)',
            transport='synchronous in-process delivery through SessionReceiver; zero compute/link simulation time',
            limitations=['Numerical simulation, not flight qualification.',
                ('mvp_admission.passed=false: broad_joint 3/4.' if self.controller_mode=='original' else
                 'Candidate A research profile; geometric simulator validation, not hardware admission.'),
                'Pi budget not demonstrated: audit 14/14 at zero delay, 11/14 at 2 ms.',
                'MANUAL altitude assist; release does not brake XY.',
                'Receiver fallback uses manual channel; no transport-fault scenarios in this demo.',
                'STOP damps velocity, does not hold absolute position.'])

    def recording(self):
        return plain(dict(schema=SCHEMA, manifest=self.manifest(),frames=self.rows,
                          events=self.events,input_schedule=self.input_schedule,
                          end_time_s=self.time_s,yaw_bursts=self.yaw_bursts()))

    def yaw_bursts(self):
        from simulation.metrics import measure_yaw_bursts
        rates=[row['angular_velocity_rad_s'][2] for row in self.rows[::5]]
        return plain(measure_yaw_bursts(rates,dt=.02))

    def snapshot(self):
        has_state=self.trace is not None and self.trace['network_pwm'] is not None
        status=('paused' if self.paused else 'running' if has_state and
                self.request!='MANUAL' else 'not_running')
        neural=dict(status=status,count=self.system.runtime.policy.metadata['hidden_size'],
            control_time_s=None if not has_state else self.trace['tick_time_s'],
            values=plain(self.system.runtime.policy.h.copy()) if has_state and
                (self.paused or self.request!='MANUAL') else None,
            color_scale_abs=float(np.max(np.abs(self.system.runtime.policy.arrays['A']))),
            meaning='Signed rate-model state h in model index order; not spikes or anatomical layout')
        return plain(dict(session_id=self.session_id,revision=self.revision,time_s=self.time_s,paused=self.paused,
            max_duration_s=self.max_duration_s,
            terminal=self.terminal,pause_reason=self.pause_reason,request=self.request,
            reset_required=self.receiver.gate.tripped,frame=self.rows[-1] if self.rows else None,
            receiver_block_reason=self.receiver.reason if self.receiver.gate.tripped else None,
            events=self.events[-30:],yaw_bursts=self.yaw_bursts(),neural=neural))
