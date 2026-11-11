"""Local-only Betaflight SITL bring-up. Run inside the dedicated Linux image.

First verifies the firmware's measured orientation/gyro and RC/motor signs
with the virtual craft fixed. Does not load a network or touch hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import socket
import struct
import subprocess
import threading
import time

import numpy as np

from simulation.ready_sim_contract import BETAFLIGHT_COMMIT, sitl_rc_packet
from simulation.sitl_contract import fdm_packet, motor_packet, msp_request, parse_msp

BIN = Path('/opt/betaflight/obj/main/betaflight_SITL.elf')
DT = .004


def quaternion(rpy):
    cr,cp,cy = np.cos(np.asarray(rpy)/2)
    sr,sp,sy = np.sin(np.asarray(rpy)/2)
    return np.array([cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy])


class Session:
    """Owns a virtual firmware process and localhost-only TCP/UDP clients."""
    def __init__(self, out, name):
        if platform.system() != 'Linux' or not BIN.is_file():
            raise RuntimeError('Run in the dedicated SITL container')
        out.mkdir(parents=True, exist_ok=True)
        self.log = (out / f'{name}.log').open('wb')
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(('127.0.0.1',9002))
        self.rx.settimeout(.03)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.lock = threading.Lock()
        self.q = np.array([1.,0.,0.,0.])
        self.omega = np.zeros(3)
        self.pos, self.vel, self.force = np.array([0.,0.,1.]), np.zeros(3), np.array([0.,0.,9.81])
        self.pwm = np.array([1000.,1500.,1500.,1500.])
        self.aux = np.full(12,1000.)
        self.motors = np.zeros(4)
        self.sim_time = 0.
        self.roundtrips = []
        self.timeouts = 0
        self.received = 0
        self.error = None
        self.callback = None
        self.done = threading.Event()
        self.ready = threading.Event()
        self.proc = subprocess.Popen(['stdbuf','-oL',str(BIN)], cwd=out,
                                     stdout=self.log, stderr=subprocess.STDOUT)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.tcp = None
        self.buffer = bytearray()

    def _loop(self):
        # UDP binds before virtualGyroDev/virtualAccDev exist. Sending FDM
        # during first-boot EEPROM initialization can crash upstream SITL.
        self.ready.wait()
        deadline = time.monotonic()
        tick = 0
        try:
            while not self.done.is_set() and self.proc.poll() is None:
                started = time.monotonic()
                with self.lock:
                    self.sim_time += DT
                    frame = fdm_packet(self.sim_time,self.pos,self.vel,self.q,self.omega,self.force)
                    rc = sitl_rc_packet(self.pwm,timestamp_s=self.sim_time,aux=self.aux)
                # Firmware sends at most one motor packet per accepted state.
                self.tx.sendto(frame,('127.0.0.1',9003))
                if tick % 5 == 0:
                    self.tx.sendto(rc,('127.0.0.1',9004))
                try:
                    data,_ = self.rx.recvfrom(1024)
                    motors = motor_packet(data)
                    with self.lock:
                        self.motors = motors
                    self.received += 1
                    self.roundtrips.append(time.monotonic()-started)
                    if self.callback is not None:
                        self.callback(motors)
                except socket.timeout:
                    self.timeouts += 1
                    if self.callback is not None:
                        raise RuntimeError('Motor timeout during closed-loop physics')
                tick += 1
                deadline += DT
                # Avoid fast catch-up bursts after CLI/startup pauses.
                if deadline < time.monotonic()-DT:
                    deadline = time.monotonic()
                self.done.wait(max(0.,deadline-time.monotonic()))
        except Exception as exc:
            self.error = repr(exc)
            print('SITL transport error:', self.error, flush=True)
            self.done.set()

    def connect(self):
        deadline = time.monotonic()+8
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError('Firmware exited; inspect log')
            try:
                self.tcp = socket.create_connection(('127.0.0.1',5761),timeout=.3)
                self.tcp.settimeout(3)
                # A listening socket is not proof that the dyad worker and
                # firmware scheduler are ready. Require a real MSP reply.
                self.msp(1)
                self.ready.set()
                return
            except OSError:
                if self.tcp:
                    self.tcp.close()
                    self.tcp = None
                    self.buffer.clear()
                time.sleep(.05)
        raise RuntimeError('No SITL UART1 TCP listener')

    def msp(self, command, payload=b''):
        self.tcp.sendall(msp_request(command,payload))
        while True:
            response = parse_msp(self.buffer)
            if response is not None:
                if response[0] == command:
                    return response[1]
                continue
            chunk = self.tcp.recv(4096)
            if not chunk:
                raise RuntimeError('MSP connection closed')
            self.buffer.extend(chunk)

    def cli(self, command):
        self.tcp.sendall(command.encode() + (b'' if command == '#' else b'\n'))
        result = bytearray()
        deadline = time.monotonic()+4
        prompted = False
        self.tcp.settimeout(.1)
        while time.monotonic()<deadline:
            try:
                chunk=self.tcp.recv(8192)
                if not chunk: break
                result.extend(chunk)
                if result.endswith(b'# '):
                    prompted = True
                    break
            except socket.timeout:
                continue
        self.tcp.settimeout(2)
        if not prompted and command != 'save':
            raise RuntimeError(f'No CLI prompt for {command!r}; reply={bytes(result)!r}, transport={self.error}, packets={self.received}')
        return result.decode(errors='replace')

    def status(self):
        data = self.msp(150)
        boxes = list(self.msp(119))
        flags = struct.unpack_from('<I',data,6)[0]
        extra = data[15]
        disable = struct.unpack_from('<I',data,17+extra)[0]
        return {'armed': bool(flags & (1 << boxes.index(0))) if 0 in boxes else False,
                'angle': bool(flags & (1 << boxes.index(1))) if 1 in boxes else False,
                'disable_flags': disable, 'mode_bits': flags, 'box_ids': boxes}

    def close(self):
        self.done.set()
        self.ready.set()
        self.thread.join(timeout=2)
        if self.tcp: self.tcp.close()
        self.tx.close(); self.rx.close()
        if self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill(); self.proc.wait(timeout=2)
        self.log.close()


def configure(out, config):
    if (out/'eeprom.bin').exists():
        raise ValueError('Use a new output directory; do not silently reuse EEPROM')
    session = Session(out,'configure')
    transcript = []
    try:
        session.connect()
        print('Initial MSP API:', list(session.msp(1)), flush=True)
        transcript.append(session.cli('#'))
        for command in config.read_text().splitlines():
            if not command.strip() or command.startswith('#'): continue
            response = session.cli(command)
            transcript.append(response)
            if 'ERROR' in response.upper() or 'INVALID' in response.upper():
                raise RuntimeError(f'CLI rejected {command}: {response}')
        transcript.append(session.cli('diff all'))
        transcript.append(session.cli('save'))
    finally:
        (out/'configuration.txt').write_text('\n'.join(transcript))
        session.close()


def probe(out):
    session = Session(out,'probe')
    report = {'betaflight_commit':BETAFLIGHT_COMMIT,'firmware_sha256':hashlib.sha256(BIN.read_bytes()).hexdigest(),
              'physics_enabled':False,'network_loaded':False,'attitude':[], 'gyro':[], 'rc_motors':[],
              'passed':False, 'experimental_not_admitted':True,
              'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        session.connect()
        time.sleep(2)
        report['api'] = list(session.msp(1))
        report['firmware_version'] = list(session.msp(3))
        report['initial_status'] = session.status()
        for degrees in ([0,0,0],[10,0,0],[0,20,0],[0,0,-30],[10,20,-30]):
            with session.lock:
                session.q = quaternion(np.radians(degrees))
            time.sleep(.25)
            raw = struct.unpack('<hhh',session.msp(108))
            actual = [raw[0]/10,raw[1]/10,raw[2]]
            expected = [degrees[0],degrees[1],(-degrees[2])%360]
            report['attitude'].append({'input_flu_deg':degrees,'msp_deg':actual,
                'expected_deg':expected,'passed':bool(np.allclose(actual,expected,atol=.2))})
        with session.lock: session.q = np.array([1.,0.,0.,0.])
        for degrees_s in ([0,0,0],[20,0,0],[0,20,0],[0,0,20]):
            with session.lock: session.omega = np.radians(degrees_s)
            time.sleep(.3)
            raw = struct.unpack('<9h',session.msp(102))
            expected = list(degrees_s)
            # MSP_RAW_IMU uses gyroRateDps(), which divides filtered dps by
            # sensor scale again: this payload is ADC-scaled, despite its name.
            measured = np.array(raw[3:6]) * (2000 / 32768)
            report['gyro'].append({'input_flu_deg_s':degrees_s,'msp_raw':list(raw[3:6]),
                'decoded_deg_s':measured.tolist(),
                'expected_deg_s':expected,'passed':bool(np.allclose(measured,expected,atol=.2))})
        with session.lock: session.omega = np.zeros(3)
        session.msp(205)  # Calibrate the virtual accelerometer at level.
        time.sleep(2)
        deadline = time.monotonic()+6
        while session.status()['disable_flags'] and time.monotonic()<deadline:
            time.sleep(.2)
        with session.lock: session.aux[1] = 1800
        time.sleep(.2)
        report['prearm_status'] = session.status()
        if report['prearm_status']['disable_flags']:
            raise RuntimeError(f'Pre-arm readiness not reached: {report["prearm_status"]}')
        with session.lock: session.aux[0] = 1800
        time.sleep(.5)
        report['armed_status'] = session.status()
        if not report['armed_status']['armed']:
            raise RuntimeError(f'Virtual arming denied: {report["armed_status"]}')
        for name,pwm in [('neutral',[1250,1500,1500,1500]),
                         ('roll',[1250,1550,1500,1500]),
                         ('pitch',[1250,1500,1550,1500]),
                         ('yaw',[1250,1500,1500,1550])]:
            # Reset accumulated I-term between clamped-axis probes by disarming.
            with session.lock:
                session.aux[0] = 1000
                session.pwm = np.array([1000.,1500.,1500.,1500.])
            time.sleep(.3)
            with session.lock: session.aux[0] = 1800
            time.sleep(.3)
            if not session.status()['armed']:
                raise RuntimeError('Re-arm failed between independent axis probes')
            with session.lock: session.pwm = np.array(pwm,dtype=float)
            time.sleep(.12)
            with session.lock: motors = session.motors.copy()
            moments = np.array([[-1,-1,1,1],[1,-1,1,-1],[-1,1,1,-1]]) @ (motors**2)
            report['rc_motors'].append({'case':name,'pwm':pwm,'native_outputs':motors.tolist(),
                'moment_sign_proxy':moments.tolist(),'raw_msp_motor':list(struct.unpack('<8H',session.msp(104)))})
            with session.lock: session.pwm = np.array([1250.,1500.,1500.,1500.])
            time.sleep(.3)
        with session.lock:
            session.aux[0] = 1000
            session.pwm[0] = 1000
        time.sleep(.2)
        report['disarmed_status'] = session.status()
        report['passed'] = (
            all(x['passed'] for x in report['attitude']+report['gyro'])
            and report['armed_status']['armed'] and report['armed_status']['angle']
            and not report['disarmed_status']['armed']
            and report['rc_motors'][1]['moment_sign_proxy'][0] > 0
            and report['rc_motors'][2]['moment_sign_proxy'][1] > 0
            and report['rc_motors'][3]['moment_sign_proxy'][2] < 0
            and session.error is None)
    finally:
        report['received_motor_packets'] = session.received
        report['startup_or_run_timeouts'] = session.timeouts
        report['transport_error'] = session.error
        report['roundtrip_ms'] = ({'median':float(np.median(session.roundtrips)*1000),
            'max':float(np.max(session.roundtrips)*1000)} if session.roundtrips else None)
        (out/'report.json').write_text(json.dumps(report,indent=2))
        session.close()
    return report


def closed_loop(out, prepared):
    """Bounded physics smoke using firmware motor commands; no project PID."""
    from evaluate.ready_sim_spike import BulletPlant
    from simulation.ready_sim_contract import make_urdf
    report = json.loads((prepared/'report.json').read_text())
    if not all(x['passed'] for x in report['attitude']+report['gyro']):
        raise ValueError('Firmware sensor contract probes must pass first')
    moments = {x['case']:x['moment_sign_proxy'] for x in report['rc_motors']}
    if not (moments['roll'][0]>0 and moments['pitch'][1]>0 and moments['yaw'][2]<0):
        raise ValueError('Firmware motor signs must pass first')
    p = json.loads(Path('/work/configs/ready_sim_reference_v1.json').read_text())['uncertainty_study']['nominal_dynamics']
    results = []
    for case in ('hover','roll','pitch','yaw'):
        target = out/case
        target.mkdir(parents=True,exist_ok=False)
        shutil.copy2(prepared/'eeprom.bin',target/'eeprom.bin')
        asset=target/'reference.urdf'
        make_urdf(asset,p)
        plant=BulletPlant(p,asset)
        session=Session(target,'firmware')
        states=[]; commands=[]; motors_log=[]
        outcome={'case':case,'duration_requested_s':4.,'controller':'actual Betaflight SITL',
                 'actuator_map':'native normalized motor output -> RPM linearly; uncalibrated',
                 'network_loaded':False,'altitude_hold':False}
        try:
            session.connect()
            time.sleep(2)
            deadline=time.monotonic()+6
            while session.status()['disable_flags'] and time.monotonic()<deadline:
                time.sleep(.2)
            outcome['prearm_status']=session.status()
            if outcome['prearm_status']['disable_flags']:
                raise RuntimeError('Pre-arm readiness not reached within bounded wait')
            with session.lock:
                session.aux[:2]=1800
            time.sleep(.4)
            outcome['armed_status']=session.status()
            if not outcome['armed_status']['armed'] or not outcome['armed_status']['angle']:
                raise RuntimeError('Physics test requires ARM and ANGLE')
            hover=1000+1000*np.sqrt(p['mass']*p['g']/(4*p['thrust_coeff']))/p['max_rpm']
            states.append(plant.state())
            finished=threading.Event()

            def advance(motors):
                tick=len(motors_log)
                old=states[-1]
                plant.step(np.asarray(motors,dtype=np.float32))
                state=plant.state()
                states.append(state)
                motors_log.append(motors.copy())
                with session.lock:
                    commands.append(session.pwm.copy())
                    pos,qxyzw=plant.pb.getBasePositionAndOrientation(plant.body,physicsClientId=plant.client)
                    rotation=np.array(plant.pb.getMatrixFromQuaternion(qxyzw)).reshape(3,3)
                    session.pos,session.vel=state[:3],state[3:6]
                    session.q=np.array(qxyzw)[[3,0,1,2]]
                    session.omega=state[9:12]
                    session.force=rotation.T @ ((state[3:6]-old[3:6])/DT + [0,0,p['g']])
                    session.pwm=np.array([hover,1500.,1500.,1500.])
                    t=(tick+1)*DT
                    if 1<=t<1.5 and case!='hover':
                        session.pwm[{'roll':1,'pitch':2,'yaw':3}[case]]+=50
                if not np.isfinite(state).all() or state[2]<.25 or state[2]>3 or np.max(np.abs(state[6:8]))>np.radians(45):
                    raise RuntimeError('Physical envelope exceeded; no parameter retuning')
                if len(motors_log)>=1000:
                    session.callback=None
                    finished.set()

            with session.lock: session.pwm=np.array([hover,1500.,1500.,1500.])
            # Let the receiver settle while the virtual craft is clamped.
            time.sleep(.2)
            session.callback=advance
            deadline=time.monotonic()+10
            while not finished.wait(.05) and not session.done.is_set() and time.monotonic()<deadline:
                pass
            session.callback=None
            if session.error: raise RuntimeError(session.error)
            if not finished.is_set(): raise RuntimeError('Physics loop exceeded wall-time budget')
            outcome['completed']=True
            outcome['final_status']=session.status()
        except Exception as exc:
            outcome.update(completed=False,error=repr(exc))
        finally:
            session.close()
            plant.close()
            state=np.array(states)
            np.savez_compressed(target/'trace.npz',time_s=np.arange(len(states))*DT,
                                state=state,pwm=commands,motors=motors_log)
            outcome['duration_actual_s']=len(motors_log)*DT
            if len(states):
                outcome['final_position_m']=state[-1,:3].tolist()
                outcome['final_rpy_deg']=np.degrees(state[-1,6:9]).tolist()
                outcome['max_abs_rpy_deg']=np.degrees(np.max(np.abs(state[:,6:9]),axis=0)).tolist()
                outcome['height_range_m']=[float(state[:,2].min()),float(state[:,2].max())]
            outcome['transport_error']=session.error
            outcome['received_motor_packets']=session.received
            outcome['startup_or_run_timeouts']=session.timeouts
            outcome['roundtrip_ms']=({'median':float(np.median(session.roundtrips)*1000),
                'max':float(np.max(session.roundtrips)*1000)} if session.roundtrips else None)
            (target/'report.json').write_text(json.dumps(outcome,indent=2))
            results.append(outcome)
    (out/'report.json').write_text(json.dumps(results,indent=2))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('/results/bringup_v1'))
    parser.add_argument('--config',type=Path,default=Path('/work/configs/sitl_virtual_v1.cli'))
    parser.add_argument('--physics-from',type=Path,help='Use an already validated virtual EEPROM/probe directory')
    args=parser.parse_args()
    if args.physics_from:
        results=closed_loop(args.out,args.physics_from)
        print(json.dumps(results,indent=2))
        if not all(x['completed'] for x in results):
            raise SystemExit(1)
    else:
        configure(args.out,args.config)
        result=probe(args.out)
        print(json.dumps(result,indent=2))
        if not result.get('passed',False):
            raise SystemExit(1)


if __name__=='__main__': main()
