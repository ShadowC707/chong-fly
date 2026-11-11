"""Bounded raw-input capture for calibration planning; never sends drone commands."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from .gamepad_input import GamepadState,MockGamepadSource,XInputSource,WindowsXInputBackend,summarize


def capture(source,*,count,hz,sleep=time.sleep,clock=time.monotonic):
    if type(count) is not int or not 1<=count<=12000 or type(hz) is not int or not 1<=hz<=200:
        raise ValueError('Capture requires 1..12000 samples at 1..200 Hz')
    samples=[]
    due=clock()
    for _ in range(count):
        samples.append(source.poll())
        due+=1/hz
        delay=due-clock()
        if delay>0: sleep(delay)
    return dict(version='gamepad-raw-capture-v1',summary=summarize(samples),
        requested_hz=hz,samples=[dict(**asdict(s),connected=s.connected) for s in samples],
        calibration_profile_applied=False,udp_sends=0,fc_attempts=0,
        limitation='Read completion timestamps are not RF/stick capture timestamps. Raw ranges are not a validated RC profile.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=['mock','xinput','replay'],default='mock')
    parser.add_argument('--replay',type=Path,help='Offline raw capture v1; preserves recorded timestamps, never polls hardware')
    parser.add_argument('--slot',type=int,default=0)
    parser.add_argument('--seconds',type=float)
    parser.add_argument('--hz',type=int)
    parser.add_argument('--phase',choices=['observation','neutral','sweep'],default='observation')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--list',action='store_true',help='Read connection status for the four Windows XInput slots')
    args=parser.parse_args()
    if args.list:
        backend=WindowsXInputBackend()
        print(json.dumps(dict(api=backend.library_name,slots=[dict(slot=i,api_status=status,connected=status==0)
            for i in range(4) for status,_ in (backend.get_state(i),)]),indent=2))
        return
    if args.out is None: raise ValueError('Choose a new --out path for evidence')
    if args.out.exists(): raise FileExistsError('Capture evidence is never overwritten')
    if args.source=='replay':
        if args.replay is None or args.seconds is not None or args.hz is not None:
            raise ValueError('Replay requires --replay and uses all recorded samples, times and rate')
        from .gamepad_replay import load_capture
        source=load_capture(args.replay)
        report=capture(source,count=source.sample_count,hz=source.requested_hz,sleep=lambda _:None,clock=lambda:0.)
        report['replayed_input_sha256']=source.sha256
    else:
        if args.replay is not None: raise ValueError('--replay requires --source replay')
        seconds=args.seconds if args.seconds is not None else 5.
        hz=args.hz if args.hz is not None else 50
        if not math.isfinite(seconds) or not .1<=seconds<=60 or not 1<=hz<=200:
            raise ValueError('Use .1..60 seconds at 1..200 Hz')
        count=math.ceil(seconds*hz)
        if args.source=='xinput': source=XInputSource(args.slot)
        else:
            # Explicit stationary synthetic fixture; not a calibrated stick/throttle neutral.
            state=GamepadState(0,0,0,0,0,0,0,0)
            source=MockGamepadSource([state]*count)
        report=capture(source,count=count,hz=hz)
    report.update(source=args.source,slot=args.slot if args.source=='xinput' else None,
        operator_phase_label=args.phase,physical_connection_verified=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))


if __name__=='__main__': main()
