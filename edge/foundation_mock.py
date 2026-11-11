"""ESP application foundation + two ground roles, entirely host/mock.

Real UDP loopback and pinned FlightSystem. Fake normalized sensors/manual RC,
exact synthetic clock mapping and synthetic 20-ms control clock. No FC I/O.
"""
import argparse
import json
from pathlib import Path
import time

from .hardware_mock import HostReceiver,_distribution
from .hardware_udp import UdpEndpoint,PiUdpPump
from .hardware_wire import ClockMapping,decode_receipt,decode_command
from .manual_wire import ManualInput,decode_challenge,encode_manual
from .pi_hardware_adapter import PiHardwareAdapter
from .system import RunConfig


def _wait(endpoint):
    end=time.monotonic()+2
    while time.monotonic()<end:
        packets=endpoint.poll()
        if packets: return packets
        time.sleep(.001)
    raise TimeoutError('Mock ground peer loopback timed out')


def run_foundation_mock(exe,*,takeover=False):
    session='01000000000000000100000000000000'
    config=RunConfig.from_json(Path(__file__).resolve().parents[1]/'configs/mvp_runtime_v3.json')
    adapter=PiHardwareAdapter(config,session_id=session,
        clock_mapping=ClockMapping(1_000_000,10.,0.,0.,20.))
    native=HostReceiver(exe,args=('1','1')) # explicit host-only AUTO authorization
    traces={}; receipts=[]; wall=[]; sizes={}
    # Simulation fixture, never a proposed GPW70 throttle or real fallback.
    manual=(1213.,1500.,1500.,1500.)
    try:
        with UdpEndpoint() as esp_pi,UdpEndpoint() as pi,UdpEndpoint() as esp_pc,UdpEndpoint() as pc:
            esp_pi.set_peer(pi.address); pi.set_peer(esp_pi.address)
            esp_pc.set_peer(pc.address); pc.set_peer(esp_pc.address)
            pump=PiUdpPump(adapter,pi)
            native.call('reset 1000000')
            boot=native.call('frame 0 1000000')
            challenge=decode_challenge(bytes.fromhex(boot['challenge_hex']))
            native.call('manual '+encode_manual(ManualInput(session,challenge.sequence,True,False,True,manual)).hex()+' 1001000')
            for seq in range(1,41):
                now=1_000_000+seq*20_000
                native.call(f'mode {"SHADOW" if seq<=30 else "AUTO"} {now}')
                for name,values in [('flow',(0.,0.)),('tof',(1.,)*64),('altitude',(.982,0.,0.)),('attitude',(0.,0.))]:
                    native.call(f'sample {name} {seq} {now} 1 255 '+ ' '.join(map(str,values))+f' {now}')
                frame=native.call(f'frame {seq} {now}')
                sensor=bytes.fromhex(frame['sensor_hex']); beacon=bytes.fromhex(frame['challenge_hex'])
                start=time.perf_counter_ns()
                if not esp_pc.send(beacon) or not esp_pi.send(sensor): raise RuntimeError('Mock send failed')
                challenge=decode_challenge(_wait(pc)[0])
                input_bytes=encode_manual(ManualInput(challenge.session_id,challenge.sequence,True,False,True,manual))
                if not pc.send(input_bytes): raise RuntimeError('Mock manual send failed')
                native.call(f'manual {_wait(esp_pc)[0].hex()} {now+1000}')
                deadline=time.monotonic()+2; ticks=[]
                while not ticks and time.monotonic()<deadline:
                    ticks=pump.step(now_s=10.+(seq*20_000+4000)/1_000_000)
                    if not ticks: time.sleep(.001)
                if len(ticks)!=1: raise RuntimeError('Expected one pinned model tick')
                traces[seq]=ticks[0]
                command=_wait(esp_pi)[0]
                decoded=decode_command(command)
                if decoded.sequence!=seq: raise RuntimeError('Wrong sequence')
                row=native.call(f'pi {command.hex()} {now+8000}')
                receipt_bytes=bytes.fromhex(row['receipt_hex'])
                receipt=decode_receipt(receipt_bytes); receipts.append(receipt)
                wall.append((time.perf_counter_ns()-start)/1000)
                if not esp_pi.send(receipt_bytes): raise RuntimeError('Mock receipt send failed')
                before=len(pump.receipts); deadline=time.monotonic()+2
                while len(pump.receipts)==before and time.monotonic()<deadline:
                    pump.step(now_s=10.+(seq*20_000+9000)/1_000_000)
                    if len(pump.receipts)==before: time.sleep(.001)
                if len(pump.receipts)!=before+1: raise RuntimeError('Receipt lost on host')
                sizes=dict(sensor=len(sensor),command=len(command),receipt=len(receipt_bytes),challenge=len(beacon),manual=len(input_bytes))
            extra={}
            if takeover:
                now=1_820_000
                frame=native.call(f'frame 41 {now}')
                challenge=decode_challenge(bytes.fromhex(frame['challenge_hex']))
                response=encode_manual(ManualInput(session,challenge.sequence,True,True,True,manual))
                row=native.call(f'manual {response.hex()} {now+1000}')
                extra=dict(takeover_source=row['source'],auto_revoked_by_takeover=row['auto_tripped'])
                final=native.call(f'disconnect {now+2000}')
                extra['manual_fault_after_disconnect']=final['manual_tripped']
            else:
                final=native.call('poll 1840000')
            return dict(version='esp32-foundation-mock-v1',hardware_tested=False,fc_attempts=0,
                timing_source='synthetic_20ms_control_clock_exact_clock_mapping',sensor_source='synthetic_normalized',
                transport='two_real_host_udp_pairs_plus_native_cpp_app_core',
                model_bundle=Path(config.bundle_dir).relative_to(Path(__file__).resolve().parents[1]).as_posix(),wire_sizes=sizes,
                accepted_count=sum(r.accepted_pwm is not None for r in receipts),
                accepted=[dict(sequence=r.sequence,accepted_pwm=list(r.accepted_pwm),
                    issued_pwm=list(traces[r.sequence].issued_pwm)) for r in receipts if r.accepted_pwm is not None],
                final_auto_expired=final['auto_tripped'],final_selected_source=final['source'],
                host_wall_roundtrip_us=_distribution(wall),**extra)
    finally:
        native.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foundation-exe',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--takeover',action='store_true')
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError('Choose a new report path')
    report=run_foundation_mock(args.foundation_exe,takeover=args.takeover)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='accepted'},indent=2))


if __name__=='__main__': main()
