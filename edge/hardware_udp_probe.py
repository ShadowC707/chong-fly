"""Real host UDP loopback + pinned FlightSystem + native receiver core.

Control/receiver timestamps are synthetic 20 ms ticks. Wall RTT belongs only
to this host, includes the C++ harness pipe and is not a Pi/Wi-Fi benchmark.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from .hardware_mock import HostReceiver, _distribution
from .hardware_udp import UdpEndpoint, PiUdpPump
from .hardware_wire import (ClockMapping, SensorMessage, WireMeasurement, ReceiptMessage,
                            encode_sensor, decode_command, encode_receipt)
from .pi_hardware_adapter import PiHardwareAdapter
from .system import RunConfig


def _wait(endpoint):
    deadline=time.monotonic()+2.
    while time.monotonic()<deadline:
        packets=endpoint.poll()
        if packets: return packets
        time.sleep(.001)
    raise TimeoutError('Host loopback receive timed out')


def run_probe(receiver_exe):
    session='01000000000000000100000000000000'
    cfg=RunConfig.from_json(Path(__file__).resolve().parents[1]/'configs/mvp_runtime_v3.json')
    adapter=PiHardwareAdapter(cfg,session_id=session,
        clock_mapping=ClockMapping(1_000_000,10.,0.,0.,20.))
    native=HostReceiver(receiver_exe)
    traces={}
    wall_rtts=[]
    try:
        with UdpEndpoint() as esp, UdpEndpoint() as pi:
            esp.set_peer(pi.address)
            pi.set_peer(esp.address)
            pump=PiUdpPump(adapter,pi)
            native.call('reset 1000000')
            for sequence in range(40):
                t=sequence*20_000
                now=1_000_000+t
                mode='SHADOW' if sequence<30 else 'AUTO'
                state=native.call(f'mode {mode} {now}')
                native.call(f'sensor {sequence} {now}')
                def measurement(values): return WireMeasurement(sequence,now,True,255,values)
                message=SensorMessage(session,sequence,now,mode,True,state['tripped'],
                    (1213.,1500.,1500.,1500.),None if state['selected_pwm'] is None else tuple(state['selected_pwm']),
                    'none' if state['selected_pwm'] is None else 'auto',
                    measurement((0.,0.)),measurement((1.,)*64),
                    measurement((.982,0.,0.)),measurement((0.,0.)))
                start=time.perf_counter_ns()
                if not esp.send(encode_sensor(message)): raise RuntimeError('Sensor send failed')
                # The test pump uses synthetic Pi time; waiting here only services
                # host socket delivery, it does not advance a simulated deadline.
                deadline=time.monotonic()+2.
                new=[]
                while not new and time.monotonic()<deadline:
                    new=pump.step(now_s=10.+(t+4000)/1_000_000)
                    if not new: time.sleep(.001)
                if len(new)!=1: raise RuntimeError('Expected one FlightSystem tick per sensor')
                traces[sequence]=new[0]
                commands=_wait(esp)
                if len(commands)!=1: raise RuntimeError('Expected one command datagram')
                command=decode_command(commands[0])
                if command.sequence!=sequence: raise RuntimeError('Command identity mismatch')
                row=native.call(f'receive {commands[0].hex()} {now+8000}')
                wall_rtts.append((time.perf_counter_ns()-start)/1000)
                receipt=ReceiptMessage(session,sequence,now+8000,row['reason'],row['tripped'],row['mode'],
                    None if row['accepted_pwm'] is None else tuple(row['accepted_pwm']),
                    None if row['selected_pwm'] is None else tuple(row['selected_pwm']),
                    'none' if row['selected_pwm'] is None else 'auto','not_attempted',
                    row['selected_session_id'],row['selected_sequence'])
                if not esp.send(encode_receipt(receipt)): raise RuntimeError('Receipt send failed')
                before=len(pump.receipts)
                deadline=time.monotonic()+2.
                while len(pump.receipts)==before and time.monotonic()<deadline:
                    pump.step(now_s=10.+(t+9000)/1_000_000)
                    if len(pump.receipts)==before: time.sleep(.001)
                if len(pump.receipts)!=before+1: raise RuntimeError('Receipt was not delivered')
            for observed in pump.observations: traces[observed.sequence]=observed
            final=native.call('poll 1820000')
            return dict(version='esp32-udp-host-v1',transport='host_udp_loopback',
                hardware_tested=False,fc_attempts=0,command_clock='synthetic_20ms_ticks',
                host_wall_roundtrip_us=_distribution(wall_rtts),
                accepted_count=sum(r.accepted_pwm is not None for r in pump.receipts),
                watchdog_tripped=final['tripped'],final_reason=final['reason'],
                counters=dict(pi=pump.counters,esp_udp=esp.counters,pi_udp=pi.counters),
                socket_receive_buffer_bytes=dict(esp=esp.receive_buffer_bytes,pi=pi.receive_buffer_bytes),
                traces=[trace.to_dict() for trace in traces.values()],
                receipts=[asdict(receipt) for receipt in pump.receipts],
                limitation='Real host UDP sockets; synthetic control time, fake sensors, native C++ core via pipes. No Wi-Fi/Pi/ESP target, FC output, pairing or deadline qualification.')
    finally:
        native.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receiver-exe',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError('Choose a new report path')
    report=run_probe(args.receiver_exe)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('traces','receipts')},indent=2))


if __name__=='__main__': main()
