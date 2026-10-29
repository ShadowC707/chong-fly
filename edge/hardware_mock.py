"""Host-only bytes loop: fake acquisition -> pinned FlightSystem -> C++ core.

Integer event time is synthetic. Host wall time, pipes and compilation do not
represent Pi/ESP deadlines. No socket or FC I/O; no physical fallback.
"""
import argparse
from collections import Counter, deque
from dataclasses import asdict
import heapq
import json
from pathlib import Path
import subprocess

import numpy as np

from .hardware_wire import (ClockMapping, SensorMessage, WireMeasurement, ReceiptMessage,
                            encode_sensor, encode_receipt, decode_receipt)
from .pi_hardware_adapter import PiHardwareAdapter
from .system import RunConfig, observe_receiver


class BoundedInbox:
    """Reject newest on overflow; caller records loss. Never relabel old work."""
    def __init__(self,capacity):
        if type(capacity) is not int or capacity<1: raise ValueError('Positive capacity required')
        self.capacity=capacity
        self._queue=deque()
        self.overflows=0

    def __len__(self): return len(self._queue)

    def put(self,value):
        if len(self)==self.capacity:
            self.overflows+=1
            return False
        self._queue.append(value)
        return True

    def get(self): return self._queue.popleft()


class HostReceiver:
    """Test harness IPC only. Device data plane is the binary COMMAND frame."""
    def __init__(self,exe,*,args=('1','0')):
        self.process=subprocess.Popen([str(Path(exe).resolve()),*args],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            text=True,bufsize=1)

    def call(self,action):
        self.process.stdin.write(action+'\n')
        self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line: raise RuntimeError('C++ host receiver stopped: '+self.process.stderr.read())
        return json.loads(line)

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
            raise
        finally:
            self.process.stdout.close()
            self.process.stderr.close()
        if self.process.returncode: raise RuntimeError('C++ receiver harness failed')


def _distribution(values):
    return {name:float(np.percentile(values,p)) if values else None
            for name,p in (('p50',50),('p95',95),('p99',99),('max',100))}


def run_mock(receiver_exe,*,uplink_us=4000,service_us=2000,downlink_us=4000):
    for value in (uplink_us,service_us,downlink_us):
        if type(value) is not int or value<0 or value%1000:
            raise ValueError('Mock delays must be nonnegative whole milliseconds')
    session='01000000000000000100000000000000'
    config=RunConfig.from_json(Path(__file__).resolve().parents[1]/'configs/mvp_runtime_v3.json')
    adapter=PiHardwareAdapter(config,session_id=session,
        clock_mapping=ClockMapping(1_000_000,10.,0.,0.,20.))
    receiver=HostReceiver(receiver_exe)
    inbox=BoundedInbox(1)
    events=[]
    event_counter=0
    link_overflows=0
    active=None
    traces={}
    receipts=[]
    rtts=[]
    queue_wait=[]
    service=[]
    reasons=Counter()
    # Explicit fixture only: these values are not a hardware manual/failsafe.
    manual=(1213.,1500.,1500.,1500.)

    def schedule(due,kind,payload):
        nonlocal event_counter,link_overflows
        if len(events)>=16:
            link_overflows+=1
            return
        heapq.heappush(events,(due,event_counter,kind,payload))
        event_counter+=1

    try:
        receiver.call('reset 1000000')
        for t in range(0,1_661_000,1000):
            now=1_000_000+t
            mode='SHADOW' if t<600_000 else 'AUTO'
            state=receiver.call(f'mode {mode} {now}')
            # Local mode and watchdog precede sensor announce and every receive.
            if t<1_600_000 and t%20_000==0:
                seq=t//20_000
                receiver.call(f'sensor {seq} {now}')
                def sample(values): return WireMeasurement(seq,now,True,255,values)
                message=SensorMessage(session,seq,now,mode,True,state['tripped'],manual,
                    None if state['selected_pwm'] is None else tuple(state['selected_pwm']),
                    'none' if state['selected_pwm'] is None else 'auto',
                    sample((0.,0.)),sample((1.,)*64),sample((.982,0.,0.)),sample((0.,0.)))
                schedule(t+uplink_us,'sensor',encode_sensor(message))
            while events and events[0][0]<=t:
                _,_,kind,payload=heapq.heappop(events)
                if kind=='sensor':
                    inbox.put((payload,t))
                else:
                    seq,data=payload
                    row=receiver.call(f'receive {data.hex()} {now}')
                    receipt=ReceiptMessage(session,seq,now,row['reason'],row['tripped'],row['mode'],
                        None if row['accepted_pwm'] is None else tuple(row['accepted_pwm']),
                        None if row['selected_pwm'] is None else tuple(row['selected_pwm']),
                        'none' if row['selected_pwm'] is None else 'auto','not_attempted',
                        row['selected_session_id'],row['selected_sequence'])
                    receipt=decode_receipt(encode_receipt(receipt))
                    receipts.append(asdict(receipt))
                    reasons[receipt.reason]+=1
                    rtts.append(t-seq*20_000)
                    trace=traces[seq]
                    if trace.issued_pwm is not None and trace.receiver_status=='unobserved':
                        traces[seq]=observe_receiver(trace,session_id=session,sequence=seq,
                            accepted_pwm=receipt.accepted_pwm,receiver_time_s=now/1_000_000,
                            reason=receipt.reason)
            if active is None and len(inbox):
                payload,arrived=inbox.get()
                active=(payload,arrived,t,t+service_us)
            if active is not None and active[3]<=t:
                payload,arrived,started,_=active
                result=adapter.handle_sensor(payload,now_s=10.+t/1_000_000)
                if result is not None:
                    data,trace=result
                    traces[trace.sequence]=trace
                    queue_wait.append(started-arrived)
                    service.append(t-started)
                    schedule(t+downlink_us,'command',(trace.sequence,data))
                active=None
        final=receiver.call('poll 2661000')
    finally:
        receiver.close()
    return dict(version='esp32-host-mock-v1',hardware_tested=False,fc_attempts=0,
        timing_source='synthetic_integer_microseconds',
        model_sha256=adapter.system.runtime.metadata['files']['model.npz'],
        manifest_sha256=adapter.system.manifest_sha256,
        accepted_count=sum(r['accepted_pwm'] is not None for r in receipts),
        watchdog_tripped=final['tripped'],final_reason=final['reason'],reasons=dict(reasons),
        roundtrip_us=_distribution(rtts),pi_queue_wait_us=_distribution(queue_wait),
        pi_service_us=_distribution(service),
        roundtrip_deadline_misses=sum(r>=20_000 for r in rtts),
        inbox_overflows=inbox.overflows,link_overflows=link_overflows,
        traces=[t.to_dict() for t in traces.values()],receipts=receipts,
        limitation='Host C++ core and fake sensors; no ESP build, live clock sync, Pi board, Wi-Fi, FC or flight evidence.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receiver-exe',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--uplink-us',type=int,default=4000)
    parser.add_argument('--service-us',type=int,default=2000)
    parser.add_argument('--downlink-us',type=int,default=4000)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError('Choose a new report path')
    report=run_mock(args.receiver_exe,uplink_us=args.uplink_us,service_us=args.service_us,
                    downlink_us=args.downlink_us)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('traces','receipts')},indent=2))


if __name__=='__main__': main()
