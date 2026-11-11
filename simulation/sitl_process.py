"""Host process and lockstep transport for the patched Betaflight target."""
from pathlib import Path
import socket
import struct
import subprocess
import time

import numpy as np

from simulation.sitl_contract import fdm_packet, msp_request, parse_msp
from simulation.sitl_lockstep import decode_step, encode_step

BIN = Path('/opt/betaflight/obj/main/betaflight_SITL.elf')


class Session:
    def __init__(self,out,*,archive=True):
        out.mkdir(parents=True,exist_ok=True)
        self.path=out/'firmware.log'
        self.log=self.path.open('wb')
        self.proc=self.udp=None
        self.tcp=None; self.buffer=bytearray()
        self.archive=archive
        self.seq=0; self.clock=None; self.acks=[]; self.wall_ms=[]; self.elapsed_us=0
        self.pwm=np.array([1000.,1500.,1500.,1500.]); self.aux=np.full(12,1000.)
        self.pos=np.array([0.,0.,1.]); self.vel=np.zeros(3)
        self.q=np.array([1.,0.,0.,0.]); self.omega=np.zeros(3); self.force=np.array([0.,0.,9.81])
        try:
            self.proc=subprocess.Popen(['stdbuf','-oL',str(BIN)],cwd=out,stdout=self.log,stderr=subprocess.STDOUT)
            self.udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            self.udp.connect(('127.0.0.1',9005)); self.udp.settimeout(2.)
            deadline=time.monotonic()+8
            while 'CHONG_LOCKSTEP_READY' not in self.path.read_text(errors='replace'):
                if self.proc.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('Firmware did not reach explicit post-init readiness')
                time.sleep(.01)
            self.tcp=socket.create_connection(('127.0.0.1',5761),timeout=2.)
            self.tcp.setblocking(False)
        except BaseException:
            self.close()
            raise

    def step(self,rc_new=None,*,duration_us=4000):
        if duration_us not in (250,500,1000,2000,4000):
            raise ValueError('Supported coupled steps: 250, 500, 1000, 2000, 4000 us')
        if duration_us!=4000 and rc_new is None:
            raise ValueError('Nondefault steps require explicit RC freshness')
        self.seq+=1
        if rc_new is None: rc_new=(self.seq-1)%5==0
        self.elapsed_us+=duration_us
        fdm=fdm_packet(self.elapsed_us/1e6,self.pos,self.vel,self.q,self.omega,self.force)
        frame=encode_step(self.seq,duration_us,rc_new,fdm,self.pwm,self.aux)
        start=time.perf_counter()
        self.udp.send(frame)
        raw=self.udp.recv(1024)
        ack=decode_step(raw,self.seq,self.clock,duration_us)
        self.clock=ack['clock_us']; self.last_frame=frame; self.last_raw=raw
        self.last_ack=ack
        if self.archive:
            self.acks.append([ack['seq'],ack['clock_us'],ack['motor_updates']])
            self.wall_ms.append((time.perf_counter()-start)*1000)
        return np.array(ack['motors'],dtype=np.float32)

    def pump(self,steps):
        return [self.step() for _ in range(steps)]

    def msp(self,command,payload=b''):
        self.tcp.sendall(msp_request(command,payload))
        for _ in range(500):
            response=parse_msp(self.buffer)
            if response is not None:
                if response[0]!=command: raise ValueError('Unexpected MSP response')
                return response[1]
            self.step()
            try:
                chunk=self.tcp.recv(4096)
                if not chunk: raise RuntimeError('MSP closed')
                self.buffer.extend(chunk)
            except BlockingIOError: pass
        raise TimeoutError('MSP budget exceeded')

    def cli(self,command):
        self.tcp.sendall(command.encode()+(b'' if command=='#' else b'\n'))
        text=bytearray()
        for _ in range(500):
            if self.proc.poll() is not None:
                if command=='save': return text.decode(errors='replace')
                raise RuntimeError('Unexpected firmware exit')
            try: self.step()
            except (ConnectionRefusedError,socket.timeout):
                if command=='save' and self.proc.poll()==0: return text.decode(errors='replace')
                raise
            try:
                chunk=self.tcp.recv(8192)
                text.extend(chunk)
            except BlockingIOError: pass
            if text.endswith(b'# '): return text.decode(errors='replace')
        raise TimeoutError(f'CLI budget exceeded for {command}')

    def status(self):
        data=self.msp(150); boxes=list(self.msp(119))
        flags=struct.unpack_from('<I',data,6)[0]
        return {'armed':bool(flags & (1<<boxes.index(0))),
                'angle':bool(flags & (1<<boxes.index(1))),
                'disable_flags':struct.unpack_from('<I',data,17+data[15])[0]}

    def arm(self):
        self.aux[0]=1000; self.aux[1]=1800; self.pwm[:]=[1000,1500,1500,1500]
        self.pump(1500) # Six MODEL seconds, independent of laptop wall time.
        ready=self.status()
        if ready['disable_flags']: raise RuntimeError(f'Prearm blocked: {ready}')
        self.aux[0]=1800; self.pump(125)
        state=self.status()
        if not(state['armed'] and state['angle']): raise RuntimeError(f'ARM/ANGLE failed: {state}')
        return state

    def close(self):
        if self.tcp: self.tcp.close()
        if self.udp: self.udp.close()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=2)
        self.log.close()
