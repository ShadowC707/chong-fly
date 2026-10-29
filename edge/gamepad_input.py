"""Read-only notebook input seam. No RC mapping, UDP, ARM, vibration or FC I/O.

XInput is the Windows API, not a promise about USB/RF transport or freshness.
Mock and real input expose identical raw samples; disconnect has state=None.
"""
import ctypes
from dataclasses import dataclass
import os
from statistics import median
import time
from typing import Protocol

AXES=('left_trigger','right_trigger','left_x','left_y','right_x','right_y')


def _integer(value,low,high):
    if type(value) is not int or not low<=value<=high: raise ValueError('Input integer outside bounds')


class XInputGamepad(ctypes.Structure):
    _fields_=[('buttons',ctypes.c_uint16),('left_trigger',ctypes.c_uint8),('right_trigger',ctypes.c_uint8),
        ('left_x',ctypes.c_int16),('left_y',ctypes.c_int16),('right_x',ctypes.c_int16),('right_y',ctypes.c_int16)]


class XInputState(ctypes.Structure):
    _fields_=[('packet_number',ctypes.c_uint32),('gamepad',XInputGamepad)]


@dataclass(frozen=True)
class GamepadState:
    packet_number: int
    buttons: int
    left_trigger: int
    right_trigger: int
    left_x: int
    left_y: int
    right_x: int
    right_y: int

    def __post_init__(self):
        _integer(self.packet_number,0,2**32-1)
        _integer(self.buttons,0,65535)
        for field in AXES: _integer(getattr(self,field),0 if 'trigger' in field else -32768,255 if 'trigger' in field else 32767)

    @classmethod
    def from_xinput(cls,state):
        return cls(int(state.packet_number),int(state.gamepad.buttons),
            *(int(getattr(state.gamepad,field)) for field in AXES))


@dataclass(frozen=True)
class GamepadSample:
    sample_id: int
    read_completed_us: int  # notebook monotonic time, not physical stick/RF capture
    source: str
    api_status: int
    state: GamepadState | None

    @property
    def connected(self): return self.api_status==0 and self.state is not None


class InputSource(Protocol):
    def poll(self) -> GamepadSample: ...


class _Samples:
    def __init__(self,source,clock_us):
        self.source=source
        self.clock_us=clock_us or (lambda:time.monotonic_ns()//1000)
        self._sample_id=0
        self._last_us=None

    def _sample(self,status,state):
        _integer(status,0,2**32-1)
        completed=self.clock_us()
        _integer(completed,0,2**64-1)
        if self._last_us is not None and completed<self._last_us: raise ValueError('Notebook clock regressed')
        if self._sample_id>=2**64: raise OverflowError('Input sample counter exhausted')
        if status==0 and not isinstance(state,GamepadState): raise ValueError('Successful input requires validated state')
        result=GamepadSample(self._sample_id,completed,self.source,status,state if status==0 else None)
        self._sample_id+=1
        self._last_us=completed
        return result


class WindowsXInputBackend:
    def __init__(self):
        if os.name!='nt': raise RuntimeError('Real XInput requires Windows; mock is explicit, never an automatic fallback')
        self.library=None
        for name in ('xinput1_4.dll','xinput9_1_0.dll'):
            try:
                library=ctypes.WinDLL(name)
                function=library.XInputGetState
            except (OSError,AttributeError): continue
            function.argtypes=[ctypes.c_uint32,ctypes.POINTER(XInputState)]
            function.restype=ctypes.c_uint32
            self.library=library
            self.get=function
            self.library_name=name
            break
        if self.library is None: raise RuntimeError('Windows XInput DLL not available')
        if ctypes.sizeof(XInputState)!=16: raise RuntimeError('Unexpected Windows XInput ABI')

    def get_state(self,slot):
        _integer(slot,0,3)
        raw=XInputState()
        status=int(self.get(slot,ctypes.byref(raw)))
        return status,GamepadState.from_xinput(raw) if status==0 else None


class XInputSource(_Samples):
    def __init__(self,slot,*,backend=None,clock_us=None):
        _integer(slot,0,3) # validate before loading a DLL
        super().__init__('xinput',clock_us)
        self.slot=slot
        self.backend=backend if backend is not None else WindowsXInputBackend()

    def poll(self):
        status,state=self.backend.get_state(self.slot)
        return self._sample(status,state)


class MockGamepadSource(_Samples):
    def __init__(self,states,*,clock_us=None):
        if not isinstance(states,(list,tuple)) or len(states)>12000: raise ValueError('Bounded scripted states required')
        if any(s is not None and not isinstance(s,GamepadState) for s in states): raise ValueError('Invalid mock state')
        super().__init__('mock',clock_us)
        self.states=iter(states)

    def poll(self):
        state=next(self.states,None)
        return self._sample(1167 if state is None else 0,state)


def summarize(samples):
    states=[s.state for s in samples if s.connected]
    return dict(samples=len(samples),connected_samples=len(states),disconnected_samples=len(samples)-len(states),
        packet_number_changes=sum(a.packet_number!=b.packet_number for a,b in zip(states,states[1:])),
        axes={field:dict(min=min(values),max=max(values),median=float(median(values)))
            for field in AXES for values in ([getattr(s,field) for s in states],) if values},
        button_masks_seen=sorted({s.buttons for s in states}),
        rf_freshness='unknown',physical_transport='unverified',device_identity='not established by XInput slot')
