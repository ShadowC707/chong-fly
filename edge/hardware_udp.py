"""Bounded UDP I/O around CFHW v1. No implicit pairing, reset, retries or FC I/O.

Default bind is loopback. Exact IPv4 peer filtering is NOT authentication;
remote use needs a separately established trusted link and clock/session setup.
"""
from collections import Counter, OrderedDict, deque
import errno
import ipaddress
import socket

from .hardware_wire import MAX_FRAME, decode_receipt
from .pi_hardware_adapter import SensorDecodeError
from .system import observe_receiver


def _address(value,*,bind=False):
    try:
        host,port=value
        address=ipaddress.IPv4Address(host)
    except (ValueError,TypeError,ipaddress.AddressValueError) as exc:
        raise ValueError('Literal IPv4 address and port required') from exc
    if (type(port) is not int or not (0 if bind else 1)<=port<=65535
            or address.is_multicast or address.is_unspecified or int(address)==0xffffffff):
        raise ValueError('Unicast IPv4 address and valid port required')
    return str(address),port


class UdpEndpoint:
    """One fixed peer, bounded receive work and no application packet queue.

    Bind/peer are local configuration, never learned from received traffic.
    poll examines at most receive_budget datagrams, including rejected senders.
    The kernel receive buffer is bounded separately; OS drops are not all visible.
    """
    def __init__(self,bind=('127.0.0.1',0),*,receive_budget=8):
        if type(receive_budget) is not int or not 1<=receive_budget<=64:
            raise ValueError('Receive budget must be 1..64')
        bind=_address(bind,bind=True)
        self.receive_budget=receive_budget
        self.peer=None
        self.counters=Counter()
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,MAX_FRAME*receive_budget)
            self.socket.bind(bind)
            self.socket.setblocking(False)
            self.address=self.socket.getsockname()
            self.receive_buffer_bytes=self.socket.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF)
        except BaseException:
            self.socket.close()
            raise

    def set_peer(self,address):
        if self.peer is not None: raise ValueError('New peer requires a new endpoint/session setup')
        self.peer=_address(address)

    def send(self,data):
        if self.peer is None: raise ValueError('Set a trusted peer explicitly before I/O')
        if not isinstance(data,bytes) or not 1<=len(data)<=MAX_FRAME:
            raise ValueError('Expected 1..1024 frame bytes')
        try:
            sent=self.socket.sendto(data,self.peer)
        except BlockingIOError:
            self.counters['send_blocked']+=1
            return False
        except OSError:
            self.counters['send_error']+=1
            return False
        if sent!=len(data):
            self.counters['send_error']+=1
            return False
        self.counters['sent']+=1
        return True

    def poll(self):
        if self.peer is None: raise ValueError('Set a trusted peer explicitly before I/O')
        packets=[]
        for _ in range(self.receive_budget):
            try:
                # Extra byte detects an oversized datagram even on systems that
                # truncate recvfrom silently. Winsock instead reports WSAEMSGSIZE.
                data,sender=self.socket.recvfrom(MAX_FRAME+1)
            except BlockingIOError:
                break
            except OSError as exc:
                if exc.errno==errno.EMSGSIZE or getattr(exc,'winerror',None)==10040:
                    self.counters['oversize']+=1
                    continue
                self.counters['receive_error']+=1
                break
            self.counters['examined']+=1
            if sender!=self.peer:
                self.counters['peer_rejected']+=1
            elif len(data)>MAX_FRAME:
                self.counters['oversize']+=1
            elif not data:
                self.counters['empty']+=1
            else:
                self.counters['received']+=1
                packets.append(data)
        else:
            # Means the scan limit was used, not proof that more packets exist.
            self.counters['scan_budget_exhausted']+=1
        return packets

    def close(self): self.socket.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()


class PiUdpPump:
    """One nonblocking I/O step. The caller owns scheduling and timestamps.

    This does not provide real-time scheduling or hardware clock synchronization.
    Missing replies expire at the receiver; neither errors nor receipts retry RC.
    """
    def __init__(self,adapter,endpoint,*,receipt_capacity=64):
        if type(receipt_capacity) is not int or not 1<=receipt_capacity<=256:
            raise ValueError('Receipt capacity must be 1..256')
        self.adapter=adapter
        self.endpoint=endpoint
        self.counters=Counter()
        self.receipts=deque(maxlen=receipt_capacity)
        self.observations=deque(maxlen=receipt_capacity)
        self.pending=OrderedDict()
        self.capacity=receipt_capacity

    def _receipt(self,data):
        receipt=decode_receipt(data)
        if receipt.session_id!=self.adapter.session_id:
            self.counters['session_ignored']+=1
            return
        if len(self.receipts)==self.capacity: self.counters['receipt_evicted']+=1
        self.receipts.append(receipt)
        trace=self.pending.get(receipt.sequence)
        if trace is None:
            self.counters['receipt_without_issue']+=1
            return
        observed=observe_receiver(trace,session_id=receipt.session_id,sequence=receipt.sequence,
            accepted_pwm=receipt.accepted_pwm,receiver_time_s=receipt.receiver_us/1_000_000,
            reason=receipt.reason)
        del self.pending[receipt.sequence]
        self.observations.append(observed)

    def step(self,*,now_s):
        traces=[]
        for data in self.endpoint.poll():
            is_receipt=len(data)>5 and data[5]==3
            try:
                if is_receipt:
                    self._receipt(data)
                    continue
                result=self.adapter.handle_sensor(data,now_s=now_s)
            except (ValueError,TypeError,OverflowError) as exc:
                # Decoding/mapping failure must never produce a replacement RC.
                # Receiver watchdog remains authoritative if no valid reply comes.
                self.counters['malformed' if is_receipt or isinstance(exc,SensorDecodeError)
                              else 'adapter_error']+=1
                self.counters['errors']+=1
                self.last_error=str(exc)
                continue
            if result is None:
                self.counters['session_ignored']+=1
                continue
            response,trace=result
            traces.append(trace)
            if trace.issued_pwm is not None:
                if len(self.pending)==self.capacity:
                    self.pending.popitem(last=False)
                    self.counters['pending_evicted']+=1
                self.pending[trace.sequence]=trace
            self.endpoint.send(response)  # once; failure is counted, never retried
        return traces
