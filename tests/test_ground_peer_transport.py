"""Two ground peers over real host loopback; no manual wire/ESP Wi-Fi/FC claim."""
import time

from edge.hardware_udp import UdpEndpoint


def channels():
    pi,esp_pi,laptop,esp_manual=(UdpEndpoint(receive_budget=2) for _ in range(4))
    pi.set_peer(esp_pi.address)
    esp_pi.set_peer(pi.address)
    laptop.set_peer(esp_manual.address)
    esp_manual.set_peer(laptop.address)
    return pi,esp_pi,laptop,esp_manual


def receive(endpoint):
    until=time.monotonic()+2
    while time.monotonic()<until:
        data=endpoint.poll()
        if data: return data
        time.sleep(.001)
    raise AssertionError('Loopback packet not received')


def test_pi_and_laptop_keep_separate_peers_even_with_identical_payloads():
    endpoints=channels()
    pi,esp_pi,laptop,esp_manual=endpoints
    try:
        # Intentionally identical opaque bytes: source role must not depend on
        # payload appearance or become rebound to the most recent sender.
        payload=b'host-transport-probe-only'
        laptop.socket.sendto(payload,esp_pi.address)
        pi.socket.sendto(payload,esp_manual.address)
        assert pi.send(payload) and laptop.send(payload)
        assert receive(esp_pi)==[payload]
        assert receive(esp_manual)==[payload]
        assert esp_pi.counters['peer_rejected']==1
        assert esp_manual.counters['peer_rejected']==1
        assert esp_pi.peer==pi.address and esp_manual.peer==laptop.address
    finally:
        for endpoint in endpoints: endpoint.close()


def test_busy_manual_channel_does_not_consume_pi_receive_budget():
    endpoints=channels()
    pi,esp_pi,laptop,esp_manual=endpoints
    try:
        for _ in range(5): assert laptop.send(b'manual-probe')
        assert pi.send(b'pi-probe')
        assert esp_manual.poll()==[b'manual-probe']*2
        assert esp_manual.counters['scan_budget_exhausted']==1
        assert receive(esp_pi)==[b'pi-probe']
        assert esp_pi.counters['examined']==1
        assert esp_pi.counters['peer_rejected']==0
    finally:
        for endpoint in endpoints: endpoint.close()
