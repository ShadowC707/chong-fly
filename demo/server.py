"""Loopback-only UI server. A single worker owns all simulation state."""
import json
import mimetypes
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import uuid
import signal
from urllib.parse import urlsplit, parse_qs

from .runner import DemoRunner
from .recording import load_recording, plain
from .body import visual_contract

WEB = Path(__file__).parent/'web'
RECORDINGS = Path(__file__).parent/'recordings'
SCENE_OPTIONS = {'backend','scene','seed','start','yaw','scene_options','max_duration_s','controller_mode'}


class Pacer:
    """Wall time schedules bounded fixed steps; it never becomes physics dt."""
    def __init__(self, runner):
        self.runner, self.last, self.revision, self.credit = runner, None, None, 0.

    def poll(self, now):
        r = self.runner
        if r.paused or self.last is None or self.revision != r.revision:
            self.last, self.revision, self.credit = now, r.revision, 0.
            return
        elapsed, self.last = now-self.last, now
        if elapsed < 0:
            r.pause('invalid_wall_clock'); self.credit=0.; return
        if elapsed > .25:
            self.credit=0.; return
        self.credit += elapsed
        steps = min(round(.02/r.step_dt), int((self.credit+1e-9)/r.step_dt))
        self.credit -= steps*r.step_dt
        # Slow machines run slower; no long catch-up batches.
        self.credit = min(self.credit, .02)
        r.advance(steps)


class DemoService:
    def __init__(self, *, sitl_enabled=False):
        self.lock = threading.RLock()
        self.runner = None
        self.error = None
        self.client_id = None
        self.last_heartbeat = None
        self.pacer = None
        self.stopping = threading.Event()
        self.sitl_enabled = sitl_enabled

    def new_run(self, *, backend='standalone', **options):
        if backend == 'sitl':
            if not self.sitl_enabled:
                raise ValueError('Для Betaflight SITL запустіть scripts/sitl/demo.ps1 через Docker Desktop.')
            if 'controller_mode' in options:
                raise ValueError('Betaflight SITL має фіксований профіль Candidate A.')
            from .sitl import SitlRunner
            factory = SitlRunner
        elif backend == 'standalone':
            factory = DemoRunner
        else:
            raise ValueError('Unknown physics backend')
        if self.runner and not self.runner.terminal and (not self.runner.paused or self.runner.request!='MANUAL'):
            self.runner.pause('new_scene_requested')
        new = factory(run_id=uuid.uuid4().hex, **options)
        if self.runner: self.runner.close()
        self.runner, self.pacer, self.error = new, Pacer(new), None
        return new

    def start_fly(self, **options):
        """Fresh live policy run; AUTO still goes through the unchanged admission."""
        r=self.new_run(**dict(dict(scene='random_room',seed=20261021,start=(-.7,0.,1.),yaw=0.),**options))
        r.event(dict(type='resume'))
        r.event(dict(type='mode',mode='AUTO'))
        return r

    def heartbeat(self, client, now):
        if self.client_id not in (None, client):
            raise ValueError('Another browser owns the live controls; pause/close it and wait 2 s')
        self.client_id, self.last_heartbeat = client, now

    def poll(self, now):
        with self.lock:
            if self.last_heartbeat is not None and now-self.last_heartbeat > 1.5:
                if self.runner and not self.runner.terminal and (not self.runner.paused or self.runner.request!='MANUAL'):
                    self.runner.pause('client_disconnected')
                self.client_id = self.last_heartbeat = None
            if self.pacer: self.pacer.poll(now)

    def work(self):
        while not self.stopping.wait(.002):
            try: self.poll(time.monotonic())
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                    if self.runner:
                        self.runner.pause('server_error')
                        self.runner.terminal = True
                        self.runner.close()


def serve(port=8765, *, container_sitl=False):
    if container_sitl:
        from simulation.sitl_process import BIN
        if not BIN.is_file():
            raise SystemExit('Use scripts/sitl/demo.ps1 with the pinned Docker image.')
    service = DemoService(sitl_enabled=container_sitl)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass

        def send(self, status, body, mime='application/json; charset=utf-8'):
            if not isinstance(body, bytes): body=json.dumps(plain(body),allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            try:
                self.end_headers()
                self.wfile.write(body)
            except ConnectionError: pass

        def check_host(self):
            port = self.server.server_port
            if self.headers.get('Host') not in (f'127.0.0.1:{port}', f'localhost:{port}'):
                raise ValueError('Loopback Host required')

        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                self.check_host()
                with service.lock:
                    if path == '/api/catalog':
                        names = sorted(p.name for p in RECORDINGS.glob('*.json*'))
                        return self.send(200,dict(recordings=names,error=service.error,sitl_enabled=service.sitl_enabled,
                                                 drone_visual_template=visual_contract(.1)['unit_geometry']))
                    if path == '/api/replay':
                        name = parse_qs(urlsplit(self.path).query).get('name',[''])[0]
                        if name not in [p.name for p in RECORDINGS.glob('*.json*')]:
                            raise ValueError('Unknown recording')
                        return self.send(200,load_recording(RECORDINGS/name))
                    if path == '/api/snapshot':
                        query = parse_qs(urlsplit(self.path).query)
                        after = int(query.get('after',['0'])[0])
                        r = service.runner
                        if r and 'session_id' in query and query['session_id'][0] != r.session_id: after=0
                        return self.send(200,dict(error=service.error,state=None if r is None else r.snapshot(),
                            count=0 if r is None else len(r.rows),
                            frames=[] if r is None else r.rows[max(0,after):]))
                    if path == '/api/manifest':
                        return self.send(200,service.runner.manifest() if service.runner else {})
                    if path == '/api/export':
                        if not service.runner or not service.runner.rows: raise ValueError('No live recording yet')
                        return self.send(200,service.runner.recording())
                files={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
                if path not in files: return self.send(404,dict(error='Not found'))
                p=WEB/files[path]
                self.send(200,p.read_bytes(),(mimetypes.guess_type(p.name)[0] or 'text/plain')+'; charset=utf-8')
            except (ValueError, OSError) as exc: self.send(400,dict(error=str(exc)))

        def do_POST(self):
            try:
                host = f'127.0.0.1:{self.server.server_port}'
                self.check_host()
                origin=self.headers.get('Origin')
                if origin and origin not in ('http://'+host, f'http://localhost:{self.server.server_port}'):
                    raise ValueError('Same-origin UI required')
                length=int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 8192: raise ValueError('Invalid event size')
                data=json.loads(self.rfile.read(length))
                client=data.get('client_id')
                if not isinstance(client,str) or not 1 <= len(client) <= 100: raise ValueError('Client ID required')
                with service.lock:
                    service.heartbeat(client,time.monotonic())
                    if self.path == '/api/heartbeat': return self.send(200,dict(ok=True))
                    if self.path == '/api/start-fly':
                        opts=data.get('options',{})
                        if set(opts)-SCENE_OPTIONS: raise ValueError('Unknown scene option')
                        r=service.start_fly(**opts)
                        return self.send(200,dict(state=r.snapshot(),manifest=r.manifest()))
                    if self.path == '/api/new':
                        opts=data.get('options',{})
                        if set(opts)-SCENE_OPTIONS: raise ValueError('Unknown scene option')
                        r=service.new_run(**opts)
                        return self.send(200,dict(state=r.snapshot(),manifest=r.manifest()))
                    if self.path != '/api/event': raise ValueError('Unknown endpoint')
                    r=service.runner
                    if r is None: raise ValueError('Create a live scene first')
                    if data.get('session_id') != r.session_id: raise ValueError('Stale UI session')
                    # Reject delayed input after focus-loss/pause, even on resume.
                    if data.get('revision') != r.revision: raise ValueError('Stale UI input generation')
                    r.event(data['event'])
                    service.error = None
                    return self.send(200,dict(state=r.snapshot()))
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                self.send(400,dict(error=str(exc)))

    # Docker publishes only to the host loopback. The standalone server
    # continues to bind directly to loopback.
    httpd=ThreadingHTTPServer(('0.0.0.0' if container_sitl else '127.0.0.1',port),Handler)
    if container_sitl:
        def shutdown(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, shutdown)
    worker=threading.Thread(target=service.work,daemon=True); worker.start()
    print(f'Chong-Fly local demo: http://127.0.0.1:{httpd.server_port}',flush=True)
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        service.stopping.set(); worker.join(); httpd.server_close()
        with service.lock:
            if service.runner: service.runner.close()
