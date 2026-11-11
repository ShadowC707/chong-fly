"""Standalone benchmark: python benchmark.py --bundle . --out benchmark-host.json.

No motor outputs. Measurements describe this machine, not a Raspberry Pi estimate.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import tracemalloc

# Set before NumPy import. Do not assume an external scheduler is single-threaded.
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'

import numpy as np

if __package__:
    from .runtime import NumpyPolicy, FlightRuntime, AltitudeSample
else:
    from runtime import NumpyPolicy, FlightRuntime, AltitudeSample


def process_memory():
    if sys.platform=='win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
                (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
                'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        kernel=ctypes.WinDLL('kernel32'); psapi=ctypes.WinDLL('psapi')
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        counters=Counters(); counters.cb=ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            return dict(process_rss_bytes=counters.WorkingSetSize,process_peak_rss_bytes=counters.PeakWorkingSetSize)
    elif sys.platform.startswith('linux'):
        values={}
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:','VmHWM:')):
                k,v,*_=line.split(); values[k]=int(v)*1024
        return dict(process_rss_bytes=values.get('VmRSS:'),process_peak_rss_bytes=values.get('VmHWM:'))
    return dict(process_rss_bytes=None,process_peak_rss_bytes=None)


def measure(step,steps):
    for i in range(200):
        step(i)
    elapsed=np.empty(steps)
    for i in range(steps):
        start=time.perf_counter_ns(); step(i+200)
        elapsed[i]=(time.perf_counter_ns()-start)/1e6
    tracemalloc.start(); tracemalloc.reset_peak()
    for i in range(200):
        step(i+steps+200)
    _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    return dict(steps=steps,mean_ms=float(elapsed.mean()),median_ms=float(np.median(elapsed)),
        p95_ms=float(np.percentile(elapsed,95)),p99_ms=float(np.percentile(elapsed,99)),
        max_ms=float(elapsed.max()),over_20ms=int(np.sum(elapsed>20)),
        traced_peak_bytes_over_200_steps=peak)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--coordinator',choices=('baseline','anti_rollback','turn_latch','both'),default='both')
    parser.add_argument('--preset',choices=('original_v1','near_zero_v1'),default='original_v1')
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new benchmark report filename')
    if not 100<=args.steps<=100000:
        raise ValueError('steps must be 100..100000')
    rng=np.random.default_rng(31)
    flows=rng.uniform(-.2,.2,(128,2)).astype(np.float32)
    tofs=np.ones((128,64),np.float32); tofs[:64,:24]=.15
    policy=NumpyPolicy(args.bundle)
    policy_result=measure(lambda i:policy.step_np(flows[i%128],tofs[i%128]),args.steps)
    state_bytes=policy.h.nbytes+policy.fraction.nbytes
    parameter_bytes=policy.parameter_bytes
    del policy
    flight=FlightRuntime(args.bundle,coordinator_mode=args.coordinator,coordinator_preset=args.preset)
    def step(i):
        return flight.step(flows[i%128],tofs[i%128],
            AltitudeSample(.982,0.,0.,i*.02,True),now_s=i*.02)
    pipeline_result=measure(step,args.steps)
    report=dict(version='edge-benchmark-v1',host=dict(platform=platform.platform(),machine=platform.machine(),
        processor=platform.processor(),python=platform.python_version(),numpy=np.__version__),
        bundle_metadata_sha256=(args.bundle/'metadata.sha256').read_text().strip(),
        coordinator=args.coordinator,preset=args.preset,neural=policy_result,pipeline=pipeline_result,
        parameter_bytes=parameter_bytes,state_and_integrator_bytes=state_bytes,
        model_file_bytes=(args.bundle/'model.npz').stat().st_size,
        torch_loaded='torch' in sys.modules,scipy_loaded='scipy' in sys.modules,
        memory=process_memory(),thread_environment={k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        notes=['Measured on the named host; no Raspberry Pi performance claim.',
            'RSS includes Python, NumPy, loaded libraries and allocator caches, not only model arrays.',
            'tracemalloc is a separate allocation probe; it is not total RSS and does not affect the timed loop.',
            'No sensor I/O, serialization, communications or real-time scheduling included.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
