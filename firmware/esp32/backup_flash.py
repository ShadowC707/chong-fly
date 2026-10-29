"""Read-only full Flash backup, verified per chunk by esptool's stub MD5.

Output becomes .bin only after every chunk succeeds. No write/erase/eFuse API.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def copy_flash(read,reconnect,output,*,size,chunk_size=65536,max_attempts=4,progress=None):
    output=Path(output)
    part=Path(str(output)+'.part')
    if output.exists() or part.exists(): raise FileExistsError('Use a new backup path')
    if min(size,chunk_size,max_attempts)<1: raise ValueError('Positive backup bounds required')
    digest=hashlib.sha256()
    retries=0
    with part.open('xb') as target:
        for offset in range(0,size,chunk_size):
            length=min(chunk_size,size-offset)
            for attempt in range(max_attempts):
                try:
                    data=read(offset,length)
                    if not isinstance(data,bytes) or len(data)!=length:
                        raise OSError('Short Flash read')
                    break
                except OSError:
                    if attempt+1==max_attempts: raise
                    retries+=1
                    reconnect()
            target.write(data)
            digest.update(data)
            if progress: progress(offset+length,size,retries)
        target.flush()
        os.fsync(target.fileno())
    # Full bytes are durable before publication; keep interrupted .part evidence.
    part.rename(output)
    return dict(complete=True,size_bytes=size,sha256=digest.hexdigest(),retries=retries,
                verification='esptool MD5 per successful read chunk; SHA256 for full local image')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baud',type=int,default=460800)
    args=parser.parse_args()
    import esptool
    import serial
    loader=None
    def connect():
        nonlocal loader
        if loader is not None: loader._port.close()
        loader=esptool.detect_chip(args.port,baud=115200)
        if loader.CHIP_NAME!='ESP32-S3': raise RuntimeError('Expected ESP32-S3')
        loader=esptool.run_stub(loader)
        loader.change_baud(args.baud)
        # Configure the actual detected 16 MB NOR device in volatile loader state.
        flash_id=loader.flash_id()
        if (flash_id>>16)&0xff != 0x18: raise RuntimeError('Expected a 16 MB Flash device')
        loader.flash_set_parameters(16*1024*1024)
    def read(offset,length):
        try: return loader.read_flash(offset,length)
        except (esptool.FatalError,serial.SerialException) as exc: raise OSError(str(exc)) from exc
    last_percent=-1
    def progress(done,total,retries):
        nonlocal last_percent
        percent=done*100//total
        if percent!=last_percent:
            print(json.dumps(dict(percent=percent,retries=retries)),flush=True)
            last_percent=percent
    try:
        connect()
        report=copy_flash(read,connect,args.out,size=16*1024*1024,progress=progress)
        report.update(port=args.port,baud=args.baud,chip='ESP32-S3',esptool_version=esptool.__version__)
        with args.out.with_suffix('.json').open('x',encoding='utf-8') as stream:
            json.dump(report,stream,indent=2)
            stream.write('\n')
        print(json.dumps(report),flush=True)
    finally:
        if loader is not None:
            try: loader.hard_reset()
            finally: loader._port.close()


if __name__=='__main__': main()
