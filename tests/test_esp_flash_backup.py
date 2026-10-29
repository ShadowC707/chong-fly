from pathlib import Path
import pytest

from firmware.esp32.backup_flash import copy_flash


def test_backup_retries_only_failed_chunk_and_publishes_complete_bytes(tmp_path):
    calls=[]
    reconnects=[]
    def read(offset,size):
        calls.append((offset,size))
        if len(calls)==2: raise OSError('bad serial frame')
        return bytes([offset//4])*size
    out=tmp_path/'flash.bin'
    report=copy_flash(read,lambda:reconnects.append(True),out,size=12,chunk_size=4)
    assert out.read_bytes()==bytes(4)+bytes([1])*4+bytes([2])*4
    assert calls==[(0,4),(4,4),(4,4),(8,4)]
    assert len(reconnects)==1 and report['retries']==1 and report['complete']


def test_short_or_failed_read_never_becomes_complete_backup(tmp_path):
    out=tmp_path/'flash.bin'
    with pytest.raises(OSError):
        copy_flash(lambda offset,size:b'x',lambda:None,out,size=8,chunk_size=4,max_attempts=2)
    assert not out.exists()
    assert Path(str(out)+'.part').exists()


def test_backup_does_not_overwrite_existing_evidence(tmp_path):
    out=tmp_path/'flash.bin'
    out.write_bytes(b'original')
    with pytest.raises(FileExistsError): copy_flash(lambda a,b:bytes(b),lambda:None,out,size=4,chunk_size=4)
    assert out.read_bytes()==b'original'
