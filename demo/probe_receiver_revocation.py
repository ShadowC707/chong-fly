"""Minimal existing receiver behavior; no physics, tuning or hardware."""
import json
from edge.session_receiver import SessionReceiver, CommandPacket
from edge.flight_bridge import CommandLease


def probe():
    r=SessionReceiver(); r.reset(session_id='revocation-probe',now_s=0.)
    r.set_mode('AUTO',now_s=0.); r.announce_sensor(0,now_s=0.)
    first=r.receive(CommandPacket(r.session_id,0,
        CommandLease(0,(1213.,1500.,1500.,1500.),.04)),now_s=0.)
    r.set_mode('MANUAL',now_s=.004)
    revoked=dict(reason=r.reason,tripped=r.gate.tripped,pwm=r.gate.pwm)
    r.set_mode('AUTO',now_s=.02); r.announce_sensor(1,now_s=.02)
    second=r.receive(CommandPacket(r.session_id,1,
        CommandLease(1,(1213.,1500.,1500.,1500.),.04)),now_s=.02)
    return dict(first_accepted=first.accepted_pwm is not None,after_manual=revoked,
                reentry_accepted=second.accepted_pwm is not None,reentry_reason=second.reason,
                reentry_tripped=second.tripped)


if __name__=='__main__': print(json.dumps(probe(),indent=2))
