"""Explicit experimental integration; shared pinned FlightBridge is untouched."""
from edge.system import FlightSystem


class YawHandoverSystem(FlightSystem):
    def tick(self, *args, **kwargs):
        result = super().tick(*args, **kwargs)
        if result['reason'] == 'engaged' and result['lease'] is not None:
            value = float(result['lease'].pwm[3])
            self.runtime.coordinator.synchronize_issued_yaw(value)
            result['yaw_handover'] = dict(source='issued', yaw_pwm=value,
                                          sequence=result['lease'].sequence)
        return result


def install(system, *, profile='phased_smooth'):
    from .controllers import install as install_profile
    return install_profile(system, profile=profile)
