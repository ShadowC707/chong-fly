"""Explicit simulator-only assembly of the validated research controller."""
from copy import deepcopy
from pathlib import Path
from edge.system import FlightSystem
from edge.runtime import sha256
from simulation.phased_coordinator import PhasedCoordinator
ROOT=Path(__file__).resolve().parents[1]

from .profiles import OVERLAY_PROFILES


def install(system, *, profile='phased_v2', turn_envelope=None):
    if profile not in OVERLAY_PROFILES or (turn_envelope is not None and profile != 'phased_v2'):
        raise ValueError('Unknown or incompatible coordinator overlay')
    old=system.runtime.coordinator
    system_class = FlightSystem
    sources = ['demo/controllers.py']
    navigation = None
    options = dict(anti_rollback=old.anti_rollback, turn_latch=old.turn_latch, config=old.config,
                   stop_hover=old.stop_hover, behavior_config=old.behavior_config, seed=old.seed)
    if profile == 'candidate_a_brake':
        from simulation.threat_gated_xy import ThreatGatedXYCoordinator
        new = ThreatGatedXYCoordinator(**options)
        overlay = 'threat-gated-brake-v1'
        sources.append('simulation/threat_gated_xy.py')
    else:
        sources.append('simulation/phased_coordinator.py')
        options['phased_config'] = getattr(old, 'phased_config', None)
        if profile == 'phased_v2':
            new = PhasedCoordinator(**options)
            overlay = 'brake-turn-clear-v2'
        else:
            from .yaw_experiment import YawHandoverSystem
            from simulation.phased_yaw import SmoothPhasedCoordinator
            from simulation.phased_recovery import RecoveryPhasedCoordinator, PocketEscapeCoordinator
            from simulation.side_clearance import SideClearanceCoordinator
            cls = {'phased_smooth': SmoothPhasedCoordinator, 'phased_recovery': RecoveryPhasedCoordinator,
                   'phased_escape': PocketEscapeCoordinator, 'phased_side_guard': SideClearanceCoordinator,
                   'phased_swath': SideClearanceCoordinator}[profile]
            new = cls(**options)
            overlay = {'phased_smooth': 'cosine-bounded-v1', 'phased_recovery': 'bounded-wait-retry-v1',
                       'phased_escape': 'consistent-direction-v1',
                       'phased_side_guard': 'recent-edge-yaw-restraint-v2',
                       'phased_swath': 'observed-expansion-v1'}[profile]
            system_class = YawHandoverSystem
            sources.extend(('simulation/phased_yaw.py', 'demo/yaw_experiment.py'))
            if profile != 'phased_smooth':
                sources.append('simulation/phased_recovery.py')
            if profile in ('phased_side_guard', 'phased_swath'):
                sources.append('simulation/side_clearance.py')
            if profile == 'phased_swath':
                from simulation.swath_navigation import SwathNavigation
                from .swath_system import SwathSystem
                navigation = SwathNavigation(new, system.config.session_id)
                system_class = SwathSystem
                sources.extend(('simulation/swath_navigation.py', 'edge/free_space_swath.py',
                    'edge/empirical_corridor.py', 'edge/survey_heading.py', 'edge/survey_memory.py',
                    'demo/swath_system.py', 'demo/swath_adapter.py', 'simulation/survey_heading_sensor.py'))
    envelope_contract=None
    if turn_envelope is not None:
        from simulation.turn_envelope import resolve_turn_config
        envelope_contract=turn_envelope.contract(new.phased_config)
        new.phased_config=resolve_turn_config(new.phased_config,turn_envelope)
    m=deepcopy(system.manifest)
    m['version'] = {'phased_v2': 'flight-system-phased-experiment-v2',
                    'phased_smooth': 'flight-system-yaw-shaping-experiment-v1'}.get(
                        profile, 'flight-system-standalone-navigation-v1')
    m['experiment']=dict(research_only=True,profile=profile,overlay=overlay,active_bundle_unchanged=True)
    if system_class is not FlightSystem:
        m['experiment']['handover_sync'] = 'After bridge engagement: issued reference yaw'
    if envelope_contract is not None:
        m['version']='flight-system-turn-envelope-experiment-v1'
        m['experiment']['turn_envelope']=envelope_contract
        m['experiment']['overlay']='brake-turn-clear-v2-with-turn-envelope'
        sources.append('simulation/turn_envelope.py')
    for name in sources:
        p = ROOT/name
        m['executed_modules'][p.name] = dict(path=str(p), sha256=sha256(p))
    m['resolved']['coordinator']=new.contract()
    m['limitations'].append('Experimental coordinator overlay; not deployment admission.')
    system.runtime.coordinator=new
    result = system_class(system.config,system.runtime,m)
    if navigation is not None:
        result.navigation = navigation
    return result

