"""Compatibility entrypoint for explicit read-only import smoke checks."""
import importlib

MODULES = ('edge.runtime', 'edge.system', 'edge.sensors', 'edge.session_receiver',
           'edge.transport', 'artifacts.io', 'simulation.sensor_schedule')


def main():
    for name in MODULES:
        importlib.import_module(name)
    print('Read-only import smoke passed: '+', '.join(MODULES))


if __name__ == '__main__':
    main()
