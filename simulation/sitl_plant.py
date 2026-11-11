"""Rigid-body adapters shared by the flight loop and backend comparisons."""
import numpy as np

from simulation.ready_sim_contract import map_motors

DT = .004


class LegacyPlant:
    def __init__(self, p, _asset):
        from simulation.drone_env import DroneDynamicsParams, QuadcopterDynamics
        self.body = QuadcopterDynamics(DroneDynamicsParams(**p), dt=DT)
        self.body.pos[:] = [0, 0, 1]
        self.body.motor_rpms[:] = self.body.hover_rpm

    def state(self):
        b = self.body
        return np.concatenate((b.pos, b.vel, b.quaternion_to_euler(b.quat), b.omega))

    def step(self, motors):
        self.body.step(motors)

    def close(self):
        pass


class BulletPlant:
    def __init__(self, params, asset, *, dt=DT, start=(0., 0., 1.), yaw=0.):
        if dt not in (.004, .002, .001, .0005, .00025):
            raise ValueError('Supported physics steps are 4, 2, 1, 0.5 and 0.25 ms')
        self.dt = dt
        import pybullet as pb
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
        from gym_pybullet_drones.utils.enums import Physics

        class ReferenceAviary(CtrlAviary):
            def _parseURDFParameters(self):
                self.URDF = str(asset.resolve())
                self.G = params['g']
                return super()._parseURDFParameters()

        self.pb, self.params = pb, params
        self.env = ReferenceAviary(initial_xyzs=np.array([start]),
            initial_rpys=np.array([[0., 0., yaw]]),
            pyb_freq=round(1/dt), ctrl_freq=round(1/dt), gui=False, user_debug_gui=False,
            physics=Physics.PYB)
        self.client = self.env.CLIENT
        self.body = int(self.env.DRONE_IDS[0])
        # Disable Bullet's implicit damping: apply the SAME explicit reference law.
        for link in range(-1, pb.getNumJoints(self.body, physicsClientId=self.client)):
            pb.changeDynamics(self.body, link, linearDamping=0, angularDamping=0,
                              physicsClientId=self.client)
        self.rpm = np.full(4, self.env.HOVER_RPM, dtype=np.float32)

    def state(self):
        pb = self.pb
        pos, quat = pb.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
        vel, omega_world = pb.getBaseVelocity(self.body, physicsClientId=self.client)
        rotation = np.array(pb.getMatrixFromQuaternion(quat)).reshape(3, 3)
        return np.concatenate((pos, vel, pb.getEulerFromQuaternion(quat), rotation.T @ omega_world))

    def step(self, motors):
        p, pb, state = self.params, self.pb, self.state()
        self.rpm += self.dt / (self.dt + p['motor_tau']) * (motors * p['max_rpm'] - self.rpm)
        drag = -p['drag_linear'] * state[3:6] * np.linalg.norm(state[3:6])
        pb.applyExternalForce(self.body, -1, drag, state[:3], pb.WORLD_FRAME,
                              physicsClientId=self.client)
        pb.applyExternalTorque(self.body, -1, -p['drag_angular'] * state[9:12],
                               pb.LINK_FRAME, physicsClientId=self.client)
        self.env.step(map_motors(self.rpm)[None, :])

    def close(self):
        self.env.close()
