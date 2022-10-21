import numpy as np
from d4rl.hand_manipulation_suite import RelocateEnvV0

ADD_BONUS_REWARDS = True


class MyRelocateEnvV0(RelocateEnvV0):
    def __init__(self, random_start=False, terminate_on_success=False, **kwargs):
        self.random_start = random_start
        self.terminate_on_success = terminate_on_success
        super().__init__(**kwargs)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()

        reward = -0.1 * np.linalg.norm(palm_pos - obj_pos)  # take hand to object
        if obj_pos[2] > 0.04:  # if object off the table
            reward += 1.0  # bonus for lifting the object
            reward += -0.5 * np.linalg.norm(
                palm_pos - target_pos
            )  # make hand go to target
            reward += -0.5 * np.linalg.norm(
                obj_pos - target_pos
            )  # make object go to target

        if ADD_BONUS_REWARDS:
            if np.linalg.norm(obj_pos - target_pos) < 0.1:
                reward += 10.0  # bonus for object close to target
            if np.linalg.norm(obj_pos - target_pos) < 0.05:
                reward += 20.0  # bonus for object "very" close to target

        goal_achieved = True if np.linalg.norm(obj_pos - target_pos) < 0.1 else False

        if self.terminate_on_success:
            done = goal_achieved
        else:
            done = False

        return (
            ob,
            reward,
            done,
            dict(goal_achieved=goal_achieved, is_success=goal_achieved),
        )

    def reset_model(self):
        qp = self.init_qpos.copy()

        # qp
        if self.random_start:
            # Randomize all qpos
            qp += self.np_random.uniform(low=-0.1, high=0.1, size=qp.shape)

        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        # Don't randomize object and target
        # self.model.body_pos[self.obj_bid,0] = self.np_random.uniform(low=-0.15, high=0.15)
        # self.model.body_pos[self.obj_bid,1] = self.np_random.uniform(low=-0.15, high=0.3)
        # self.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=-0.2, high=0.2)
        # self.model.site_pos[self.target_obj_sid,1] = self.np_random.uniform(low=-0.2, high=0.2)
        # self.model.site_pos[self.target_obj_sid,2] = self.np_random.uniform(low=0.15, high=0.35)
        self.model.body_pos[self.obj_bid, 0] = 0
        self.model.body_pos[self.obj_bid, 1] = 0
        self.model.site_pos[self.target_obj_sid, 0] = 0.1
        self.model.site_pos[self.target_obj_sid, 1] = 0.1
        self.model.site_pos[self.target_obj_sid, 2] = 0.25

        self.sim.forward()
        return self.get_obs()
