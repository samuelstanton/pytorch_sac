import numpy as np


def gym_hopper(obses, actions, next_obses):
        assert len(obses.shape) == len(next_obses.shape) == len(actions.shape) == 2

        height = next_obses[:, 0]
        angle = next_obses[:, 1]
        not_done = np.isfinite(next_obses).all(axis=-1) \
                   * np.abs(next_obses[:, 1:] < 100).all(axis=-1) \
                   * (height > .7) \
                   * (np.abs(angle) < .2)

        done = ~not_done
        done = done[:,None]
        return done


def gym_walker2d(obses, actions, next_obses):
        assert len(obses.shape) == len(next_obses.shape) == len(actions.shape) == 2

        height = next_obses[:, 0]
        angle = next_obses[:, 1]
        not_done = (height > 0.8) \
                   * (height < 2.0) \
                   * (angle > -1.0) \
                   * (angle < 1.0)
        done = ~not_done
        done = done[:,None]
        return done


def gym_halfcheetah(obses, actions, next_obses):
        assert len(obses.shape) == len(next_obses.shape) == len(actions.shape) == 2

        done = np.array([False]).repeat(len(obses))
        done = done[:,None]
        return done
