import argparse
from os.path import join, exists
import numpy as np
import sys

from stable_baselines3 import PPO
from dream_wrap import CustomEnv

sys.path.append('/home/cak/enes/world-models-with-RL')

def generate_data(rollouts, data_dir, model_dir): # pylint: disable=R0914
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."

    # env = gym.make("CarRacing-v2")
    env = CustomEnv()
    seq_len = env._max_episode_steps


    for i in range(rollouts):
        s, _ =env.reset()
        model = PPO.load(model_dir) # , env=env maybe this is not required
    
        s_rollout = []
        r_rollout = []
        d_rollout = []
        a_rollout = []

        t = 0
        while True:
            action = model.predict(s)[0]
            t += 1

            # s, r, done, _ = env.step(action)
            s, r, terminated, truncated, _ = env._based_step(action)         
            done = terminated or truncated

            # env.env.viewer.window.dispatch_events()
            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]
            a_rollout += [action]

            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--model-dir', type=str, help='Pretrained model directory for PPO')
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.model_dir)
