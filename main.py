import q_learning as q_learning_agent
import nonrl as non_rl_agent
import DQN as dqn_agent
import PPO as ppo_agent

def run_all():
    print("Running Q-Learning...")
    q_learning_agent.train_q_learning()
    
    print("\nRunning Non-RL...")
    non_rl_agent.run_non_rl()

    print("\nRunning DQN...")
    dqn_agent.train_dqn()

    print("\nRunning PPO...")
    ppo_agent.train_ppo()

if __name__ == "__main__":
    run_all()
