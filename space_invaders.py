import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

def update_environment(env, action):
    """
    Function to update the environment based on the given action.
    
    Parameters:
    - env: OpenAI Gym environment object
    - action: Action to take in the environment
    
    Returns:
    - observation: Current observation after taking the action
    - reward: Reward received after taking the action
    - done: Boolean indicating if the episode is done
    - info: Additional information from the environment
    """
    result = env.step(action)
    if len(result) == 4:
        observation, reward, done, info = result
    elif len(result) == 5:  # If additional values are returned
        observation, reward, done, info, _ = result  # Ignore additional values
    else:
        raise ValueError("Unexpected number of return values from env.step()")
    return observation, reward, done, info

def visualize_environment(env):
    """
    Function to visualize the environment.
    
    Parameters:
    - env: OpenAI Gym environment object
    """
    fig = plt.figure()
    img = plt.imshow(env.render())
    plt.axis('off')

    def animate(i):
        img.set_array(env.render())
        return (img,)

    anim = animation.FuncAnimation(fig, animate, frames=100, interval=50)
    display(display_animation(anim))

if __name__ == '__main__':
    print("Creating Game")
    # Example usage:
    env = gym.make('SpaceInvaders-v0', render_mode='rgb_array')
    env.reset()

    # Define video codec and output file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    episodes = 5

    for episode in range(episodes):
        video_out = cv2.VideoWriter(f'episodes{episode}_video.mp4', fourcc, 30.0, (env.render().shape[1], env.render().shape[0]))

        state = env.reset()
        done = False 
        score = 0
        frames = []  # Store frames for the current episode


        while not done: 
            img = env.render()  # Ensure 'rgb_array' mode for rendering
            frames.append(img)
            action = random.choice([0,1,2,3,4,5])
            n_state, reward, done, info, _ = env.step(action)
            score += reward
        
        # Convert frames to video
        for frame in frames:
            video_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        print(f'Episode: {episode} Score: {score}')
        video_out.release()
        cv2.destroyAllWindows()
