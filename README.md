## Reinforcement Learning for Light Transport Rendering


This program aims to reproduce the Dahm & Keller (2017) Reinforcement Learning method (Learning Light Transport the Reinforcement Learning Way) for rendering. The method is based on the idea of using a reinforcement learning algorithm to learn how to sample light paths in a scene, which can lead to more efficient rendering. 

The implementation is done in Python and uses mitsuba 3, a differentiable physically based renderer. The code is structured in a way that allows for easy experimentation and modification of the reinforcement learning algorithm.

The main components of the code include:
- A scene representation that defines the geometry, materials, and light sources in the scene.
- A reinforcement learning agent that learns to sample light paths based on the rewards received from the rendering process.
- A rendering loop that iteratively updates the agent's policy and renders the scene using the learned sampling strategy.

# Dependency Installation
To run the program, you will need to have Python installed along with the necessary dependencies, including mitsuba 3.

 You can install the required dependencies using pip:
```
pip install mitsuba
```

# Usage
To run the program, simply execute the main script:
```
python render_cbox_rl.py
```

You should obtain four images:
- render_result: The final rendered image using the reinforcement learning method.
- render_no_update: A reference image rendered using the same algorithm but without updating the policy, serving as a baseline for comparison.
- render_no_guiding: A reference image rendered using a standard path tracing method without any guiding, serving as another baseline for comparison.
- render_result_mi: An image rendered using the multiple importance sampling (MIS) technique, which is a common method for improving the efficiency of path tracing.

