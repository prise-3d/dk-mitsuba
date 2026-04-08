## Reinforcement Learning for Light Transport Rendering


This program aims to reproduce the Dahm & Keller (2017) Reinforcement Learning method (Learning Light Transport the Reinforcement Learning Way) for rendering. The method is based on the idea of using a reinforcement learning algorithm to learn how to sample light paths in a scene, which can lead to more efficient rendering. 

The implementation is done in Python and uses mitsuba 3, a differentiable physically based renderer. The code is structured in a way that allows for easy experimentation and modification of the reinforcement learning algorithm.

The main components of the code include:
- A scene representation that defines the geometry, materials, and light sources in the scene.
- A reinforcement learning agent that learns to sample light paths based on the rewards received from the rendering process.
- A rendering loop that iteratively updates the agent's policy and renders the scene using the learned sampling strategy.

