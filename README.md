# Fluid flow fields reconstruction using Physics Informed Neural Networks (PINNs)

Physics-Informed Neural Networks (PINNs) represent a novel approach to deep learning, marrying the predictive capabilities of artificial neural networks with the intrinsic structure of physical models. By utilizing a set of governing differential equations to encapsulate the physical behavior, PINNs weave these rules directly into the neural network's architecture. This enables PINNs to employ observed data for solving intricate tasks such as inverse inference or functional synthesis.

In contrast to conventional deep learning techniques, PINNs provide more stable and reliable results by explicitly incorporating physical constraints into the model's architecture. PINNs demonstrate a unique advantage in not requiring extensive training data; they leverage the generally known structure of the system under investigation. Furthermore, their ability to integrate explicit physical knowledge provides the potential not only to build a superior predictive model, but also to unveil previously undiscovered implicit mathematical properties of the system.

The current project revolves around developing a Physics-Informed Neural Network model to reconstruct fluid flow fields around a cylinder using limited data. The project's two primary objectives are as follows:

1. Acquiring data through numerical simulation to establish a reference solution to the problem.

2. Combining the understanding of physics with the learning prowess of neural networks to formulate a predictive model, which will then be evaluated using the acquired data.
   
The resulting model should be capable of predicting fluid flow attributes (such as velocity and pressure) based on the initial conditions and system parameters. This is expected to be achieved with minimal training data gathered through numerical simulation (CFD). The project aims to demonstrate the efficacy of PINNs as a tool for fluid dynamics studies with limited training data, opening doors to a multitude of practical applications.
