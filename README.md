# Unit Neurons
Unit Neurons is a repository for development of a C++ neuron-based neural network library
where each neuron is expressed using object instances embedded with its own states and functionalities,
in hopes of gaining more understanding of neural nets through the perspective of complex systems.

We call for contributions on further development of the library by adding more functionalities, fixing bugs, etc.

## Vision
Artifical neural networks are models that attempt to imitate features and functionalities of biological
neural networks. Preexisting neural network libraries such as TensorFlow, PyTorch, and Keras strictly use
matrix multiplication and neurons expressed in layers to execute feedforward and feedack loops.

However, in reality, a biological neural network is a network of mutually interacting neurons with their own function
of calculating the signal output given the signal input. In our library, we treat each neuron as an object instance
that includes a state, a feedforward method, and a feedback method.

## Structure of a Unit Neuron
An abstract unit neuron class includes a protected array `memory`, which stores values unique to each neuron which is
used for calculating forward and feedback loops, such as postsynaptic weights, a public variable `state`, which
indicates the current output signal of the neuron, and an array of unit neurons' pointers `previous`, which stores
the neurons that the current neuron inputs the external signals from. Public methods `feedforward()` and
`feedback(float* fb_input)` are virtual methods which are required to be defined by its subclasses.

You can take a look at the abstract class
[here](https://github.com/johnlime/unit_neurons/blob/master/Unit%20Neurons/unit_neuron.hpp).

## Current Features
- Made abstract classes for unit neurons that support floating point values (2nd generation neural networks).
- Inter-neuron pointer routing.
- Kohonen's SOM implementation
  - Neighboring neuron assignment for Kohonen's SOM
  - Made input/output neurons and mapping neurons and global operator necessary for Kohonen's Self Organizing Map (SOM).
  - Visualization via Processing (3x3)

  ![Example KSOM output](Processing%20Visualization/kohonen_som_trained/ksom_3x3.png)

- Feedforward network with gradient descent
  - Global operator calculates least mean squares
  - Feedforward and feedback functions outputs partial differentiation of each weights
  - Example training of neural network with 8 hidden neurons activated via ReLU or Tanh function
  - Visualization via Processing (fit to sine wave)

  ![Example GD output](Processing%20Visualization/gradient_descent_sine/gd_sine.png)

## Open Problems
- Implementation of other activation functions (such as Leaky ReLU, etc).

- Implementations of other neural network models are highly welcomed
  - Spiking neural networks
    - Boltzmann machine
    - Hopfield network

## Major Contributors
