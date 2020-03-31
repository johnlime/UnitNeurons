//
//  gradient_descent.hpp
//  Unit Neurons
//
//  Created by John Lime on 2020/03/26.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#ifndef float_gd_ff_neuron_hpp
#define float_gd_ff_neuron_hpp

#include <stdio.h>
#include "unit_neuron.hpp"
#include <string>
#include <sstream>
#include <stdexcept>

class FloatFeedForwardNeuron: public FloatUnitNeuron{
protected:
    float lr = 0.7f;                    // learning rate (hyperparameter)
    float (*activation) (float);        // activation function
    float (*activ_deriv) (float);       // derivative of the activation function
    float pre_activ;
    
public:
    FloatFeedForwardNeuron(FloatUnitNeuron** _prevs, int _num_prevs, float (*_activation) (float), float (*_gradient) (float));
    FloatFeedForwardNeuron(FloatUnitNeuron** _prevs, int _num_prevs, std::string const &_activ);
    void feedforward();
    void feedback(float* fb_input);
};

class FloatGradientDescent: FloatGlobalOperator{
private:
    FloatFeedForwardNeuron** targets;   // feed forward neurons in the order of input to output
    int num_targets;
    int* layer_sizes;                   // size of feed forward neural network layers in the order of input to output
    int num_layers;
    float loss;
    
public:
    FloatGradientDescent(FloatFeedForwardNeuron** _targets, int _num_targets, int* _layer_sizes, int _num_layers);
    void calculate_l1_loss(float correct_value);
    void execute();
};

#endif /* gradient_descent_hpp */
