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
#include "fb_query_manager.hpp"
#include <string>
#include <sstream>
#include <stdexcept>

class FloatFeedForwardNeuron: public FloatUnitNeuron{
protected:
    float (*activation) (float);        // activation function
    float (*activ_deriv) (float);       // derivative of the activation function
    float pre_activ;
    FeedbackQueryManager* query_manager;
    
public:
    float lr = 0.7f;                    // learning rate (hyperparameter)
    FloatFeedForwardNeuron(FloatUnitNeuron** _prevs, int _num_prevs, FeedbackQueryManager* _query_manager, float (*_activation) (float), float (*_gradient) (float));
    FloatFeedForwardNeuron(FloatUnitNeuron** _prevs, int _num_prevs, FeedbackQueryManager* _query_manager, std::string const &_activ);
    void feedforward();
    float* update_memory(float* fb_input);
    void feedback(float* fb_input);
};

class FloatGradientDescent: FloatGlobalOperator{
private:
    FloatFeedForwardNeuron** targets;   // feed forward neurons in the order of input to output
    int num_targets;
    int* layer_sizes;                   // size of feed forward neural network layers in the order of input to output
    int num_layers;
    float* grad_loss;
    
public:
    FloatGradientDescent(FloatFeedForwardNeuron** _targets, int _num_targets, int* _layer_sizes, int _num_layers);
    void calculate_l1_loss(float correct_value);
    void calculate_l1_loss(float correct_value, float coef);
    void calculate_l1_loss(float* correct_value);
    void calculate_l1_loss(float* correct_value, float* coef);
    void calculate_l1_loss(int* indices, int length, float* correct_value, float* coef);
    void calculate_cross_entropy_loss(float* correct_value);
    void calculate_cross_entropy_loss(float* correct_value, float* coef);
    void calculate_cross_entropy_loss(int index, float coef);
    void execute();
};

float* softmax (float* x, int size);

#endif /* gradient_descent_hpp */
