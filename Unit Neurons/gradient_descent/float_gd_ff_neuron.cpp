//
//  float_gd_ff_neuron.cpp
//  Unit Neurons
//
//  Created by John Lime on 2020/03/26.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#include "gradient_descent.hpp"

FloatFeedForwardNeuron:: FloatFeedForwardNeuron(FloatUnitNeuron** _prevs, int _num_prevs, const std::string _activ)
{
    previous = _prevs;
    num_prev = _num_prevs;
    memory = (float*) malloc(num_prev);     // synaptic weights
    for (int i = 0; i < num_prev; i++)
    {
        memory[i] = (float) rand() / RAND_MAX;
    }
    
    if (_activ == "identity")
    {
        activation = [](float x)
        {
            return x;
        };
        
        activ_deriv = [](float x)
        {
            return 1.0f;
        };
    }
    
    else if (_activ == "relu")
    {
        activation = [](float x)
        {
            if (x < 0)
            {
                x = 0.0f;
            }
            return x;
        };
        
        activ_deriv = [](float x)
        {
            if (x > 0.0f)
            {
                return 1.0f;
            }
            else
            {
                return 0.0f;
            }
        };
    }
    
    else
    {
        throw std::invalid_argument("No such activation function " + _activ + " found");
    }
}

void FloatFeedForwardNeuron:: feedforward()
{
    pre_activ = 0.0f;
    for (int i = 0; i < num_prev; i++)
    {
        pre_activ += memory[i] * previous[i]->state;
    }
    state = activation(pre_activ);
}

void FloatFeedForwardNeuron:: feedback(float *fb_input)
{
    for (int i = 0; i < num_prev; i++)
    {
        memory[i] += activ_deriv(pre_activ) * previous[i]->state * (*fb_input);     // pd_activ / pd_pre_activ * pd_pre_activ / pd_weight * L1_loss
    }
    
    // feedback to previous neurons outside of this function presumably in parallel
}
