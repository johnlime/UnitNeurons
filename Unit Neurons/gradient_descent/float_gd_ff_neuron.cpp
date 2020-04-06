//
//  float_gd_ff_neuron.cpp
//  Unit Neurons
//
//  Created by John Lime on 2020/03/26.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#include <math.h>
#include "gradient_descent.hpp"

FloatFeedForwardNeuron:: FloatFeedForwardNeuron(FloatUnitNeuron** _prevs, int _num_prevs, FeedbackQueryManager* _query_manager, std::string const &_activ)
{
    previous = _prevs;
    num_prev = _num_prevs;
    memory = (float*) malloc(num_prev * sizeof(float));     // synaptic weights
    query_manager = _query_manager;
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
    
    else if (_activ == "tanh")
    {
        activation = [](float x)
        {
            return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
        };
        
        activ_deriv = [](float x)
        {
            return 1 - ((exp(x) - exp(-x)) / (exp(x) + exp(-x)) * (exp(x) - exp(-x)) / (exp(x) + exp(-x)));
        };
    }
    
    else
    {
        throw std::invalid_argument("No such activation function " + _activ + " found");
    }
}

FloatFeedForwardNeuron:: FloatFeedForwardNeuron(FloatUnitNeuron** _prevs, int _num_prevs, FeedbackQueryManager* _query_manager, float (*_activation) (float), float (*_gradient) (float))
{
    previous = _prevs;
    num_prev = _num_prevs;
    memory = (float*) malloc(num_prev);     // synaptic weights
    query_manager = _query_manager;
    for (int i = 0; i < num_prev; i++)
    {
        memory[i] = (float) rand() / RAND_MAX;
    }
    activation = _activation;
    activ_deriv = _gradient;
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
    float new_fb [num_prev];
    for (int i = 0; i < num_prev; i++)
    {
        new_fb[i] = previous[i]->state * (*fb_input);
        memory[i] += lr * activ_deriv(pre_activ) * new_fb[i];     // pd_activ / pd_pre_activ * pd_pre_activ / pd_weight * L1_loss
    }
    
    // feedback to previous neurons in query (presumably in parallel)
    for (int i = 0; i < num_prev; i++)
    {
        FeedbackQuery tmp;
        tmp.neuron = previous[i];
        float tmp_fb_input [1] = {fb_input[i]};
        tmp.fb_input = tmp_fb_input;
        query_manager->add_query(tmp);
    }
}
