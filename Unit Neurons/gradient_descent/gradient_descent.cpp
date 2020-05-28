//
//  gradient_descent.cpp
//  Unit Neurons
//
//  Created by John Lime on 2020/03/28.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#include "gradient_descent.hpp"
#include <math.h>

FloatGradientDescent:: FloatGradientDescent(FloatFeedForwardNeuron** _targets, int _num_targets)
{
    targets = _targets;
    num_targets = _num_targets;
    grad_loss = (float*) malloc(num_targets);
}

FloatGradientDescent:: ~FloatGradientDescent()
{
    delete [] grad_loss;
}

void FloatGradientDescent:: calculate_l1_loss(float correct_value)
{
    if (num_targets != 1)
    {
        throw std::invalid_argument("Output dimension of neural network does not match dimension of correct value(s)");
    }
    
    float target_output = targets[num_targets - 1]->state;
    grad_loss[0] = correct_value - target_output;
}

void FloatGradientDescent:: calculate_l1_loss(float correct_value, float coef)
{
    calculate_l1_loss(correct_value);
    grad_loss[0] *= coef;
}

void FloatGradientDescent:: calculate_l1_loss(float* correct_value)
{
    float target_output [num_targets];
    for (int i = 0; i < num_targets; i++)
    {
        target_output[i] = targets[i]->state;
    }
    
    for (int i = 0; i < num_targets; i++)
    {
        grad_loss[i] = correct_value[i] - target_output[i];
    }
}

void FloatGradientDescent:: calculate_l1_loss(float* correct_value, float* coef)
{
    calculate_l1_loss(correct_value);
    for (int i = 0; i < num_targets; i++)
    {
        grad_loss[i] *= coef[i];
    }
}

void FloatGradientDescent:: calculate_l1_loss(int* indices, int length, float* correct_value, float* coef)
{
    float target_output [num_targets];
    for (int i = 0; i < num_targets; i++)
    {
        target_output[i] = targets[i]->state;
    }

    // gradient should be zero at default
    for (int i = 0; i < num_targets; i++)
    {
        grad_loss[i] = 0;
    }

    for (int i = 0; i < length; i++)
    {
        grad_loss[indices[i]] = (correct_value[indices[i]] - target_output[indices[i]]) * coef[indices[i]];
    }
}

void FloatGradientDescent:: calculate_cross_entropy_loss(float* correct_value)
{
    float target_output [num_targets];
    float sum_of_softmax = 0.0f;
    for (int i = 0; i < num_targets; i++)
    {
        target_output[i] = targets[i]->state;
        sum_of_softmax += exp(target_output[i]);
    }
    
    // correct value should be one hot encoded
    for (int i = 0; i < num_targets; i++)
    {
        // calculate gradient of softmax
        grad_loss[i] = correct_value[i] - exp(target_output[i]) / sum_of_softmax;
    }
}

void FloatGradientDescent:: calculate_cross_entropy_loss(float* correct_value, float* coef)
{
    calculate_cross_entropy_loss(correct_value);
    for (int i = 0; i < num_targets; i++)
    {
        grad_loss[i] *= coef[i];
    }
}

void FloatGradientDescent:: calculate_cross_entropy_loss(int index, float coef)
{
    float target_output [num_targets];
    float sum_of_softmax = 0.0f;
    for (int i = 0; i < num_targets; i++)
    {
        target_output[i] = targets[i]->state;
        sum_of_softmax += exp(target_output[i]);
    }
    
    // gradient should be 0 at default
    for (int i = 0; i < num_targets; i++)
    {
        grad_loss[i] = 0;
    }
    
    // calculate gradient of softmax
    grad_loss[index] = (1 - exp(target_output[index]) / sum_of_softmax) * coef;
}

void FloatGradientDescent:: execute()
{
    // update final layer only
    for (int i = 0; i < num_targets; i++)
    {
        float tmp_loss [1] = {grad_loss[i]};
        targets[i]->feedback(tmp_loss);
    }
}
