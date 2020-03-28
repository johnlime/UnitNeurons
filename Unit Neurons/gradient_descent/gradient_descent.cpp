//
//  gradient_descent.cpp
//  Unit Neurons
//
//  Created by John Lime on 2020/03/28.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#include "gradient_descent.hpp"

FloatGradientDescent:: FloatGradientDescent(FloatFeedForwardNeuron** _targets, int _num_targets, int* _layer_sizes, int _num_layers)
{
    targets = _targets;
    num_targets = _num_targets;
    layer_sizes = _layer_sizes;
    num_layers = _num_layers;
}

void FloatGradientDescent:: calculate_l1_loss(float correct_value, float target_output)
{
    loss = correct_value - target_output;
}

void FloatGradientDescent:: execute()
{
    int i = 0;
    while (i < num_targets)
    {
        for (int j = 0; j < num_layers; j++)
        {
            targets[i]->feedback(&loss);
            i += 1;
        }
    }
}
