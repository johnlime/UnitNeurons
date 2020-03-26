//
//  input_output.cpp
//  Unit Neurons
//
//  Created by John Lime on 2020/02/17.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#include "input_output.hpp"

FloatInputNeuron:: FloatInputNeuron()
{
    memory = nullptr;
}

void FloatInputNeuron:: feedforward()
{
    return;
}

void FloatInputNeuron:: feedback(float* fb_input)
{
    return;
}

void FloatInputNeuron:: assign_value(float value)
{
    state = value;
}
