//
//  input_output.cpp
//  Unit Neurons
//
//  Created by John Lime on 2020/02/17.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#include "input_output.hpp"

FloatInputNeuron:: FloatInputNeuron(float* _memory)
{
    memory = _memory;
    current_idx = 0;
}

void FloatInputNeuron:: feedforward()
{
    state = memory[current_idx];
}

void FloatInputNeuron:: feedback(float *_, float *__)
{
    current_idx += 1;
}
