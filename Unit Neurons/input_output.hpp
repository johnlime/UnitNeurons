//
//  input_output.hpp
//  Unit Neurons
//
//  Created by John Lime on 2020/02/17.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#ifndef input_output_hpp
#define input_output_hpp

#include <stdio.h>
#include "unit_neuron.hpp"

class FloatInputNeuron: FloatUnitNeuron{
public:
    FloatInputNeuron(float* _memory);   // assign input dataset/trajectory
    void feedforward();                 // take current sample from dataset
    void feedback(float* _, float* __); // move up one index in dataset
    
private:
    unsigned int current_idx;           // current index
    unsigned int num_prev = 0;
    FloatUnitNeuron* previous = 0;
};

#endif /* input_output_hpp */
