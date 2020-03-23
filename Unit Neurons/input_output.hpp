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

class FloatInputNeuron :FloatUnitNeuron{
public:
    FloatInputNeuron();
    void feedforward();
    void feedback(float* ff_input, float* fb_input);
    
    void assign_value(float value);
};

#endif /* input_output_hpp */
