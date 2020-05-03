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
protected:
    void update_memory(float* fb_input, float* new_fb);
    
public:
    FloatInputNeuron();
    void feedforward();
    void feedback(float* fb_input);
    
    void assign_value(float value);
};

#endif /* input_output_hpp */
