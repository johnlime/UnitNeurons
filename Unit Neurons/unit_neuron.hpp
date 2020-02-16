//
//  unit_neuron.hpp
//  Unit Neurons
//
//  Created by John Lime on 2020/02/07.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#ifndef unit_neuron_hpp
#define unit_neuron_hpp

#include <stdio.h>
#include <stdlib.h>
#pragma once

//class UnitNeuron{
//public:
//};

class FloatUnitNeuron{
protected:
    float* memory;                  // information needed to compute feedforward and feedback functions
    
public:
    unsigned int num_prev;          // dimension of input signals
    FloatUnitNeuron* previous;      // array of neurons' pointers that the current neuron references signals from
    float state;    // signal emitted from the current neuron
    virtual void feedforward() = 0;
    virtual void feedback(float* ff_input, float* fb_input) = 0;
};

class FloatGlobalOperator{
protected:
    float* memory;
    
public:
    virtual void execute() = 0;
};

#endif /* unit_neuron_hpp */
