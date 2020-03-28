//
//  float_mapping_neuron.cpp
//  Unit Neurons
//
//  Created by John Lime on 2020/02/12.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#include "kohonen_som.hpp"
#include <math.h>

void FloatMappingNeuron:: init(FloatUnitNeuron** _prevs, int _num_prev, int _max)
{
    previous = _prevs;                                      // assign array of input neurons' references
    num_prev = _num_prev;                                   // assign number of input neurons
    memory = (float*) malloc(num_prev);                     // allocate memory for storing weight values with dim of input neurons
    
    if (_max == 0)
    {
        for (int i = 0; i < num_prev; i++)
        {
            memory[i] = rand();
        }
    }
    
    else
    {
        for (int i = 0; i < num_prev; i++)
        {
            memory[i] = ((float) rand() / RAND_MAX) * _max;
        }
    }
}

FloatMappingNeuron:: FloatMappingNeuron(FloatUnitNeuron** _prevs, int _num_prev, int _max)
{
    init(_prevs, _num_prev, _max);
}

FloatMappingNeuron:: FloatMappingNeuron(FloatUnitNeuron** _prevs, int _num_prev)
{
    init(_prevs, _num_prev, 0);
}

void FloatMappingNeuron:: assign_neighbors(FloatMappingNeuron** _neighbors, int _num_neighbors)
{
    neighbors = _neighbors;     // assign array of neighboring mapping neurons' references
    num_neighbors = _num_neighbors;
}

void FloatMappingNeuron:: feedforward()
{
    // reset counter_max
    counter_first = true;
    // calculate distance squared
    float dist = 0.0f;
    for (int i = 0; i < num_prev; i++){
        dist += (memory[i] - previous[i]->state) * (memory[i] - previous[i]->state);
    }
    state = sqrt(dist);
}

// activated by global operator
void FloatMappingNeuron:: feedback(float* fb_input)    // fb = {current count, max count}
{
    // check that input neighbor range count is lowest
    if (counter_first && fb_input[0] >= 0){   // check whether this is the first time a signal is received from neighbor
        counter = fb_input[0];      // set default counter as the input
        counter_first = false;      // first neighboring signal operation is over
        
        // calculate loss and update memory
        for (int i = 0; i < num_prev; i++){
            memory[i] += pow(lr, fb_input[1] - fb_input[0]) * (previous[i]->state - memory[i]) / state;    // normalie loss
        }
        // reduce neighbor range count
        counter -= 1;
        float fb [2] = {counter, fb_input[1]};
        // activate feedback of neighboring neurons
        for (int i = 0; i < num_neighbors; i++){
            neighbors[i]->feedback(fb);
        }
    }
    else{
        return;
    }
}

float* FloatMappingNeuron:: see_memory()
{
    return memory;
}
