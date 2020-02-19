//
//  kohonen_som.cpp
//  Unit Neurons
//
//  Created by John Lime on 2020/02/12.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#include "kohonen_som.hpp"

FloatMappingNeuron:: FloatMappingNeuron(FloatUnitNeuron* _prevs)
{
    previous = _prevs;                                      // assign array of input neurons' references
    num_prev = sizeof(_prevs) / sizeof(FloatUnitNeuron*);   // calculate number of elements in pointer array
    memory = (float*) malloc(num_prev);                     // allocate memory for storing weight values with dim of input neurons
}

void FloatMappingNeuron:: assign_neighbors(FloatMappingNeuron* _neighbors)
{
    neighbors = _neighbors;     // assign array of neighboring mapping neurons' references
}

void FloatMappingNeuron:: feedforward()
{
    // reset counter_max
    counter_first = true;
    // calculate distance squared
    float dist = 0.0f;
    for (unsigned int i = 0; i < num_prev; i++){
        dist +=
        (memory[i] - previous[i].state) * (memory[i] - previous[i].state);
    }
    state = dist;
}

void FloatMappingNeuron:: feedback(     // activated by global operator
    float* ff_input,    // original input neuron values
    float* fb_input)    // input neighbor range count
{
    // check that input neighbor range count is lowest
    if (counter_first){   // check whether this is the first time a signal is received from neighbor
        counter = fb_input[0];      // set default counter as the input
        counter_first = false;      // first neighboring signal operation is over
    }
    else if (counter < fb_input[0]){
        return;
    }
    else{
        // calculate loss and update memory
        for (int i = 0; i < num_prev; i++){
            memory[i] += lr * (ff_input[i] - memory[i]);
        }
        // reduce neighbor range count
        counter -= 1;
        // activate feedback of neighboring neurons
        for (int i = 0; i < sizeof(neighbors) / sizeof(FloatMappingNeuron*); i++){
            neighbors[i].feedback(ff_input, &counter);
        }
    }
}

FloatKohonenSOM:: FloatKohonenSOM(FloatMappingNeuron* _maps, unsigned int _neighbor_range){
    maps = _maps;
    neighbor_range = _neighbor_range;
}

void FloatKohonenSOM:: execute()
{
    FloatMappingNeuron* winner = &maps[0];
    float shortest = maps[0].state;
    
    for (int i = 1; i < sizeof(maps) / sizeof(FloatMappingNeuron*); i++){
        if (shortest > maps[i].state){
            winner = &maps[i];
            shortest = maps[i].state;
        }
    }
    float fb [1];
    fb[0] = neighbor_range;
    float ff [maps[0].num_prev];
    for (int i = 0; i < maps[0].num_prev; i++){
        ff[i] = maps[0].previous[i].state;
    }
    winner->feedback(ff, fb);
}
