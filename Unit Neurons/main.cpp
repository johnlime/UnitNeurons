//
//  main.cpp
//  Unit Neurons
//
//  Created by John Lime on 2020/02/07.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#include <iostream>
#include "input_output.hpp"
#include "kohonen_som.hpp"
#define SAMPLES 100

int main(int argc, const char * argv[]) {
    // define input neurons
    float x_memory [SAMPLES];
    float y_memory [SAMPLES];
    for (int i = 0; i < SAMPLES; i++){
        x_memory[i] = rand() % SAMPLES;
        y_memory[i] = rand() % SAMPLES;
    }
    FloatInputNeuron io_neuron [2]
    {
        {x_memory},
        {y_memory}
    };
    // define mapping neurons
    FloatMappingNeuron* maps [25];
    for (int i = 0; i < 25; i++){
        maps[i] = FloatMappingNeuron(io_neuron);
    }
        // assign neighboring neurons
    
    // define global operator
    
    /* loop through dataset
    // feedforward
    
    // feedback
     
     */
    
    return 0;
}
