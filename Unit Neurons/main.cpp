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
#include <math.h>
#define CITIES 10
#define EPOCHS 200

int main(int argc, const char * argv[]) {
    // define input neurons
    float x_memory [CITIES];
    float y_memory [CITIES];
    for (int i = 0; i < CITIES; i++){
        x_memory[i] = sin(2 * M_PI / CITIES * i);
        y_memory[i] = sin(2 * M_PI / CITIES * i);
    }
    FloatInputNeuron* x_input = new FloatInputNeuron(x_memory);
    FloatInputNeuron* y_input = new FloatInputNeuron(y_memory);
    FloatInputNeuron* io_neuron [2] = {x_input, y_input};
    
    // define mapping neurons
    FloatMappingNeuron* maps [CITIES];
    for (int i = 0; i < CITIES; i++){
        FloatMappingNeuron tmp = FloatMappingNeuron((FloatUnitNeuron**) io_neuron);
        maps[i] = &tmp;
    }
    
    // assign neighboring neurons
    maps[0]->assign_neighbors(&maps[1]);
    for (int i = 1; i < CITIES - 1; i++){
        FloatMappingNeuron* tmp [2] = {maps[i-1], maps[i+1]};
        maps[i]->assign_neighbors(tmp);
    }
    maps[CITIES - 1]->assign_neighbors(&maps[CITIES - 2]);
    
    // define global operator
    FloatKohonenSOM global_operator = FloatKohonenSOM(maps, 3);
    /* loop through dataset */
    for (int i = 0; i < EPOCHS; i++){
        // feedforward
        for(int i = 0; i < 2; i++){
            io_neuron[i]->feedforward();
            for (int i = 0; i < CITIES; i++){
                maps[i]->feedforward();
            }
        }
        // feedback
        global_operator.execute();
    }
    
    return 0;
}
