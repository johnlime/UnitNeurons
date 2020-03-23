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
#define NODE_D 3
#define EPOCHS 1
#define MAX_RANGE 100

int main(int argc, const char * argv[]) {
    // define input neurons
    FloatInputNeuron* x_input = new FloatInputNeuron();
    FloatInputNeuron* y_input = new FloatInputNeuron();
    FloatInputNeuron* io_neuron [2] = {x_input, y_input};
    
    // define mapping neurons
    FloatMappingNeuron* maps [NODE_D * NODE_D];
    for (int i = 0; i < NODE_D * NODE_D; i++){
        FloatMappingNeuron* tmp = new FloatMappingNeuron((FloatUnitNeuron**) io_neuron, 2, MAX_RANGE);
        maps[i] = tmp;
    }
    
    // assign neighboring neurons
    for (int i = 0; i < NODE_D * NODE_D; i++)
    {
        int a, b;
        
        // corners
        if (
            i == 0 ||
            i == NODE_D - 1 ||
            i == NODE_D * (NODE_D - 1) ||
            i == NODE_D * NODE_D - 1
            )
        {
            if (i == 0)
            {
                a = 1;
                b = NODE_D;
            }
            else if (i == NODE_D - 1)
            {
                a = -1;
                b = NODE_D;
            }
            
            else if (i == NODE_D * (NODE_D - 1))
            {
                a = 1;
                b = -NODE_D;
            }
            
            else// if (i == NODE_D * NODE_D - 1)
            {
                a = -1;
                b = -NODE_D;
            }
            
            FloatMappingNeuron* tmp [2];
            tmp[0] = maps[i + a];
            tmp[1] = maps[i + b];
            maps[i]->assign_neighbors(tmp, 2);
            printf("%d, %d, %d\n", i, i+a, i+b);
        }
        
        // edges
        else if (
            i % NODE_D == 0 ||
            i % NODE_D == NODE_D - 1 ||
            int(i / NODE_D) == 0 ||
            int(i / NODE_D) == NODE_D - 1
            )
        {
            // left edge
            if (i % NODE_D == 0)
            {
                a = 1;
                b = NODE_D;
            }
            
            // right edge
            else if (i % NODE_D == NODE_D - 1)
            {
                a = -1;
                b = NODE_D;
            }
            
            // upper edge
            else if (int(i / NODE_D) == 0)
            {
                a = NODE_D;
                b = 1;
            }
            
            // lower edge
            else// if (int(i / NODE_D) == NODE_D - 1)
            {
                a = -NODE_D;
                b = 1;
            }
            
            FloatMappingNeuron* tmp [3];
            tmp[0] = maps[i - b];
            tmp[1] = maps[i + a];
            tmp[2] = maps[i + b];
            maps[i]->assign_neighbors(tmp, 3);
            printf("%d, %d, %d, %d\n", i, i+a, i-b, i+b);
        }
        
        // default
        else
        {
            a = 1;
            b = NODE_D;
            
            FloatMappingNeuron* tmp [4];
            tmp[0] = maps[i - a];
            tmp[1] = maps[i + a];
            tmp[2] = maps[i + b];
            tmp[3] = maps[i - b];
            maps[i]->assign_neighbors(tmp, 4);
            printf("%d, %d, %d, %d, %d\n", i, i+a, i-b, i+b, i-a);
        }
    }
    
    // define global operator
    FloatKohonenSOM global_operator = FloatKohonenSOM(maps, NODE_D * NODE_D, 3);
    
    for (int i = 0; i < NODE_D * NODE_D; i++){
        float* tmp = maps[i]->see_memory();
        printf("{%f, %f}, \n", tmp[0], tmp[1]);
    }
    printf("%s\n", "Train");
    /* loop through dataset */
    for (int i = 0; i < EPOCHS; i++){
        // feedforward
        for(int j = 0; j < 2; j++){
            io_neuron[j]->assign_value(((float)rand() / RAND_MAX) * MAX_RANGE);
        }
        for (int j = 0; j < NODE_D * NODE_D; j++){
            maps[j]->feedforward();
        }
        // feedback
        global_operator.execute();
    }
    
    for (int i = 0; i < NODE_D * NODE_D; i++){
        float* tmp = maps[i]->see_memory();
        printf("{%f, %f}, \n", tmp[0], tmp[1]);
    }
    
    return 0;
}
