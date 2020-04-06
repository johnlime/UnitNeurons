//
//  kohonen_som.cpp
//  Unit Neurons
//
//  Created by John Lime on 2020/02/07.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#include <iostream>
#include "input_output.hpp"
#include "fb_query_manager.hpp"
#include "kohonen_som.hpp"
#include <math.h>
#define NODE_D 3
#define EPOCHS pow(10, 7)
#define MAX_RANGE 100

int main(int argc, const char * argv[]) {
    // define input neurons
    FloatInputNeuron* x_input = new FloatInputNeuron();
    FloatInputNeuron* y_input = new FloatInputNeuron();
    FloatInputNeuron* io_neuron [2] = {x_input, y_input};
    
    // define feedback query
    FeedbackQueryManager* query_manager = new FeedbackQueryManager();
    
    // define mapping neurons
    FloatMappingNeuron* maps [NODE_D * NODE_D];
    for (int i = 0; i < NODE_D * NODE_D; i++){
        FloatMappingNeuron* tmp = new FloatMappingNeuron((FloatUnitNeuron**) io_neuron, 2, query_manager,MAX_RANGE);
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
            
            FloatMappingNeuron** tmp = new FloatMappingNeuron* [2]
            {
                maps[i + a],
                maps[i + b]
            };
            maps[i]->assign_neighbors(tmp, 2);
//            printf("%d (%p): %d(%p; %p), %d(%p; %p)\n",
//                   i, maps[i],
//                   i+a, maps[i+a], tmp[0],
//                   i+b, maps[i+b], tmp[1]
//                   );
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
            
            FloatMappingNeuron** tmp = new FloatMappingNeuron* [3]
            {
                maps[i - b],
                maps[i + a],
                maps[i + b]
            };
            maps[i]->assign_neighbors(tmp, 3);
//            printf("%d (%p): %d(%p; %p), %d(%p; %p), %d(%p; %p)\n",
//                   i, maps[i],
//                   i-b, maps[i-b], tmp[0],
//                   i+a, maps[i+a], tmp[1],
//                   i+b, maps[i+b], tmp[2]
//                   );
        }
        
        // default
        else
        {
            a = 1;
            b = NODE_D;
            
            FloatMappingNeuron** tmp = new FloatMappingNeuron* [4]
            {
                maps[i - a],
                maps[i + a],
                maps[i + b],
                maps[i - b]
            };
            maps[i]->assign_neighbors(tmp, 4);
//            printf("%d (%p): %d(%p; %p), %d(%p; %p), %d(%p; %p), %d(%p; %p)\n",
//                   i, maps[i],
//                   i-a, maps[i-a], tmp[0],
//                   i+a, maps[i+a], tmp[1],
//                   i+b, maps[i+b], tmp[2],
//                   i-b, maps[i-b], tmp[3]);
        }
    }
    
    // define global operator
    FloatKohonenSOM global_operator = FloatKohonenSOM(maps, NODE_D * NODE_D, 3);
    
//    for (int i = 0; i < NODE_D * NODE_D; i++){
//        float* tmp = maps[i]->see_memory();
//        printf("{%f, %f}, \n", tmp[0], tmp[1]);
//    }
    /* loop through dataset */
    printf("TRAIN\n");
    for (int i = 0; i < EPOCHS; i++){
        // feedforward
        for(int j = 0; j < 2; j++){
            float tmp = ((float)rand() / RAND_MAX) * MAX_RANGE;
            io_neuron[j]->assign_value(tmp); //((float)rand() / RAND_MAX) * MAX_RANGE);
        }
        for (int j = 0; j < NODE_D * NODE_D; j++){
            maps[j]->feedforward();
        }
        // feedback
        global_operator.execute();
        query_manager->execute_all();
    }
    
    // output trained network
    for (int i = 0; i < NODE_D * NODE_D; i++){
        float* tmp = maps[i]->see_memory();
        printf("{%f, %f}, \n", tmp[0], tmp[1]);
    }
    
    return 0;
}
