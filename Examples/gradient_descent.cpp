//
//  gradient_descent.cpp
//  Gradient Descent
//
//  Created by John Lime on 2020/03/29.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#include <stdio.h>
#include <math.h>
#include "input_output.hpp"
#include "fb_query_manager.hpp"
#include "gradient_descent.hpp"

#define EPOCHS pow(10, 4)
#define MAX_RANGE 3.14f

int main(int argc, const char * argv[]) {
    // define input neurons
    FloatInputNeuron* input [1];
    input[0] = new FloatInputNeuron();
    
    // define feedback query
    FeedbackQueryManager* query_manager = new FeedbackQueryManager();
    
    // define feedforward neurons
    int layers [2];
    layers[0] = 8;
    layers[1] = 1;
    int num_neurons = 0;
    for (int i = 0; i < 2; i++)
    {
        num_neurons += layers[i];
    }
    FloatFeedForwardNeuron* all_neurons [num_neurons];
    int counter = 0;
    
    
    FloatFeedForwardNeuron* layer_1 [layers[0]];
    for (int i = 0; i < layers[0]; i++)
    {
        layer_1[i] = new FloatFeedForwardNeuron((FloatUnitNeuron**) input, 1, query_manager, "sigmoid");
        all_neurons[counter] = layer_1[i];
        counter ++;
    }
    
    FloatFeedForwardNeuron* layer_2 [layers[1]];
    for (int i = 0; i < layers[1]; i++)
    {
        layer_2[i] = new FloatFeedForwardNeuron((FloatUnitNeuron**) layer_1, layers[0], query_manager,"identity");
        all_neurons[counter] = layer_2[i];
        counter ++;
    }
    
    FloatGradientDescent global_operator = FloatGradientDescent(all_neurons, num_neurons, layers, 2);
    
    printf("TRAIN\n");
    for (int i = 0; i < EPOCHS; i++){
        // assign input
        float tmp = ((float)rand() / RAND_MAX) * MAX_RANGE;
        input[0]->assign_value(tmp);
        
        // feedforward
        for (int j = 0; j < num_neurons; j++){
            all_neurons[j]->feedforward();
        }
        
        // feedback
        global_operator.calculate_l1_loss(sin(tmp));
        global_operator.execute();
        query_manager->execute_all();
    }
    
    // Output trained network
    for (int i = 0; i < 100; i++){
        // assign input
        float tmp = ((float)rand() / RAND_MAX) * MAX_RANGE;
        input[0]->assign_value(tmp);

        // feedforward
        for (int j = 0; j < num_neurons; j++){
            all_neurons[j]->feedforward();
        }

//        printf("{Input: %f, Prediction: %f, Correct: %f}, \n", tmp, all_neurons[num_neurons - 1] -> state, sin(tmp));
        printf("{%f, %f}, \n", tmp, all_neurons[num_neurons - 1] -> state);
    }
    
    return 0;
}
