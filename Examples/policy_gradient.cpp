//
//  policy_gradient.cpp
//  Policy Gradient
//
//  Created by John Lime on 2020/04/10.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#include <stdio.h>
#include <math.h>
#include "input_output.hpp"
#include "fb_query_manager.hpp"
#include "gradient_descent.hpp"

#define EPOCHS pow(10, 4)
#define MAX_RANGE 100.0f

int main(int argc, const char * argv[]) {
    // define feedback query
    FeedbackQueryManager* query_manager = new FeedbackQueryManager();
    
    /* define policy */
    int policy_layers [3];
    policy_layers[0] = 32;
    policy_layers[1] = 32;
    policy_layers[2] = 4;
    
    FloatInputNeuron* x_input = new FloatInputNeuron();
    FloatInputNeuron* y_input = new FloatInputNeuron();
    FloatInputNeuron* state_input [2] = {x_input, y_input};
    
    int policy_num_neurons = 0;
    for (int i = 0; i < 3; i++)
    {
        policy_num_neurons += policy_layers[i];
    }
    FloatFeedForwardNeuron* all_policy_neurons [policy_num_neurons];
    int counter = 0;
    
    FloatFeedForwardNeuron* policy_layer_1 [policy_layers[0]];
    for (int i = 0; i < policy_layers[0]; i++)
    {
        policy_layer_1[i] = new FloatFeedForwardNeuron((FloatUnitNeuron**) state_input, 2, query_manager, "tanh");
        all_policy_neurons[counter] = policy_layer_1[i];
        counter ++;
    }
    
    FloatFeedForwardNeuron* policy_layer_2 [policy_layers[1]];
    for (int i = 0; i < policy_layers[1]; i++)
    {
        policy_layer_2[i] = new FloatFeedForwardNeuron((FloatUnitNeuron**) policy_layer_1, policy_layers[0], query_manager,"tanh");
        all_policy_neurons[counter] = policy_layer_2[i];
        counter ++;
    }
    
    FloatFeedForwardNeuron* policy_layer_3 [policy_layers[2]];
    for (int i = 0; i < policy_layers[2]; i++)
    {
        policy_layer_3[i] = new FloatFeedForwardNeuron((FloatUnitNeuron**) policy_layer_2, policy_layers[1], query_manager,"identity");
        all_policy_neurons[counter] = policy_layer_3[i];
        counter ++;
    }
    
    FloatGradientDescent policy_operator = FloatGradientDescent(all_policy_neurons, policy_num_neurons, policy_layers, 3);
    
    /* define state value function */
    int v_layers [3];
    v_layers[0] = 32;
    v_layers[1] = 32;
    v_layers[2] = 1;
    
    int v_num_neurons = 0;
    for (int i = 0; i < 6; i++)
    {
        v_num_neurons += v_layers[i];
    }
    FloatFeedForwardNeuron* all_v_neurons [v_num_neurons];
    counter = 0;
    
    FloatFeedForwardNeuron* v_layer_1 [v_layers[0]];
    for (int i = 0; i < v_layers[0]; i++)
    {
        v_layer_1[i] = new FloatFeedForwardNeuron((FloatUnitNeuron**) state_input, 2, query_manager, "tanh");
        all_v_neurons[counter] = v_layer_1[i];
        counter ++;
    }
    
    FloatFeedForwardNeuron* v_layer_2 [v_layers[1]];
    for (int i = 0; i < v_layers[1]; i++)
    {
        v_layer_2[i] = new FloatFeedForwardNeuron((FloatUnitNeuron**) v_layer_1, v_layers[0], query_manager,"tanh");
        all_v_neurons[counter] = v_layer_2[i];
        counter ++;
    }
    
    FloatFeedForwardNeuron* v_layer_3 [v_layers[2]];
    for (int i = 0; i < v_layers[2]; i++)
    {
        v_layer_3[i] = new FloatFeedForwardNeuron((FloatUnitNeuron**) v_layer_2, v_layers[1], query_manager,"identity");
        all_v_neurons[counter] = v_layer_3[i];
        counter ++;
    }
    
    FloatGradientDescent v_operator = FloatGradientDescent(all_v_neurons, v_num_neurons, v_layers, 3);
    
    printf("TRAIN\n");
    // trajectory storage
    float obs [1000][2];
    float next_obs [1000][2];
    int action [1000];
    float log_pi [1000];
    float advantage [1000];
    
    float tmp_obs [2];
    float* tmp_action = (float*) malloc(policy_layers[2] * sizeof(float));
    float max_reward = 1000;
    for (int iter = 0; iter < EPOCHS; iter++){
        // initial location
        tmp_obs[0] = ((float)rand() / RAND_MAX) * MAX_RANGE - MAX_RANGE / 2;
        tmp_obs[1] = ((float)rand() / RAND_MAX) * MAX_RANGE - MAX_RANGE / 2;
        state_input[0]->assign_value(tmp_obs[0]);
        state_input[1]->assign_value(tmp_obs[1]);
        
        // trajectory sampling
        for (int t = 0; t < 1000; t++)
        {
            // tmp_obs always contains the current state
            for (int i = 0; i < 2; i++)
            {
                obs[t][i] = tmp_obs[i];
            }
            
            for (int i = 0; i < policy_num_neurons; i++)
            {
                all_policy_neurons[i]->feedforward();
            }
            
            // get action and its softmax probs
            for (int i = 0; i < policy_layers[2]; i++)
            {
                tmp_action[i] = policy_layer_3[i]->state;
            }
            tmp_action = softmax(tmp_action, policy_layers[2]);
            
            // get max of output
            int max_index = 0;
            int max_prob = 0;
            for (int i = 0; i < policy_layers[2]; i++)
            {
                if (max_prob < tmp_action[i])
                {
                    max_index = i;
                    max_prob = tmp_action[i];
                }
            }
            action[t] = max_index;
            log_pi[t] = log(max_prob);
            
            /*
            Generalized Advantage Estimator (TD residual)
            More detailed explanation at https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/
            */
            advantage[t] = max_reward - sqrt(tmp_obs[0] * tmp_obs[0] + tmp_obs[1] * tmp_obs[1]); // GAE reward
            
            // get next obs (environment)
            if (max_index == 0)
            {
                next_obs[t][0] = tmp_obs[0] + 1;
            }
            else if (max_index == 1)
            {
                next_obs[t][1] = tmp_obs[1] + 1;
            }
            else if (max_index == 2)
            {
                next_obs[t][0] = tmp_obs[0] - 1;
            }
            else // max_index == 3
            {
                next_obs[t][1] = tmp_obs[1] - 1;
            }
            
            // state value for current obs
            for (int i = 0; i < v_num_neurons; i++)
            {
                all_v_neurons[i]->feedforward();
            }
            advantage[t] += v_layer_3[0]->state; // GAE current obs
            
            // get next obs
            state_input[0]->assign_value(next_obs[t][0]);
            state_input[1]->assign_value(next_obs[t][0]);
            
            // state value for next obs
            for (int i = 0; i < v_num_neurons; i++)
            {
                all_v_neurons[i]->feedforward();
            }
            advantage[t] -= v_layer_3[0]->state; // GAE next obs
            
            // update obs
            tmp_obs[0] = next_obs[t][0];
            tmp_obs[1] = next_obs[t][1];
        }
        
        // q function feedback
        global_operator.calculate_l1_loss(sin(tmp));
        global_operator.execute();
        query_manager->execute_all();
        
        // ppo loss function
        
        // policy feedback
        global_operator.calculate_l1_loss(sin(tmp));
        global_operator.execute();
        query_manager->execute_all();
    }
    
    // Output trained network
    
    return 0;
}