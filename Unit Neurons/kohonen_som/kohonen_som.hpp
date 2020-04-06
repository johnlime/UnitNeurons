//
//  kohonen_som.hpp
//  Unit Neurons
//
//  Created by John Lime on 2020/02/12.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#ifndef kohonen_som_hpp
#define kohonen_som_hpp

#include <stdio.h>
#include "unit_neuron.hpp"
#include "fb_query_manager.hpp"

class FloatMappingNeuron: public FloatUnitNeuron{
protected:
    float lr = 0.3f;                    // learning rate (hyperparameter)
    bool counter_first = true;          // Is this the first time that a signal from neighbor is received?
    float counter = 0;                  // track current neighbor count
    FloatMappingNeuron** neighbors;     // array of pointers to neighboring mapping neurons
    int num_neighbors;
    FeedbackQueryManager query_manager;
   
public:
    FloatMappingNeuron(FloatUnitNeuron** _prevs, int _num_prev, FeedbackQueryManager _query_manager, int _max);
    FloatMappingNeuron(FloatUnitNeuron** _prevs, int _num_prev, FeedbackQueryManager _query_manager);
    // assign array of input neurons' pointers during instantiation
    void init(FloatUnitNeuron** _prevs, int _num_prev, FeedbackQueryManager _query_manager, int _max);
    // assign array of neighboring neurons' pointers after instantiation
    void assign_neighbors(FloatMappingNeuron** _neighbors, int _num_neighbors);
    void feedforward();
    void feedback(float* fb_input);
    float* see_memory();
};

// checks winning neuron
class FloatKohonenSOM: public FloatGlobalOperator{
private:
    FloatMappingNeuron** maps;          // array of mapping neurons' pointers
    int num_maps;
    int neighbor_range;        // range of neighboring neurons
    
public:
    // assign array of mapping neurons' pointers and range of neighboring neurons during instantiation
    FloatKohonenSOM(FloatMappingNeuron** _maps, int _num_maps, int _neighbor_range);
    void execute();
};

#endif /* kohonen_som_hpp */
