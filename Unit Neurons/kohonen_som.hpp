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

class FloatMappingNeuron: public FloatUnitNeuron{
protected:
    float lr = 0.5f;                    // learning rate (hyperparameter)
    bool counter_first = true;          // Is this the first time that a signal from neighbor is received?
    float counter = 0;                  // track current neighbor count
    FloatMappingNeuron** neighbors;     // array of pointers to neighboring mapping neurons
   
public:
    FloatMappingNeuron(FloatUnitNeuron** _prevs);               // assign array of input neurons' pointers during instantiation
    void assign_neighbors(FloatMappingNeuron** _neighbors);     // assign array of neighboring neurons' pointers after instantiation
    void feedforward();
    void feedback(float* ff_input, float* fb_input);
};

// checks winning neuron
class FloatKohonenSOM: public FloatGlobalOperator{
private:
    FloatMappingNeuron** maps;          // array of mapping neurons' pointers
    unsigned int neighbor_range;        // range of neighboring neurons
    
public:
    // assign array of mapping neurons' pointers and range of neighboring neurons during instantiation
    FloatKohonenSOM(FloatMappingNeuron** _maps, unsigned int _neighbor_range);
    void execute();
};

#endif /* kohonen_som_hpp */
