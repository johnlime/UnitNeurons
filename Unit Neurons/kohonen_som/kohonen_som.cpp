//
//  kohonen_som1.cpp
//  Unit Neurons
//
//  Created by John Lime on 2020/03/25.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#include "kohonen_som.hpp"

FloatKohonenSOM:: FloatKohonenSOM(FloatMappingNeuron** _maps, int _num_maps, int _neighbor_range){
    maps = _maps;
    num_maps = _num_maps;
    neighbor_range = _neighbor_range;
}

void FloatKohonenSOM:: execute()
{
    FloatMappingNeuron* winner = maps[0];
    float shortest = maps[0]->state;
    int tmp = 0;
    for (int i = 1; i < num_maps; i++){
        if (shortest > maps[i]->state){
            winner = maps[i];
            shortest = maps[i]->state;
            tmp = i;
        }
    }
    float fb [2] = {float(neighbor_range), float(neighbor_range)};
    float ff [winner->num_prev];
    for (int i = 0; i < winner->num_prev; i++)
    {
        ff[i] = winner->previous[i]->state;
    }
    winner->feedback(ff, fb);
}
