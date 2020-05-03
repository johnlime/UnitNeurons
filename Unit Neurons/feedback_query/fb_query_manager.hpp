//
//  query_manager.hpp
//  Unit Neurons
//
//  Created by John Lime on 2020/04/06.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#ifndef query_manager_hpp
#define query_manager_hpp

#include "unit_neuron.hpp"
#include <stdio.h>
#include <vector>
#include <future>

struct FeedbackQuery
{
    FloatUnitNeuron* neuron;    // reference to neuron
    float fb_input [2];         // 2-element array of float
};

class FeedbackQueryManager{
private:
    FeedbackQuery* query_list;  // dynamic array of queries
    int num_query;
    std::vector<std::future<void>> fb_futures;
    
public:
    FeedbackQueryManager();
    void add_query(FeedbackQuery _new);
    void execute_all();
    void print_current_queries();
};

#endif /* query_manager_hpp */
