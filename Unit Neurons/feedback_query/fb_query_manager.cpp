//
//  query_manager.cpp
//  Unit Neurons
//
//  Created by John Lime on 2020/04/06.
//  Copyright © 2020 Mioto Takahashi. All rights reserved.
//

#include "fb_query_manager.hpp"
#include <stdlib.h>
#include <string.h>

#define ASYNC 1

FeedbackQueryManager:: FeedbackQueryManager()
{
    query_list = (FeedbackQuery*) malloc(0 * sizeof(FeedbackQuery));
    num_query = 0;
}

void FeedbackQueryManager:: add_query(FeedbackQuery _new)
{
    query_list = (FeedbackQuery*) realloc(query_list, (num_query + 1) * sizeof(FeedbackQuery));
    query_list[num_query] = _new;
    num_query ++;
}

void FeedbackQueryManager:: execute_all()
{
    while (num_query > 0)
    {
#if ASYNC
        // lock write to own neuron
        // feedback function in own neuron
        // execute feedback
        // change number of queries
        // move queries upward
        
#else
        query_list[0].neuron->feedback(query_list[0].fb_input);                     // execute the top function
        num_query --;                                                               // reduce number of executed queries
        memmove(query_list, query_list + 1, num_query * sizeof(FeedbackQuery));     // move 1 element upward
#endif
    }
    
    free(query_list);
    query_list = (FeedbackQuery*) malloc(0 * sizeof(FeedbackQuery));
    num_query = 0;
}

void FeedbackQueryManager:: print_current_queries()
{
    for (int i = 0; i < num_query; i++)
    {
        printf("%p\n", query_list[i].neuron);
        printf("%f, %f\n", query_list[i].fb_input[0], query_list[i].fb_input[1]);
    }
}
