#include <gtest/gtest.h>
#include <iostream>
#include "cost.hpp"

TEST(CostTest, Constructor) {
    int w_size(3);
    int u_size(2);
    float w[w_size] = {0.2, 0.4, 2};
    float g[w_size] = {1, 2.3, 5};
    float lambda(0.5);
    float inv_s[u_size] = {2.0, 2.0};
    Cost c(w, w_size, g, w_size, lambda, inv_s, u_size);
}
/*
TEST(CostTest, Init) {

}

TEST(CostTest, OneStep) {

}

TEST(CostTest, OneFinal) {

}

TEST(CostTest, OneTraj) {

}

TEST(CostTest, MultiTraj) {

}
*/
