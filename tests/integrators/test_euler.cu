#include <cassert>
#include <vector>
#include <stdio.h>

#include <../../src/integrators/euler.cuh>

template <typename T>
void test_dqdd2dxd() {
    // Define the constants
    const int NUM_POS = 1;
    const int SIZE = 3;

    // Create input and output arrays
    float dqdd[SIZE];
    float result;

    dqdd[0] = -9.81;
    dqdd[1] = 0.0;
    dqdd[2] = 1.0;
    
    /* Expected result matrix is
         [ 0.  ,  1.  ,  0.  ],
         [-9.81,  0.  ,  1.  ]
    */
    std::vector<float> output_matrix = {
        0.0, -9.81,
        1.0, 0.0,
        0.0, 1.0
    };

    int test_AB_rows = 2;
    int test_AB_cols = 3;

    // validate each entry is computed correctly
    for (int i = 0; i < test_AB_rows; i++) {
        for (int j = 0; j < test_AB_cols; j++) {
            result = dqdd2dxd<float>(dqdd, i, j, NUM_POS); 
            assert(result == output_matrix[i + test_AB_rows * j]);
        }
    }
    
    // All test cases passed
    printf("Test dqdd2dxd passed.\n");
}

int main() {
    test_dqdd2dxd<float>();
    return 0;
}