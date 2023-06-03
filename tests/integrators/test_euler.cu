#include <cassert>
#include <vector>
#include <stdio.h>
#include <cmath>

#include <../../src/integrators/euler.cuh>

template <typename T>
void test_dqdd2dxd() {
    // Define the constants
    const int NUM_POS = 1;
    const int SIZE = 3;

    // Create input and output arrays
    T dqdd[SIZE];
    T result;

    dqdd[0] = -9.81;
    dqdd[1] = 0.0;
    dqdd[2] = 1.0;
    
    /* Expected result matrix is
         [ 0.  ,  1.  ,  0.  ],
         [-9.81,  0.  ,  1.  ]
    */
    std::vector<T> output_matrix = {
        0.0, -9.81,
        1.0, 0.0,
        0.0, 1.0
    };

    int test_AB_rows = 2;
    int test_AB_cols = 3;

    // validate each entry is computed correctly
    for (int i = 0; i < test_AB_rows; i++) {
        for (int j = 0; j < test_AB_cols; j++) {
            result = dqdd2dxd<T>(dqdd, i, j, NUM_POS); 
            assert(result == output_matrix[i + test_AB_rows * j]);
        }
    }
    
    // All test cases passed
    printf("Test dqdd2dxd passed.\n");
}

template <typename T>
void test_integrator() {
    T tolerance = 1e-4;  // Specify the tolerance value

    // Initialize the input values
    T dt = 0.1;
    std::vector<T> s_x = {0.5425, 6.2479};
    std::vector<T> s_qdd = {-8.3399};
    int num_pos = 1;

    // Initialize the expected output
    std::vector<T> expected_s_next_state = {1.1672, 5.4139};

    // Declare variables to hold the computed output
    std::vector<T> actual_s_next_state(s_x.size());
    
    // Call the function to be tested
    _integrator(actual_s_next_state.data(), s_x.data(), s_qdd.data(), dt, num_pos);

    // Verify the output matches the expected result using assert()
    for (size_t i = 0; i < expected_s_next_state.size(); i++) {
        // Compare the absolute difference with the tolerance
        assert(std::abs(actual_s_next_state[i] - expected_s_next_state[i]) <= tolerance);
    }

    printf("Test _integrator passed.\n");
}

template <typename T>
void test_integrator_gradient() {
    T tolerance = 1e-4;  // Specify the tolerance value

    std::vector<T> s_dqdd = {-8.4017, 0.0, 1.0};
    T dt = 0.1;
    int dim_AB_r = 2;
    int dim_AB_c = 3;
    int num_pos = 1;

    // Initialize the expected output
    std::vector<T> expected_ABk = {
        1.0, 0.1, 0.0,
        -0.8402, 1.0, 0.1
    };

    // Declare variables to hold the computed output
    std::vector<T> actual_ABk(dim_AB_r * dim_AB_c);

    // Call the function to be tested
    _integratorGradient(actual_ABk.data(), s_dqdd.data(), dt, dim_AB_r, dim_AB_c, num_pos);

    // Verify the output matches the expected result using assert() with tolerance
    for (size_t i = 0; i < dim_AB_r; i++) {
        for (size_t j = 0; j < dim_AB_c; j++) {
            assert(std::abs(actual_ABk[j * dim_AB_r + i] - expected_ABk[i * dim_AB_c + j]) <= tolerance);
        }
    }

    // Print a message if the test passes
    printf("Test integratorGradient passed.\n");
}

int main() {
    // test dqdd2dxd function with float
    test_dqdd2dxd<float>();
    // test dqdd2dxd function with double
    test_dqdd2dxd<double>();
    // test integrator function with float
    test_integrator<float>();
    // test integrator function with double
    test_integrator<double>();
    // test integratorGradient function with float
    test_integrator_gradient<float>();
    // test integratorGradient function with double
    test_integrator_gradient<double>();
    return 0;
}