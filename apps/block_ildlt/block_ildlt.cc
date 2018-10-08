/**
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <iostream>
#include <apps/block_ildlt/block_ildlt_invariants.h>


/* ************************************************************************** */

int
main(
    int argc,
    char * argv[])
{
    if(argc != 10)
    {
        printf("Usage: culip-block-ildlt [path to matrix] "\
            "[path to permutation] [path to blocking] [path to pivots] " \
            "[precision] " \
            "[pivot method] [level of fill] [fill_factor] [threshold]\n" \
            "\n" \
            "where\n" \
            "\n" \
            "[path to matrix] - path to matrix mtx file\n" \
            "[path to permutation] - path to a permutation/reordering, saved " \
            "as 0-based mtx vector\n" \
            "[path to blocking] - path to blocking, saved a list of first " \
            "rows in a block and number of rows as last element stored as " \
            "0-based mtx\n" \
            "[path to pivots] - path to a row-wise 0/1 mtx vector, " \
            "1 means the row is a pivot row (2x2 -> 1 in row i, 0 in i+1)\n" \
            "[precision] - 0 for float/single, 1 for double\n" \
            "[pivot method] - 0 for static/from file, 1 for Bunch-Kaufmann, "\
            "2 for Rook\n" \
            "[level of fill] - nonnegative integral level of block-fill\n" \
            "[fill factor] - the allowed fine-fill in inside blocks, >= 1.0\n" \
            "[threshold] - relative threshold for element dropping\n");
        std::exit(EXIT_FAILURE);
    }

    /* read parameters */
    const char * matrix_path = argv[1];
    const char * permutation_path = argv[2];
    const char * blocking_path = argv[3];
    const char * ispivot_path = argv[4];
    const mat_int_t real_type = std::atoi(argv[5]);
    const mat_int_t pivot_method = std::atoi(argv[6]);
    const mat_int_t fill_level = std::atoi(argv[7]);
    const double fill_factor = std::max(1.0, std::atof(argv[8]));
    const double threshold = std::atof(argv[9]);

    const char * str_pivot_method;
    if(pivot_method == 0)
    {
        str_pivot_method = "Static";
    }
    else if(pivot_method == 1)
    {
        str_pivot_method = "Bunch-Kaufmann";
    }
    else if (pivot_method == 2)
    {
        str_pivot_method = "Rook";
    }
    else
    {
        printf("Unknown pivot method...\n");
        std::exit(EXIT_FAILURE);
    }

    const char * str_real_type;
    if(real_type == 0)
    {
        str_real_type = "single";
    }
    else if(real_type == 1)
    {
        str_real_type = "double";
    }
    else
    {
        printf("Unknown real type...\n");
        std::exit(EXIT_FAILURE);
    }

    printf("Input parameters:\n");
    printf("\tPrecision / real type: %s\n", str_real_type);
    printf("\tPivoting method: %s\n", str_pivot_method);
    printf("\tLevel of fill: %d\n", fill_level);
    printf("\tFill factor: %g\n", fill_factor);
    printf("\tThreshold: %g\n", threshold);

    std::string path_matrix(matrix_path);
    std::string path_blocking(blocking_path);
    std::string path_permutation(permutation_path);
    std::string path_ispivot(ispivot_path);

    printf("\n");
    if(real_type == 0)
    {
        if(pivot_method == 0)
            bildlt_fun<float, false, false>(path_matrix,
                path_blocking, path_permutation, path_ispivot,
                fill_level, fill_factor, threshold);
        if(pivot_method == 1)
            bildlt_fun<float, true, false>(path_matrix,
                path_blocking, path_permutation, path_ispivot,
                fill_level, fill_factor, threshold);
        if(pivot_method == 2)
            bildlt_fun<float, false, true>(path_matrix,
                path_blocking, path_permutation, path_ispivot,
                fill_level, fill_factor, threshold);
    }
    else
    {
        if(pivot_method == 0)
            bildlt_fun<double, false, false>(path_matrix,
                path_blocking, path_permutation, path_ispivot,
                fill_level, fill_factor, threshold);
        if(pivot_method == 1)
            bildlt_fun<double, true, false>(path_matrix,
                path_blocking, path_permutation, path_ispivot,
                fill_level, fill_factor, threshold);
        if(pivot_method == 2)
            bildlt_fun<double, false, true>(path_matrix,
                path_blocking, path_permutation, path_ispivot,
                fill_level, fill_factor, threshold);
    }

    /* clean up the mem pool */
    GlobalMemPool().cleanup();

    return EXIT_SUCCESS;
}

