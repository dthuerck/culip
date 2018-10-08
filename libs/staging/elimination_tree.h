/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_STAGING_ELIMINATION_TREE_H_
#define __CULIP_STAGING_ELIMINATION_TREE_H_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>

#include <libs/staging/triangular.h>
#include <libs/staging/flexible_triangular.h>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

/* forward declaration */
struct EliminationTreeNode;
using EliminationTreeNode_ptr = std::unique_ptr<EliminationTreeNode>;

struct EliminationTreeNode
{
    mat_int_t ix;

    /* must be sorted! */
    std::vector<mat_int_t> rows;

    mat_int_t parent;
    mat_int_t ancestor;

    /* functions */
    EliminationTreeNode_ptr copy() const;
};

/* ************************************************************************** */

template<typename T>
class EliminationTree
{
public:
    EliminationTree(const mat_int_t m);
    EliminationTree(const mat_int_t m, const mat_int_t num_pivots,
        const mat_int_t * piv_starts);
    ~EliminationTree();

    /* one-shot pattern, no pivoting */
    Triangular_ptr<T> extract_pattern(const csr_matrix_t<T> * A);

    /* row-by-row pattern, enabling pivoting */
    void init_pivot_pattern(const csr_matrix_t<T> * A);
    void pivot_1x1(const mat_int_t cur_row, const mat_int_t piv_a);
    void pivot_2x2(const mat_int_t cur_row, const mat_int_t piv_a,
        const mat_int_t piv_b);
    mat_int_t row_pattern(const mat_int_t row, mat_int_t * buf_ix);

    void postorder(mat_int_t * permutation);

    /* export data */
    void export_tree(std::vector<EliminationTreeNode_ptr>& buf);

    /* for debug */
    void print() const;

protected:
    void init_nodes();
    void update_tree(const mat_int_t col, const mat_int_t piv_node);
    void build(const FlexibleTriangular<T> * L);
    void rebuild(const mat_int_t cur_row);

protected:
    mat_int_t m_m;

    /* pivoted input data */
    FlexibleTriangular_ptr<T> m_pL;

    /* elimination tree */
    std::vector<EliminationTreeNode_ptr> m_nodes;
    std::vector<mat_int_t> m_row_in_node;

    /* list of 1x1 / 2x2 pivots */
    mat_int_t m_num_pivots;
    std::vector<mat_int_t> m_is_piv;

    /* for postordering */
    std::vector<mat_int_t> m_children_offsets;
    std::vector<mat_int_t> m_children;
};

template<typename T>
using EliminationTree_ptr = std::unique_ptr<EliminationTree<T>>;


NS_STAGING_END
NS_CULIP_END

#endif /* __CULIP_STAGING_ELIMINATION_TREE_H_ */