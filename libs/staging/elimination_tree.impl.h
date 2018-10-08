/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/staging/elimination_tree.h>

#include <algorithm>
#include <functional>
#include <numeric>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

EliminationTreeNode_ptr
EliminationTreeNode::
copy()
const
{
    EliminationTreeNode_ptr node =
        EliminationTreeNode_ptr(new EliminationTreeNode);

    node->ix = ix;
    node->rows = std::vector<mat_int_t>(rows.begin(), rows.end());
    node->parent = parent;
    node->ancestor = ancestor;

    return node;
}

/* ************************************************************************** */

template<typename T>
EliminationTree<T>::
EliminationTree(
    const mat_int_t m)
: m_m(m)
{
    m_nodes.resize(m);
    std::fill(m_nodes.begin(), m_nodes.end(), nullptr);

    m_row_in_node.resize(m);

    /* initialize nodes for 1x1 pivots */
    m_num_pivots = m;
    m_is_piv.resize(m);
    std::fill(m_is_piv.begin(), m_is_piv.end(), 1);

    init_nodes();
}

/* ************************************************************************** */

template<typename T>
EliminationTree<T>::
EliminationTree(
    const mat_int_t m,
    const mat_int_t num_pivots,
    const mat_int_t * piv_starts)
: m_m(m)
{
    m_nodes.resize(num_pivots);
    std::fill(m_nodes.begin(), m_nodes.end(), nullptr);

    m_row_in_node.resize(m);

    /* intialize nodes with correct rows */
    m_num_pivots = num_pivots;
    m_is_piv.resize(m);
    std::fill(m_is_piv.begin(), m_is_piv.end(), 0);
    for(mat_int_t i = 0; i < num_pivots; ++i)
        m_is_piv[piv_starts[i]] = 1;

    init_nodes();
}

/* ************************************************************************** */

template<typename T>
EliminationTree<T>::
~EliminationTree()
{

}

/* ************************************************************************** */

template<typename T>
Triangular_ptr<T>
EliminationTree<T>::
extract_pattern(
    const csr_matrix_t<T> * A)
{
    /* compute elimination tree */
    init_pivot_pattern(A);

    std::vector<mat_int_t> L_csr_row(m_m + 1);
    std::vector<mat_int_t> L_csr_col;

    L_csr_row[0] = 0;
    #pragma omp parallel for
    for(mat_int_t r = 0; r < m_m; ++r)
    {
        /* note: indices are already sorted! */
        std::vector<mat_int_t> buf(r + 1);
        mat_int_t row_len = row_pattern(r, buf.data());
        L_csr_row[r] = row_len;
    }

    /* compute offsets */
    mat_int_t hold = L_csr_row[0];
    L_csr_row[0] = 0;
    for(mat_int_t i = 1; i < m_m + 1; ++i)
    {
        const mat_int_t res = L_csr_row[i - 1] + hold;
        hold = L_csr_row[i];
        L_csr_row[i] = res;
    }

    const mat_int_t nnz = L_csr_row[m_m];
    L_csr_col.resize(nnz);

    # pragma omp parallel for
    for(mat_int_t r = 0; r < m_m; ++r)
    {
        /* note: indices are already sorted! */
        std::vector<mat_int_t> buf(r + 1);
        mat_int_t row_len = row_pattern(r, buf.data());
        std::copy(buf.data(), buf.data() + row_len,
            L_csr_col.data() + L_csr_row[r]);
    }

    /* un-permute all data into L */
    Triangular_ptr<T> L = Triangular_ptr<T>(new Triangular<T>(m_m, nnz));
    std::copy(L_csr_row.begin(), L_csr_row.end(), L->raw_row_ptr());
    std::copy(L_csr_col.begin(), L_csr_col.end(), L->raw_col_ptr());
    std::fill(L->raw_val_ptr(), L->raw_val_ptr() + nnz, 1);

    return L;
}

/* ************************************************************************** */

template<typename T>
void
EliminationTree<T>::
init_pivot_pattern(
    const csr_matrix_t<T> * A)
{
    /* extract A's data into a pivotable triangular matrix */
    m_pL = FlexibleTriangular_ptr<T>(new FlexibleTriangular<T>(A));

    build(m_pL.get());
}

/* ************************************************************************** */

template<typename T>
void
EliminationTree<T>::
pivot_1x1(
    const mat_int_t cur_row,
    const mat_int_t piv_a)
{
    /* exchange rows in L */
    m_pL->pivot(cur_row, piv_a);

    /* set cur_row as 1x1 pivot */
    m_is_piv[cur_row] = 1;
    if(cur_row < m_m)
        m_is_piv[cur_row + 1] = 1;

    rebuild(cur_row);
}

/* ************************************************************************** */

template<typename T>
void
EliminationTree<T>::
pivot_2x2(
    const mat_int_t cur_row,
    const mat_int_t piv_a,
    const mat_int_t piv_b)
{
    /* exchange rows in L */
    m_pL->pivot(cur_row, piv_a);
    m_pL->pivot(cur_row + 1, piv_b);

    /* set (cur_row, cur_row + 1) as 2x2 pivot */
    m_is_piv[cur_row] = 1;
    m_is_piv[cur_row + 1] = 0;

    /* we started with 1x1 pivots, so reduce the number of pivots */
    --m_num_pivots;

    rebuild(cur_row);
}

/* ************************************************************************** */

template<typename T>
mat_int_t
EliminationTree<T>::
row_pattern(
    const mat_int_t row,
    mat_int_t * buf_ix)
{
    auto contains =
        [](const EliminationTreeNode_ptr& c_node, const mat_int_t id)
        {
            return (std::find(c_node->rows.begin(), c_node->rows.end(), id)
                != c_node->rows.end());
        };

    std::vector<char> markers(m_num_pivots);

    /* start below here */
    mat_int_t * cols;
    T * vals;
    const mat_int_t row_size = m_pL->row(row, cols, vals);

    /* reset markers */
    const mat_int_t p = m_row_in_node[row];
    std::fill(markers.begin(), markers.end(), 0);

    /* always include diagonal element - set marker */
    mat_int_t nz = 1;
    markers[p] = 1;
    buf_ix[0] = row;
    for(mat_int_t j = 0; j < row_size; ++j)
    {
        const mat_int_t col = cols[j];

        /* only read lower sub-pivot diagonal part */
        if(col >= row)
            continue;

        /* add nodes from traversal */
        mat_int_t node = m_row_in_node[col];
        while(!contains(m_nodes[node], row) && !markers[node])
        {
            /* traverse (mark) node */
            markers[node] = 1;

            /**
            * Note: only add _all_ rows in k x k nodes if
            * they are neither start nor end node
            */
            for(const mat_int_t& n : m_nodes[node]->rows)
            {
                buf_ix[nz] = n;
                ++nz;
            }

            /* skip to parent */
            node = m_nodes[node]->parent;
        }
    }

    /* sort pattern */
    std::sort(buf_ix, buf_ix + nz);

    return nz;
}

/* ************************************************************************** */

template<typename T>
void
EliminationTree<T>::
postorder(
    mat_int_t * permutation)
{
    /* create a list of children */
    std::vector<mat_int_t> num_children(m_num_pivots, 0);
    std::vector<mat_int_t> roots;

    m_children_offsets.resize(m_num_pivots + 1);
    m_children.resize(m_num_pivots);

    for(mat_int_t i = 0; i < m_num_pivots; ++i)
    {
        const mat_int_t i_parent = m_nodes[i]->parent;
        if(i_parent != -1)
            ++num_children[i_parent];
        else
            roots.push_back(i);
    }
    std::sort(roots.begin(), roots.end());

    m_children_offsets[0] = 0;
    for(mat_int_t i = 1; i < m_num_pivots + 1; ++i)
        m_children_offsets[i] = m_children_offsets[i - 1] + num_children[i - 1];

    std::fill(num_children.begin(), num_children.end(), 0);
    for(mat_int_t i = 0; i < m_num_pivots; ++i)
    {
        const mat_int_t i_parent = m_nodes[i]->parent;
        if(i_parent != -1)
        {
            m_children[m_children_offsets[i_parent] + num_children[i_parent]] =
                i;
            ++num_children[i_parent];
        }
    }

    /* sort children */
    for(mat_int_t i = 0; i < m_num_pivots; ++i)
        std::sort(m_children.data() + m_children_offsets[i],
            m_children.data() + m_children_offsets[i + 1]);

    /* use DFS to create post ordering */
    std::function<mat_int_t(const mat_int_t, const mat_int_t)> dfstree =
    [&](const mat_int_t cur, const mat_int_t k) -> mat_int_t
    {
        mat_int_t new_k = k;

        for(mat_int_t i = m_children_offsets[cur];
            i < m_children_offsets[cur + 1]; ++i)
            new_k = dfstree(m_children[i], new_k);

        for(const mat_int_t row : m_nodes[cur]->rows)
            permutation[new_k++] = row;

        return new_k;
    };

    mat_int_t k = 0;
    for(const mat_int_t r : roots)
        k = dfstree(r, k);
}

/* ************************************************************************** */

template<typename T>
void
EliminationTree<T>::
export_tree(
    std::vector<EliminationTreeNode_ptr>& buf)
{
    buf.resize(m_nodes.size());
    for(mat_int_t i = 0; i < m_nodes.size(); ++i)
        buf[i] = m_nodes[i]->copy();
}

/* ************************************************************************** */

template<typename T>
void
EliminationTree<T>::
print()
const
{
    printf("Elimination Tree:\n");
    for(const EliminationTreeNode_ptr& node : m_nodes)
    {
        printf("- Node %d: rows ", node->ix);
        for(const mat_int_t n : node->rows)
            printf("%d ", n);
        printf("- parent %d, ancestor %d\n", node->parent, node->ancestor);
    }
}

/* ************************************************************************** */

template<typename T>
void
EliminationTree<T>::
init_nodes()
{
    mat_int_t row_ptr = 0;
    mat_int_t node_ptr = 0;

    while(row_ptr < m_m)
    {
        const mat_int_t piv_order = (row_ptr == m_m - 1) ?
            1 : (2 - m_is_piv[row_ptr + 1]);

        m_nodes[node_ptr] = EliminationTreeNode_ptr(new EliminationTreeNode);
        m_nodes[node_ptr]->ix = node_ptr;
        m_nodes[node_ptr]->parent = (mat_int_t) -1;
        m_nodes[node_ptr]->ancestor = (mat_int_t) -1;
        m_nodes[node_ptr]->rows.resize(piv_order);

        for(mat_int_t j = 0; j < piv_order; ++j)
        {
            m_nodes[node_ptr]->rows[j] = row_ptr + j;
            m_row_in_node[row_ptr + j] = node_ptr;
        }

        ++node_ptr;
        row_ptr += piv_order;
    }
}

/* ************************************************************************** */

template<typename T>
void
EliminationTree<T>::
update_tree(
    const mat_int_t col_node,
    const mat_int_t piv_node)
{
    EliminationTreeNode_ptr& node = m_nodes[col_node];

    if(node->parent == (mat_int_t) -1)
    {
        /* no parent - connect to pivot */
        node->parent = piv_node;

        /* always update ancestors */
        node->ancestor = piv_node;

        return;
    }
    else
    {
        /* has parent: continue traversal via ancestor */
        if(node->ancestor < piv_node)
        {
            update_tree(node->ancestor, piv_node);

            /* always update ancestors */
            m_nodes[node->parent]->ancestor = piv_node;
        }
    }
}

/* ************************************************************************** */

template<typename T>
void
EliminationTree<T>::
build(
    const FlexibleTriangular<T> * L)
{
    /* initialize ancestry and parents with empty nodes */
    const mat_int_t num_pivots = m_nodes.size();

    /* build up row elimination trees, updating ancestors */
    for(mat_int_t i = 0; i < num_pivots; ++i)
    {
        /* edit rows */
        for(const mat_int_t r : m_nodes[i]->rows)
        {
            mat_int_t * cols;
            T * vals;
            const mat_int_t row_size = L->row(r, cols, vals);

            /* 1x1 pivots resp. first row of 2x2 pivots */
            for(mat_int_t k = 0; k < row_size; ++k)
            {
                const mat_int_t col = cols[k];
                if(col < m_nodes[i]->rows[0])
                {
                    update_tree(m_row_in_node[col], i);
                }
            }
        }
    }
}

/* ************************************************************************** */

/* rebuilds the tree from cur_row onwards */
template<typename T>
void
EliminationTree<T>::
rebuild(
    const mat_int_t cur_row)
{
    /* erase all old nodes >= cur_row */
    m_nodes.erase(std::remove_if(
        m_nodes.begin(),
        m_nodes.end(),
        [&](const EliminationTreeNode_ptr& en)
        {
            bool larger_row = false;
            for(mat_int_t j : en->rows)
                larger_row |= (j >= cur_row);

            return larger_row;
        }),
        m_nodes.end());

    /* remove invalid ancestors from earlier nodes */
    const mat_int_t cur_row_old_node = m_row_in_node[cur_row];
    for(mat_int_t i = 0; i < m_nodes.size(); ++i)
    {
        if(m_nodes[i]->parent > cur_row_old_node)
            m_nodes[i]->parent = -1;
        if(m_nodes[i]->ancestor > cur_row_old_node)
            m_nodes[i]->ancestor = m_nodes[i]->parent;
    }

    /* rebuild nodes */
    mat_int_t row_ptr = cur_row;
    while(row_ptr < m_m)
    {
        const mat_int_t piv_order = (row_ptr == m_m - 1) ?
            1 : (2 - m_is_piv[row_ptr + 1]);

        /* create node */
        m_nodes.emplace_back(EliminationTreeNode_ptr(new
            EliminationTreeNode));
        m_nodes.back()->ix = m_nodes.size() - 1;
        m_nodes.back()->parent = -1;
        m_nodes.back()->ancestor = -1;
        m_nodes.back()->rows.resize(piv_order);

        for(mat_int_t j = 0; j < piv_order; ++j)
        {
            m_nodes.back()->rows[j] = row_ptr + j;
            m_row_in_node[row_ptr + j] = m_nodes.size() - 1;
        }

        row_ptr += piv_order;
    }

    /* use pivoted matrix to build up the rest of the tree */
    row_ptr = cur_row;
    while(row_ptr < m_m)
    {
        const mat_int_t piv_order = (row_ptr == m_m - 1) ?
            1 : (2 - m_is_piv[row_ptr + 1]);
        const mat_int_t in_node = m_row_in_node[row_ptr];

        for(mat_int_t r = row_ptr; r < row_ptr + piv_order; ++r)
        {
            mat_int_t * cols;
            T * vals;
            const mat_int_t row_size = m_pL->row(r, cols, vals);

            /* 1x1 pivots resp. first row of 2x2 pivots */
            for(mat_int_t k = 0; k < row_size; ++k)
            {
                const mat_int_t col = cols[k];
                if(col < m_nodes[in_node]->rows[0])
                {
                    update_tree(m_row_in_node[col], in_node);
                }
            }
        }

        row_ptr += piv_order;
    }
}

NS_STAGING_END
NS_CULIP_END