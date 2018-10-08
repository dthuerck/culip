/**
 *  Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 *  This software may be modified and distributed under the terms
 *  of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/algorithms/permute_btf.h>

NS_CULIP_BEGIN
NS_ALGORITHMS_BEGIN

BTF::
BTF()
: m_c_pqueue(),
  m_r_pqueue()
{

}

/* ************************************************************************** */

BTF::
~BTF()
{

}

/* ************************************************************************** */

mat_int_t
BTF::
permute(
    const mat_int_t m,
    const mat_int_t n,
    const mat_int_t * sp_A_csc_col,
    const mat_int_t * sp_A_csc_row,
    const mat_int_t * sp_A_csr_row,
    const mat_int_t * sp_A_csr_col,
    mat_int_t * row_order,
    mat_int_t * col_order)
{
    /* mark (in)active elements for rows and columns */
    std::vector<unsigned char> row_active(m, 1);
    std::vector<unsigned char> col_active(n, 1);

    /**
     * Hypergraph structure: columns of A contain adjacency lists of
     * the column-nodes, columns of A' contain adjacency lists of
     * the row-nodes (a.k.a. rows of A).
     */

    /* insert col-nodes with nnz per col into the queue */
    for(mat_int_t i = 0; i < n; ++i)
    {
        const mat_int_t col_rank = sp_A_csc_col[i + 1] - sp_A_csc_col[i];

        /**
         * if(col_rank == 0)
         *   std::cerr << "Empty col detected." << std::endl;
         */

        m_c_pqueue.push(i, col_rank);
    }

    /* insert row-nodes with nnz per row into queue */
    for(mat_int_t i = 0; i < m; ++i)
    {
        const mat_int_t row_rank = sp_A_csr_row[i + 1] - sp_A_csr_row[i];

        /**
         * if(row_rank == 0)
         *   std::cerr << "Empty row detected." << std::endl;
         */

        m_r_pqueue.push(i, row_rank);
    }

    /* begin Greedy procedure: iteratively pick one column */
    std::vector<mat_int_t> r_rest;
    std::vector<mat_int_t> c_rest;

    mat_int_t cur_degree, cur_col, cur_row_degree;
    mat_int_t it = 0;

    bool update_row;
    bool update_col;

    mat_int_t cur_row = (mat_int_t) -1;
    while(it < std::min(m, n) && !m_c_pqueue.empty() && !m_r_pqueue.empty())
    {
        /* reset update markers */
        update_row = false;
        update_col = false;

        /* get node with smallest key */
        m_c_pqueue.top(cur_col, cur_degree);

        /* if cur_degree (node degree) is not 1, matrix is not reducible */
        if(cur_degree == 1)
        {
            /* remove node from queue */
            m_c_pqueue.pop();

            /* add col node to output vector (+1 for 1-based indexing) */
            col_order[it] = cur_col + 1;
            col_active[cur_col] = 0;

            /* find only remaining row the col is connected to */
            for(mat_int_t j = sp_A_csc_col[cur_col];
                j < sp_A_csc_col[cur_col + 1]; ++j)
            {
                const mat_int_t row = sp_A_csc_row[j];

                if(row_active[row])
                {
                    cur_row = row;
                    break;
                }
            }

            /**
             *
             * if(cur_row == (T) -1)
             * {
             *    std::cerr << "Found empty row." << std::endl;
             * }
             */

            /* add row to output vector (+1 for 1-based indexing) */
            row_order[it] = cur_row + 1;
            row_active[cur_row] = 0;

            /* remove row from heap */
            m_r_pqueue.remove(cur_row);

            /* update both row and column heaps */
            update_row = true;
            update_col = true;

            /* step to next position in output */
            ++it;
        }
        else if(cur_degree == 0)
        {
            /* empty column now: move to column rest */
            c_rest.push_back(cur_col);

            /* no need to update rows, just remove column */
            col_active[cur_col] = 0;
            m_c_pqueue.pop();

            /* no updates necessary */
        }
        else
        {
            /* warn user once that there is no PTF */
            if(r_rest.empty())
            {
                std::cerr << "Note: Matrix A has no pure triangular form." <<
                    std::endl;
            }

            /* select a row for deletion */
            m_r_pqueue.top(cur_row, cur_row_degree);
            m_r_pqueue.pop();

            row_active[cur_row] = 0;
            r_rest.push_back(cur_row);

            /* only update column heap */
            update_row = false;
            update_col = true;
        }

        /* decrease degree of incident columns for deleted row */
        for(mat_int_t j = sp_A_csr_row[cur_row]; update_col && j <
            sp_A_csr_row[cur_row + 1]; ++j)
        {
            const mat_int_t col = sp_A_csr_col[j];

            if(col_active[col])
            {
                /* column still active -> decrease its degree in queue */
                m_c_pqueue.update(col, m_c_pqueue.value(col) - 1);
            }
        }

        /* decrease degree of incident rows for deleted column */
        for(mat_int_t j = sp_A_csc_col[cur_col]; update_row && j <
            sp_A_csc_col[cur_col + 1]; ++j)
        {
            const mat_int_t row = sp_A_csc_row[j];

            if(row_active[row])
            {
                /* row still active -> decrease its degree in queue */
                m_r_pqueue.update(row, m_r_pqueue.value(row) - 1);
            }
        }
    }

    /* save n_u */
    const mat_int_t n_u = it;
    std::cout << "Rank bound: " << n_u << std::endl;

    /* add remaining columns in any order */
    mat_int_t c_it = it;
    for(mat_int_t i = 0; i < n; ++i)
        if(col_active[i])
            col_order[c_it++] = i + 1;

    /* add rest cols in any order */
    for(const mat_int_t& c : c_rest)
        col_order[c_it++] = c + 1;

    /* add remaining (unprocessed) rows in any order */
    mat_int_t r_it = it;
    for(mat_int_t i = 0; i < m; ++i)
        if(row_active[i])
            row_order[r_it++] = i + 1;

    /* add rest rows in any order */
    for(const mat_int_t& r : r_rest)
        row_order[r_it++] = r + 1;

    /* return lower bound on rank */
    return n_u;
}

NS_ALGORITHMS_END
NS_CULIP_END
