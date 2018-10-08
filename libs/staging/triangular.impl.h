/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/staging/triangular.h>

#include <algorithm>
#include <stack>
#include <numeric>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

template<typename T>
Triangular<T>::
Triangular(
    const mat_int_t m,
    const mat_int_t nnz,
    const bool lower)
: m_m(m),
  m_lower(lower),
  m_nnz(nnz),
  m_scheduled(false),
  m_csr_row(m + 1),
  m_csr_col(nnz),
  m_csr_val(nnz)
{
    /* initialize levels (unknown) */
    m_num_levels = 1;
    m_level_offsets.resize(2);
    m_level_offsets[0] = 0;
    m_level_offsets[1] = m;
    m_levels.resize(m);
    std::iota(m_levels.begin(), m_levels.end(), 0);
}

/* ************************************************************************** */

template<typename T>
Triangular<T>::
Triangular(
    const mat_int_t nnz,
    const mat_int_t * A_i,
    const mat_int_t * A_j,
    const T * A_k)
: m_nnz(nnz)
{
    ijk_compress(A_i, A_j, A_k);

    /* initialize levels (unknown) */
    m_num_levels = 1;
    m_level_offsets.resize(2);
    m_level_offsets[0] = 0;
    m_level_offsets[1] = m_m;
    m_levels.resize(m_m);
    std::iota(m_levels.begin(), m_levels.end(), 0);
}

/* ************************************************************************** */

template<typename T>
Triangular<T>::
Triangular(
    const csr_matrix_t<T> * A)
{
    /* compute new offsets */
    m_csr_row = std::vector<mat_int_t>(A->m + 1, 0);
    for(mat_int_t i = 0; i < A->m; ++i)
    {
        const mat_int_t i_len = A->csr_row[i + 1] - A->csr_row[i];

        mat_int_t diag_ptr = 0;
        for(mat_int_t j = 0; j < i_len; ++j)
        {
            if(A->csr_col[A->csr_row[i] + j] <= i)
            {
                diag_ptr = j + 1;
            }
        }

        m_csr_row[i] = diag_ptr;
    }

    mat_int_t hold = m_csr_row[0];
    m_csr_row[0] = 0;
    for(mat_int_t i = 1; i < A->m + 1; ++i)
    {
        const mat_int_t res = m_csr_row[i - 1] + hold;
        hold = m_csr_row[i];
        m_csr_row[i] = res;
    }

    const mat_int_t L_nnz = m_csr_row.back();
    m_csr_col.resize(L_nnz);
    m_csr_val.resize(L_nnz);

    m_lower = true;
    m_m = A->m;
    m_nnz = L_nnz;

    /* copy column and value arrys */
    for(mat_int_t i = 0; i < A->m; ++i)
    {
        const mat_int_t L_i_len = m_csr_row[i + 1] - m_csr_row[i];
        std::copy(A->csr_col + A->csr_row[i], A->csr_col + A->csr_row[i] +
            L_i_len, m_csr_col.data() + m_csr_row[i]);
        std::copy(A->csr_val + A->csr_row[i], A->csr_val + A->csr_row[i] +
            L_i_len, m_csr_val.data() + m_csr_row[i]);
    }

    /* initialize levels (unknown) */
    m_num_levels = 1;
    m_level_offsets.resize(2);
    m_level_offsets[0] = 0;
    m_level_offsets[1] = A->m;
    m_levels.resize(A->m);
    std::iota(m_levels.begin(), m_levels.end(), 0);
}

/* ************************************************************************** */

template<typename T>
Triangular<T>::
~Triangular()
{
}

/* ************************************************************************** */

template<typename T>
mat_int_t
Triangular<T>::
m()
const
{
    return m_m;
}

/* ************************************************************************** */

template<typename T>
mat_int_t
Triangular<T>::
nnz()
const
{
    return m_nnz;
}

/* ************************************************************************** */

template<typename T>
void
Triangular<T>::
resize(
    const mat_int_t new_nnz)
{
    m_nnz = new_nnz;

    m_csr_col.resize(new_nnz);
    m_csr_val.resize(new_nnz);
}

/* ************************************************************************** */

template<typename T>
mat_int_t
Triangular<T>::
row_length(
    const mat_int_t row)
const
{
    if(row >= m_m)
        return 0;

    return (m_csr_row[row + 1] - m_csr_row[row]);
}

/* ************************************************************************** */

template<typename T>
const mat_int_t *
Triangular<T>::
row_col(
    const mat_int_t row)
const
{
    return (const mat_int_t *) (m_csr_col.data() + m_csr_row[row]);
}

/* ************************************************************************** */

template<typename T>
const T *
Triangular<T>::
row_val(
    const mat_int_t row)
const
{
    return (const T *) (m_csr_val.data() + m_csr_row[row]);
}

/* ************************************************************************** */

template<typename T>
mat_int_t *
Triangular<T>::
row_col_rw(
    const mat_int_t row)
{
    return (m_csr_col.data() + m_csr_row[row]);
}

/* ************************************************************************** */

template<typename T>
T *
Triangular<T>::
row_val_rw(
    const mat_int_t row)
{
    return (m_csr_val.data() + m_csr_row[row]);
}

/* ************************************************************************** */

template<typename T>
mat_int_t *
Triangular<T>::
raw_row_ptr()
{
    return m_csr_row.data();
}

/* ************************************************************************** */

template<typename T>
mat_int_t *
Triangular<T>::
raw_col_ptr()
{
    return m_csr_col.data();
}

/* ************************************************************************** */

template<typename T>
T *
Triangular<T>::
raw_val_ptr()
{
    return m_csr_val.data();
}

/* ************************************************************************** */

template<typename T>
csr_matrix_ptr<T>
Triangular<T>::
to_csr()
const
{
    /* allocate new CSR */
    csr_matrix_ptr<T> csr = make_csr_matrix_ptr<T>(m_m, m_m, m_nnz, false);
    std::copy(m_csr_row.begin(), m_csr_row.end(), csr->csr_row);
    std::copy(m_csr_col.begin(), m_csr_col.end(), csr->csr_col);
    std::copy(m_csr_val.begin(), m_csr_val.end(), csr->csr_val);

    // /* only store pointers in CSR matrix */
    // csr_matrix_ptr<T> csr = make_csr_matrix_ptr<T>(m_m, m_m, m_nnz,
    //     m_csr_row.data(), m_csr_col.data(), m_csr_val.data(), false);

    return csr;
}

/* ************************************************************************** */

template<typename T>
mat_int_t
Triangular<T>::
sanalysis(
    const mat_int_t b_len,
    const mat_int_t * b_ix)
const
{
    return sub_sanalysis(m_m, b_len, b_ix);
}

/* ************************************************************************** */

template<typename T>
void
Triangular<T>::
sanalysis_import(
    const mat_int_t b_len,
    const mat_int_t * b_ix,
    const mat_int_t x_len,
    const mat_int_t * x_ix)
const
{
    sub_sanalysis_import(b_len, b_ix, x_len, x_ix);
}

/* ************************************************************************** */

template<typename T>
void
Triangular<T>::
sanalysis_export(
    mat_int_t * x_ix)
const
{
    sub_sanalysis_export(x_ix);
}

/* ************************************************************************** */

template<typename T>
void
Triangular<T>::
sfsolve(
    const T * b_val,
    T * x_val)
const
{
    sub_sfsolve(m_m, b_val, x_val);
}

/* ************************************************************************** */

template<typename T>
void
Triangular<T>::
dfsolve(
    const T * b_val,
    T * x_val)
const
{
    sub_dfsolve(m_m, b_val, x_val);
}

/* ************************************************************************** */

template<typename T>
void
Triangular<T>::
dbsolve(
    const T * b_val,
    T * x_val)
const
{
    /* initialize x = b */
    std::copy(b_val, b_val + m_m, x_val);

    for(mat_int_t i = m_m - 1; i >= 0; --i)
    {
        const mat_int_t l_size = row_length(i);
        const mat_int_t * l_col = row_col(i);
        const T * l_val = row_val(i);

        /* find diagonal entry and divide solution */
        if(l_col[l_size - 1] == i)
            x_val[i] /= l_val[l_size - 1];

        /* scatter updates to other values */
        for(mat_int_t j = 0; j < l_size; ++j)
            if(l_col[j] < i)
                x_val[l_col[j]] -= l_val[j] * x_val[i];
    }
}

/* ************************************************************************** */

template<typename T>
mat_int_t
Triangular<T>::
sub_sanalysis(
    const mat_int_t sub_m,
    const mat_int_t b_len,
    const mat_int_t * b_ix)
const
{
    /* copy b's data */
    m_b_len = b_len;
    m_b_ix.resize(b_len);
    std::copy(b_ix, b_ix + b_len, m_b_ix.begin());

    if(sub_m == 0)
        return 0;

    /* compute reach to determine x's nonzero pattern */
    std::vector<bool> nz_marker(sub_m, false);

    /* mark all nodes that are nz by b_i != 0 */
    for(mat_int_t b_i = 0; b_i < b_len; ++b_i)
    {
        const mat_int_t ix = b_ix[b_i];

        if(ix < sub_m)
            nz_marker[ix] = true;
    }

    /* for all other nodes, follow paths to one of the above */
    m_x_ix.clear();
    for(mat_int_t b_i = 0; b_i < sub_m; ++b_i)
    {
        const mat_int_t row_len = row_length(b_i);
        const mat_int_t * row_ix = row_col(b_i);

        for(mat_int_t i = 0; i < row_len && !nz_marker[b_i]; ++i)
        {
            nz_marker[b_i] = nz_marker[row_ix[i]] | nz_marker[b_i];
        }

        if(nz_marker[b_i])
        {
            m_x_ix.push_back(b_i);
        }
    }

    return m_x_ix.size();
}

/* ************************************************************************** */

template<typename T>
void
Triangular<T>::
sub_sanalysis_import(
    const mat_int_t b_len,
    const mat_int_t * b_ix,
    const mat_int_t x_len,
    const mat_int_t * x_ix)
const
{
    m_b_len = b_len;
    m_b_ix.resize(b_len);
    std::copy(b_ix, b_ix + b_len, m_b_ix.begin());

    m_x_ix.resize(x_len);
    std::copy(x_ix, x_ix + x_len, m_x_ix.begin());
}

/* ************************************************************************** */

template<typename T>
void
Triangular<T>::
sub_sanalysis_export(
    mat_int_t * x_ix)
const
{
    std::copy(m_x_ix.begin(), m_x_ix.end(), x_ix);
}

/* ************************************************************************** */

template<typename T>
void
Triangular<T>::
sub_sfsolve(
    const mat_int_t sub_m,
    const T * b_val,
    T * x_val)
const
{
    std::vector<T> full_x(sub_m, 0);

    /* scatter b */
    for(mat_int_t i = 0; i < m_b_len; ++i)
        if(m_b_ix[i] < sub_m)
            full_x[m_b_ix[i]] = b_val[i];

    for(mat_int_t i = 0; i < m_x_ix.size(); ++i)
    {
        const mat_int_t ix = m_x_ix[i];
        const mat_int_t row_len = row_length(ix);

        const mat_int_t * cols = row_col(ix);
        const T * vals = row_val(ix);

        /* subtract dot product from row and solution */
        for(mat_int_t j = 0; j < row_len; ++j)
            if(cols[j] != ix)
                full_x[ix] -= vals[j] * full_x[cols[j]];

        /* divide by diagonal element */
        if(cols[row_len - 1] == ix)
            full_x[ix] /= vals[row_len - 1];
    }

    /* compress solution vector */
    for(mat_int_t i = 0; i < m_x_ix.size(); ++i)
        x_val[i] = full_x[m_x_ix[i]];
}

/* ************************************************************************** */

template<typename T>
void
Triangular<T>::
sub_dfsolve(
    const mat_int_t sub_m,
    const T * b_val,
    T * x_val)
const
{
    /* initialize with x = b */
    std::copy(b_val, b_val + sub_m, x_val);

    if(!m_scheduled)
        level_schedule();

    /* parallel code using level-scheduling */
    for(mat_int_t lvl = 0; lvl < m_num_levels; ++lvl)
    {
        const mat_int_t lvl_size = m_level_offsets[lvl + 1] -
            m_level_offsets[lvl];
        const mat_int_t * lvl_rows = m_levels.data() + m_level_offsets[lvl];

        //#pragma omp parallel for
        for(mat_int_t i = 0; i < lvl_size; ++i)
        {
            const mat_int_t row = lvl_rows[i];

            if(row < sub_m)
            {
                /* solve with subdiagonal entries */
                const mat_int_t row_len = row_length(row);
                const mat_int_t * cols = row_col(row);
                const T * vals = row_val(row);

                for(mat_int_t j = 0; j < row_len; ++j)
                    if(cols[j] < row)
                        x_val[row] -= vals[j] * x_val[cols[j]];

                /* divide by diagonal entry - if available */
                if(cols[row_len - 1] == row)
                    x_val[row] /= vals[row_len - 1];
            }
        }
    }
}

/* ************************************************************************** */

template<typename T>
void
Triangular<T>::
sfsolve(
    const mat_int_t b_len,
    const mat_int_t * b_ix,
    const T * b_val,
    const mat_int_t x_len,
    const mat_int_t * x_ix,
    T * x_val)
const
{
    sub_sfsolve(m_m, b_len, b_ix, b_val, x_len, x_ix, x_val);
}

/* ************************************************************************** */

template<typename T>
void
Triangular<T>::
sub_sfsolve(
    const mat_int_t sub_m,
    const mat_int_t b_len,
    const mat_int_t * b_ix,
    const T * b_val,
    const mat_int_t x_len,
    const mat_int_t * x_ix,
    T * x_val)
const
{
    std::vector<T> full_x(sub_m, 0);

    /* scatter b */
    for(mat_int_t i = 0; i < b_len; ++i)
        if(b_ix[i] < sub_m)
            full_x[b_ix[i]] = b_val[i];

    for(mat_int_t i = 0; i < x_len; ++i)
    {
        const mat_int_t ix = x_ix[i];
        const mat_int_t row_len = row_length(ix);

        const mat_int_t * cols = row_col(ix);
        const T * vals = row_val(ix);

        /* subtract dot product from row and solution */
        for(mat_int_t j = 0; j < row_len; ++j)
            if(cols[j] != ix)
                full_x[ix] -= vals[j] * full_x[cols[j]];

        /* divide by diagonal element */
        if(cols[row_len - 1] == ix)
            full_x[ix] /= vals[row_len - 1];
    }

    /* compress solution vector */
    for(mat_int_t i = 0; i < x_len; ++i)
        x_val[i] = full_x[x_ix[i]];
}

/* ************************************************************************** */

template<typename T>
mat_int_t
Triangular<T>::
level_schedule()
const
{
    /* dispose old data */
    m_level_offsets.clear();
    m_levels.clear();

    /* transpose matrix for efficient BFS */
    Triangular_ptr<T> other = transpose();

    /* use modified BFS to find level ordering */
    std::vector<mat_int_t> deps(m_m, 0);
    std::vector<mat_int_t> roots;

    /* add all zero-in nodes to queue as first level */
    mat_int_t offset = 0;
    m_level_offsets.push_back(0);
    for(mat_int_t i = 0; i < m_m; ++i)
    {
        /**
         * Note: diagonal could be empty, hence must check indices
         */

        const mat_int_t len = row_length(i);
        const mat_int_t * ix = row_col(i);
        for(mat_int_t j = 0; j < len; ++j)
            deps[i] += (ix[j] != i);

        if(deps[i] == 0)
        {
            roots.push_back(i);
            m_levels.push_back(i);
            ++offset;
        }
    }
    m_level_offsets.push_back(offset);

    mat_int_t level = 0;
    while(!roots.empty())
    {
        const std::vector<mat_int_t> old_roots = roots;
        roots.clear();

        /* remove roots from dependencies */
        for(const mat_int_t r : old_roots)
        {
            const mat_int_t r_len = other->row_length(r);
            const mat_int_t * r_col = other->row_col(r);

            /* again, skip diagonal */
            for(mat_int_t j = 0; j < r_len; ++j)
            {
                if(r_col[j] != r)
                {
                    --deps[r_col[j]];
                    if(deps[r_col[j]] == 0)
                    {
                        roots.push_back(r_col[j]);
                        m_levels.push_back(r_col[j]);
                        ++offset;
                    }
                }
            }
        }

        m_level_offsets.push_back(offset);
        ++level;
    }

    m_num_levels = level;
    m_scheduled = true;

    return level;
}

/* ************************************************************************** */

template<typename T>
mat_int_t
Triangular<T>::
level_size(
    const mat_int_t level)
{
    if(level >= m_num_levels)
        return 0;

    return (m_level_offsets[level + 1] - m_level_offsets[level]);
}

/* ************************************************************************** */

template<typename T>
mat_int_t *
Triangular<T>::
level_nodes(
    const mat_int_t level)
{
    if(level >= m_num_levels)
        return nullptr;

    return (m_levels.data() + m_level_offsets[level]);
}

/* ************************************************************************** */

template<typename T>
Triangular_ptr<T>
Triangular<T>::
copy()
const
{
    Triangular_ptr<T> other = Triangular_ptr<T>(new Triangular<T>(m_m, m_nnz));

    std::copy(m_csr_row.begin(), m_csr_row.end(),
        other->raw_row_ptr());
    std::copy(m_csr_col.begin(), m_csr_col.end(),
        other->raw_col_ptr());
    std::copy(m_csr_val.begin(), m_csr_val.end(),
        other->raw_val_ptr());

    return other;
}

/* ************************************************************************** */

template<typename T>
Triangular_ptr<T>
Triangular<T>::
transpose()
const
{
    /* use ijk compression with swapped i/j coordinates */
    std::vector<mat_int_t> A_i(m_nnz);
    std::vector<mat_int_t> A_j(m_nnz);
    std::vector<T> A_k(m_nnz);

    mat_int_t ctr = 0;
    for(mat_int_t i = 0; i < m_m; ++i)
    {
        const mat_int_t row_len = row_length(i);
        const mat_int_t * cols = row_col(i);
        const T * vals = row_val(i);

        for(mat_int_t j = 0; j < row_len; ++j)
        {
            A_i[ctr] = cols[j];
            A_j[ctr] = i;
            A_k[ctr] = vals[j];
            ++ctr;
        }
    }

    return Triangular_ptr<T>(new Triangular<T>(m_nnz, A_i.data(),
        A_j.data(), A_k.data()));
}

/* ************************************************************************** */

template<typename T>
void
Triangular<T>::
dprint()
const
{
    sub_dprint(m_m);
}

/* ************************************************************************** */

template<typename T>
void
Triangular<T>::
sub_dprint(
    const mat_int_t sub_m)
const
{
    std::vector<T> LL(sub_m * sub_m);
    for(mat_int_t i = 0; i < sub_m; ++i)
    {
        const mat_int_t i_len = row_length(i);
        const mat_int_t * i_col = row_col(i);
        const T * i_val = row_val(i);

        for(mat_int_t j = 0; j < i_len; ++j)
            LL[i * sub_m + i_col[j]] = i_val[j];
    }

    printf("L(%d) = ", sub_m);
    for(mat_int_t i = 0; i < sub_m; ++i)
    {
        for(mat_int_t j = 0; j < sub_m; ++j)
            printf("%g ", LL[i * sub_m + j]);
        printf("\n");
    }
}

/* ************************************************************************** */

template<typename T>
void
Triangular<T>::
ijk_compress(
    const mat_int_t * A_i,
    const mat_int_t * A_j,
    const T * A_k)
{
    m_lower = true;

    /* determine major and minor size */
    m_m = 0;
    for(mat_int_t i = 0; i < m_nnz; ++i)
    {
        m_m = std::max(m_m, A_i[i]);
        m_m = std::max(m_m, A_j[i]);
    }
    ++m_m;

    /* count entries in each major cell */
    m_csr_row.resize(m_m + 1);
    std::fill(m_csr_row.begin(), m_csr_row.end(), 0);
    for(mat_int_t i = 0; i < m_nnz; ++i)
        ++m_csr_row[A_i[i]];

    /* in-place reduction to retrieve offsets */
    mat_int_t hold = m_csr_row[0];
    m_csr_row[0] = 0;
    for(mat_int_t i = 1; i < m_m + 1; ++i)
    {
        const mat_int_t res = m_csr_row[i - 1] + hold;
        hold = m_csr_row[i];
        (m_csr_row)[i] = res;
    }

    /* resize indices, values arrays */
    m_csr_col.resize(m_nnz);
    m_csr_val.resize(m_nnz);

    /* scatter minor entries and values */
    std::vector<mat_int_t> ctr(m_m, 0);
    for(mat_int_t i = 0; i < m_nnz; ++i)
    {
        if(A_j[i] > A_i[i])
            m_lower = false;

        m_csr_col[m_csr_row[A_i[i]] + ctr[A_i[i]]] = A_j[i];
        m_csr_val[m_csr_row[A_i[i]] + ctr[A_i[i]]] = A_k[i];
        ++ctr[A_i[i]];
    }

    /* sort entries per major cell using insertion sort */
    for(mat_int_t i = 0; i < m_m; ++i)
    {
        const mat_int_t i_start = m_csr_row[i];
        const mat_int_t i_end = m_csr_row[i + 1];

        for(mat_int_t p = 0; p < i_end - i_start - 1; ++p)
        {
            /* find minimum entry */
            mat_int_t min_val = m_csr_col[i_start + p];
            mat_int_t min_pos = p;
            for(mat_int_t q = p + 1; q < i_end - i_start; ++q)
            {
                const mat_int_t entry = m_csr_col[i_start + q];
                if(entry < min_val)
                {
                    min_val = entry;
                    min_pos = q;
                }
            }

            /* swap with current pivot */
            std::swap(m_csr_col[i_start + p],
                m_csr_col[i_start + min_pos]);
            std::swap(m_csr_val[i_start + p],
                m_csr_val[i_start + min_pos]);
        }
    }
}

NS_STAGING_END
NS_CULIP_END