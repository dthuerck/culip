/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/staging/flexible_triangular.h>

#include <set>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

template<typename T>
FlexibleTriangular<T>::
FlexibleTriangular(
    const mat_int_t m)
: m_m(m),
 m_row_ix(m),
 m_col_ix(m),
 m_row_val(m)
{
    for(mat_int_t i = 0; i < m; ++i)
    {
        m_row_ix[i] = new std::vector<mat_int_t>;
        m_col_ix[i] = new std::vector<mat_int_t>;
        m_row_val[i] = new std::vector<T>;
    }
}

/* ************************************************************************** */

template<typename T>
FlexibleTriangular<T>::
FlexibleTriangular(
    const csr_matrix_t<T> * A)
{
    init(A->m, A->nnz, A->csr_row, A->csr_col, A->csr_val);
}

/* ************************************************************************** */

template<typename T>
FlexibleTriangular<T>::
FlexibleTriangular(
    const Triangular<T> * L)
{
    init(L->m(), L->nnz(),
        ((Triangular<T> *) L)->raw_row_ptr(),
        ((Triangular<T> *) L)->raw_col_ptr(),
        ((Triangular<T> *) L)->raw_val_ptr());
}

/* ************************************************************************** */

template<typename T>
FlexibleTriangular<T>::
~FlexibleTriangular()
{

}

/* ************************************************************************** */

template<typename T>
mat_int_t
FlexibleTriangular<T>::
m()
const
{
    return m_m;
}

/* ************************************************************************** */

template<typename T>
mat_int_t
FlexibleTriangular<T>::
nnz()
const
{
    return m_nnz;
}

/* ************************************************************************** */

template<typename T>
mat_int_t
FlexibleTriangular<T>::
row(
    const mat_int_t row,
    mat_int_t *& ix,
    T *& val)
const
{
    ix = m_row_ix[row]->data();
    val = m_row_val[row]->data();

    return m_row_ix[row]->size();
}

/* ************************************************************************** */

template<typename T>
mat_int_t
FlexibleTriangular<T>::
col(
    const mat_int_t col,
    mat_int_t *& ix)
const
{
    ix = m_col_ix[col]->data();

    return m_col_ix[col]->size();
}

/* ************************************************************************** */

template<typename T>
void
FlexibleTriangular<T>::
set_row(
    const mat_int_t row,
    const mat_int_t nz_len,
    const mat_int_t * ix,
    const T * val)
{
    /* update NZ count */
    m_nnz -= m_row_ix[row]->size();
    m_nnz += nz_len;

    /**
     * Remove row from col ix's
     */
    for(const mat_int_t col : *m_row_ix[row])
    {
        m_col_ix[col]->erase(std::remove(m_col_ix[col]->begin(),
            m_col_ix[col]->end(), row), m_col_ix[col]->end());
    }

    /* add new row */
    m_row_ix[row]->resize(nz_len);
    m_row_val[row]->resize(nz_len);

    std::copy(ix, ix + nz_len, m_row_ix[row]->begin());
    std::copy(val, val + nz_len, m_row_val[row]->begin());

    /* insert into column list (unsorted) */
    for(const mat_int_t col : *m_row_ix[row])
        m_col_ix[col]->push_back(row);
}

/* ************************************************************************** */

template<typename T>
void
FlexibleTriangular<T>::
pivot(
    const mat_int_t r1,
    const mat_int_t r2)
{
    if(r1 == r2)
        return;

    const mat_int_t short_r = std::min(r1, r2);
    const mat_int_t long_r = std::max(r1, r2);

    /* store old rows */
    using tpl = std::pair<mat_int_t, T>;
    std::vector<tpl> short_row, long_row;

    for(mat_int_t j = 0; j < m_row_ix[short_r]->size();++j)
        short_row.emplace_back(std::make_pair((*m_row_ix[short_r])[j],
            (*m_row_val[short_r])[j]));
    for(mat_int_t j = 0; j < m_row_ix[long_r]->size();++j)
        long_row.emplace_back(std::make_pair((*m_row_ix[long_r])[j],
            (*m_row_val[long_r])[j]));

    /* sort old rows */
    auto sort_ix = [](const tpl& t0, const tpl& t1)
    {
        return (t0.first < t1.first);
    };
    std::sort(short_row.begin(), short_row.end(), sort_ix);
    std::sort(long_row.begin(), long_row.end(), sort_ix);

    /* create new rows */
    m_row_ix[short_r]->clear();
    m_row_val[short_r]->clear();
    m_row_ix[long_r]->clear();
    m_row_val[long_r]->clear();

    /**
     * remove elements from long_r, short_r from all columns
     */
    for(const tpl& sr : short_row)
        m_col_ix[sr.first]->erase(std::remove(m_col_ix[sr.first]->begin(),
            m_col_ix[sr.first]->end(), short_r),
            m_col_ix[sr.first]->end());
    for(const tpl& lr : long_row)
        m_col_ix[lr.first]->erase(std::remove(m_col_ix[lr.first]->begin(),
            m_col_ix[lr.first]->end(), long_r),
            m_col_ix[lr.first]->end());

    /* elements in both rows before column short_r just swap rows */
    mat_int_t short_ptr = 0;
    while(short_row[short_ptr].first < short_r)
    {
        m_row_ix[long_r]->push_back(short_row[short_ptr].first);
        m_row_val[long_r]->push_back(short_row[short_ptr].second);

        ++short_ptr;
    }

    mat_int_t long_ptr = 0;
    while(long_row[long_ptr].first < short_r)
    {
        m_row_ix[short_r]->push_back(long_row[long_ptr].first);
        m_row_val[short_r]->push_back(long_row[long_ptr].second);

        ++long_ptr;
    }

    /* element on (long_r, short_r) stays where it is */
    if(long_row[long_ptr].first == short_r)
    {
        m_row_ix[long_r]->push_back(long_row[long_ptr].first);
        m_row_val[long_r]->push_back(long_row[long_ptr].second);

        ++long_ptr;
    }

    /**
     * remove elements in col short_r from rows (short_r, long_r),
     * and push them into long_r
     */
    for(const mat_int_t rem_row : *m_col_ix[short_r])
    {
        if(rem_row < short_r || rem_row > long_r)
            continue;

        /* erase column short_r from row rem_row */
        for(mat_int_t j = 0; j < m_row_ix[rem_row]->size(); ++j)
        {
            if((*m_row_ix[rem_row])[j] == short_r)
            {
                /* push element into long_r */
                if(rem_row > short_r && rem_row < long_r)
                {
                    m_row_ix[long_r]->push_back(rem_row);
                    m_row_val[long_r]->push_back((*m_row_val[rem_row])[j]);
                }

                std::swap((*m_row_ix[rem_row])[j], m_row_ix[rem_row]->back());
                std::swap((*m_row_val[rem_row])[j],
                    m_row_val[rem_row]->back());

                m_row_ix[rem_row]->pop_back();
                m_row_val[rem_row]->pop_back();

                break;
            }
        }
    }
    m_col_ix[short_r]->erase(
        std::remove_if(m_col_ix[short_r]->begin(), m_col_ix[short_r]->end(),
        [&short_r, &long_r](const mat_int_t rr)
        {
            return (rr > short_r && rr < long_r);
        }),
        m_col_ix[short_r]->end());

    /* elements in long_r from short_r before diagonal go into column short_r */
    mat_int_t ctr = 0;
    while(long_row[long_ptr].first < long_r)
    {
        const mat_int_t new_row = long_row[long_ptr].first;

        m_row_ix[new_row]->push_back(short_r);
        m_row_val[new_row]->push_back(long_row[long_ptr].second);

        m_col_ix[short_r]->push_back(new_row);
        ++ctr;

        ++long_ptr;
    }

    /* swap diagonal entries (always assume NZ diagonal) */
    m_row_ix[short_r]->push_back(short_r);
    m_row_val[short_r]->push_back(long_row[long_ptr].second);

    m_row_ix[long_r]->push_back(long_r);
    m_row_val[long_r]->push_back(short_row[short_ptr].second);

    /* update col ix with new rows */
    for(const mat_int_t c : *m_row_ix[short_r])
        m_col_ix[c]->push_back(short_r);
    for(const mat_int_t c : *m_row_ix[long_r])
        m_col_ix[c]->push_back(long_r);

    /**
     * permute columns long_r, short_r below row long_r in rows
     * Note: without the 'processed' array, wouldn't work on rows
     * that contain both columns, since there we would swap forth and back
     */
    const auto switch_col = [short_r, long_r](const mat_int_t rr)
        {
            return (rr == short_r ? long_r :
                (rr == long_r ? short_r : rr));
        };
    std::vector<bool> processed(m_m, false);
    for(const mat_int_t r : *m_col_ix[short_r])
    {
        if(r > long_r)
            std::transform(m_row_ix[r]->begin(), m_row_ix[r]->end(),
                m_row_ix[r]->begin(), switch_col);
        processed[r] = true;
    }
    for(const mat_int_t r : *m_col_ix[long_r])
        if(r > long_r && !processed[r])
            std::transform(m_row_ix[r]->begin(), m_row_ix[r]->end(),
                m_row_ix[r]->begin(), switch_col);

    auto short_rem_it = std::partition(m_col_ix[short_r]->begin(),
        m_col_ix[short_r]->end(), [&long_r](const mat_int_t rr)
        {
            return (rr <= long_r);
        });
    auto short_old_end = m_col_ix[short_r]->end();
    auto long_rem_it = std::partition(m_col_ix[long_r]->begin(),
        m_col_ix[long_r]->end(), [&long_r](const mat_int_t rr)
        {
            return (rr <= long_r);
        });
    auto long_old_end = m_col_ix[long_r]->end();

    /* save content that must be swapped */
    std::vector<mat_int_t> short_r_save(short_rem_it, short_old_end);
    std::vector<mat_int_t> long_r_save(long_rem_it, long_old_end);

    m_col_ix[short_r]->erase(short_rem_it, short_old_end);
    m_col_ix[long_r]->erase(long_rem_it, long_old_end);

    m_col_ix[short_r]->insert(m_col_ix[short_r]->end(),
        long_r_save.begin(), long_r_save.end());
    m_col_ix[long_r]->insert(m_col_ix[long_r]->end(),
        short_r_save.begin(), short_r_save.end());
}

/* ************************************************************************** */

template<typename T>
void
FlexibleTriangular<T>::
order()
{
    mat_int_t nz = 0;

    /* sort row indices and values */
    for(mat_int_t i = 0; i < m_m; ++i)
    {
        const mat_int_t i_len = m_row_ix[i]->size();
        mat_int_t * i_ix = m_row_ix[i]->data();
        T * i_val = m_row_val[i]->data();

        nz += i_len;

        for(mat_int_t p = 0; p < i_len - 1; ++p)
        {
            /* find minimum entry */
            mat_int_t min_val = i_ix[p];
            mat_int_t min_pos = p;
            for(mat_int_t q = p + 1; q < i_len; ++q)
            {
                const mat_int_t entry = i_ix[q];
                if(entry < min_val)
                {
                    min_val = entry;
                    min_pos = q;
                }
            }

            /* swap with current pivot */
            std::swap(i_ix[p], i_ix[min_pos]);
            std::swap(i_val[p], i_val[min_pos]);
        }
    }

    /* sort column indices */
    for(mat_int_t j = 0; j < m_m; ++j)
        std::sort(m_col_ix[j]->begin(), m_col_ix[j]->end());
}

/* ************************************************************************** */

template<typename T>
mat_int_t
FlexibleTriangular<T>::
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
        mat_int_t * row_ix;
        T * row_val;
        const mat_int_t row_len = row(b_i, row_ix, row_val);

        for(mat_int_t i = 0; i < row_len && !nz_marker[b_i]; ++i)
        {
            nz_marker[b_i] = nz_marker[row_ix[i]] | nz_marker[b_i];
        }

        if(nz_marker[b_i])
            m_x_ix.push_back(b_i);
    }

    std::sort(m_x_ix.begin(), m_x_ix.end());

    return m_x_ix.size();
}

/* ************************************************************************** */

template<typename T>
void
FlexibleTriangular<T>::
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
FlexibleTriangular<T>::
sub_sanalysis_export(
    mat_int_t * x_ix)
const
{
    std::copy(m_x_ix.begin(), m_x_ix.end(), x_ix);
}

/* ************************************************************************** */

template<typename T>
void
FlexibleTriangular<T>::
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

        if(ix >= sub_m)
            continue;

        mat_int_t * cols;
        T * vals;
        const mat_int_t row_len = row(ix, cols, vals);

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
        if(m_x_ix[i] < sub_m)
            x_val[i] = full_x[m_x_ix[i]];
}

/* ************************************************************************** */

template<typename T>
void
FlexibleTriangular<T>::
sub_dfsolve(
    const mat_int_t sub_m,
    const T * b_val,
    T * x_val)
const
{
    /* initialize with x = b */
    std::copy(b_val, b_val + sub_m, x_val);

    for(mat_int_t i = 0; i < sub_m; ++i)
    {
        /* solve with subdiagonal entries */
        mat_int_t * cols;
        T * vals;
        const mat_int_t row_len = row(i, cols, vals);

        for(mat_int_t j = 0; j < row_len; ++j)
            if(cols[j] < i)
                x_val[i] -= vals[j] * x_val[cols[j]];

        /* divide by diagonal entry - if available */
        if(cols[row_len - 1] == i)
            x_val[i] /= vals[row_len - 1];
    }
}

/* ************************************************************************** */

template<typename T>
void
FlexibleTriangular<T>::
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
FlexibleTriangular<T>::
dbsolve(
    const T * b_val,
    T * x_val)
const
{
    /* initialize x = b */
    std::copy(b_val, b_val + m_m, x_val);

    for(mat_int_t i = m_m - 1; i >= 0; --i)
    {
        mat_int_t * l_col;
        T * l_val;
        const mat_int_t l_size = row(i, l_col, l_val);

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
Triangular_ptr<T>
FlexibleTriangular<T>::
to_fixed()
{
    /* sort elements in row / col */
    order();

    /* now export */
    Triangular_ptr<T> L = Triangular_ptr<T>(new Triangular<T>(m_m, m_nnz));

    /* row indices */
    L->raw_row_ptr()[0] = 0;
    for(mat_int_t i = 1; i < m_m + 1; ++i)
        L->raw_row_ptr()[i] = L->raw_row_ptr()[i - 1] + m_row_ix[i - 1]->size();

    /* row indices and values */
    for(mat_int_t i = 0; i < m_m; ++i)
    {
        std::copy(m_row_ix[i]->begin(), m_row_ix[i]->end(),
            L->raw_col_ptr() + L->raw_row_ptr()[i]);
        std::copy(m_row_val[i]->begin(), m_row_val[i]->end(),
            L->raw_val_ptr() + L->raw_row_ptr()[i]);
    }

    return L;
}

/* ************************************************************************** */

template<typename T>
void
FlexibleTriangular<T>::
init(
    const mat_int_t m,
    const mat_int_t nnz,
    const mat_int_t * csr_row,
    const mat_int_t * csr_col,
    const T *  csr_val)
{
    m_m = m;
    m_nnz = nnz;

    /* clean up */
    for(std::vector<mat_int_t> * v : m_row_ix)
        delete v;
    for(std::vector<mat_int_t> * v : m_col_ix)
        delete v;
    for(std::vector<T> * v : m_row_val)
        delete v;

    m_row_ix.clear();
    m_col_ix.clear();
    m_row_val.clear();

    /* create new lists-of-arrays for rows */
    std::vector<mat_int_t> col_counts(m_m, 0);
    m_row_ix.resize(m_m);
    m_row_val.resize(m_m);

    for(mat_int_t i = 0; i < m_m; ++i)
    {
        const mat_int_t i_len = csr_row[i + 1] - csr_row[i];
        const mat_int_t * i_ix = csr_col + csr_row[i];
        const T * i_val = csr_val + csr_row[i];

        m_row_ix[i] = new std::vector<mat_int_t>;
        m_row_val[i] = new std::vector<T>;

        m_row_ix[i]->reserve(i_len / 2 + 1);
        m_row_val[i]->reserve(i_len / 2 + 1);

        for(mat_int_t j = 0; j < i_len; ++j)
        {
            /* use only lower triangular part */
            if(i_ix[j] > i)
            {
                m_nnz -= (i_len - j);
                break;
            }

            m_row_ix[i]->push_back(i_ix[j]);
            m_row_val[i]->push_back(i_val[j]);

            ++col_counts[i_ix[j]];
        }
    }

    /* add column indices */
    m_col_ix.resize(m_m);
    for(mat_int_t j = 0; j < m_m; ++j)
    {
        m_col_ix[j] = new std::vector<mat_int_t>(col_counts[j]);
        col_counts[j] = 0;
    }

    for(mat_int_t i = 0; i < m_m; ++i)
    {
        const mat_int_t i_len = csr_row[i + 1] - csr_row[i];
        const mat_int_t * i_ix = csr_col + csr_row[i];

        for(mat_int_t j = 0; j < i_len; ++j)
        {
            const mat_int_t col = i_ix[j];

            if(col > i)
                break;

            (*m_col_ix[col])[col_counts[col]] = i;
            ++col_counts[col];
        }
    }
}

NS_STAGING_END
NS_CULIP_END
