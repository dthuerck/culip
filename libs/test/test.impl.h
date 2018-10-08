/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

extern "C" {
    #include <mmio.h>
};

#include <libs/test/test.h>
#include <libs/la/sparse_la.cuh>
#include <sstream>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

using namespace NS_CULIP::NS_LA;

NS_CULIP_BEGIN
NS_TEST_BEGIN

template <typename T>
size_t
Test<T>::
read_mm_triplets(
    mat_int_t& m,
    mat_int_t& n,
    std::vector<mat_int_t>& ii,
    std::vector<mat_int_t>& jj,
    std::vector<T>& cc,
    const char* filename)
{
    ii.clear();
    jj.clear();
    cc.clear();

    FILE * mm_in = NULL;
    mm_in = fopen(filename, "r");

    if(mm_in != NULL)
    {
        /* get meta data */
        MM_typecode mtype;
        int i_m, i_n, i_nnz;

        mm_read_banner(mm_in, &mtype);
        mm_read_mtx_crd_size(mm_in, &i_m, &i_n, &i_nnz);

        m = i_m;
        n = i_n;
        ii.resize(i_nnz);
        jj.resize(i_nnz);
        cc.resize(i_nnz);

        std::vector<double> d_cc(i_nnz);

        mm_read_mtx_crd_data(mm_in, m, n, i_nnz, &ii[0], &jj[0],
            &d_cc[0], mtype);

        for(mat_int_t i = 0; i < i_nnz; ++i)
        {
            /* change to 0-based index and adapt to type T */
            --ii[i];
            --jj[i];

            cc[i] = (T) d_cc[i];
        }

        /**
         * In case of symmetry, mtx only stores a triangular
         * part of the matrix, hence we need to duplicate it.
         */
        size_t sym_nnz = 0;
        if(mm_is_symmetric(mtype))
        {
            for(mat_int_t i = 0; i < i_nnz; ++i)
            {
                if (ii[i] != jj[i])
                {
                    ii.push_back(jj[i]);
                    jj.push_back(ii[i]);
                    cc.push_back(cc[i]);

                    ++sym_nnz;
                }
            }
        }

        fclose(mm_in);

        return (i_nnz + sym_nnz);
    }
    else
    {
        std::cerr << "Couldn't open " << filename << ", exiting..." <<
            std::endl;
        std::exit(EXIT_FAILURE);
    }

    return 0;
}

/* ************************************************************************** */

template<typename T>
csr_matrix_ptr<T>
Test<T>::
matrix_coo_to_csr(
    const mat_int_t& m,
    const mat_int_t& n,
    const mat_int_t& nnz,
    const std::vector<mat_int_t>& ii,
    const std::vector<mat_int_t>& jj,
    const std::vector<T>& cc)
{
    /* first pass: count elements per row */
    std::vector<mat_int_t> row_size(m, 0);
    for(mat_int_t i = 0; i < nnz; ++i)
        ++row_size[ii[i]];

    // /* check for empty rows */
    // for(mat_int_t i = 0; i < m; ++i)
    //     if(row_size[i] == 0)
    //         printf("!!! WARNING: row %d is empty!\n", i);

    /* second pass: write column offsets and reorder elements */
    csr_matrix_ptr<T> m_ptr = make_csr_matrix_ptr<T>(m, n, nnz, false);

    m_ptr->csr_row[0] = 0;
    for(mat_int_t i = 1; i < m + 1; ++i)
        m_ptr->csr_row[i] = m_ptr->csr_row[i - 1] + row_size[i - 1];

    std::fill(row_size.begin(), row_size.end(), (mat_int_t) 0);

    for(mat_int_t i = 0; i < nnz; ++i)
    {
        const mat_int_t row = ii[i];

        m_ptr->csr_col[m_ptr->csr_row[row] + row_size[row]] = jj[i];
        m_ptr->csr_val[m_ptr->csr_row[row] + row_size[row]] = cc[i];

        ++row_size[row];
    }

    /* third pass: sort elements per column (insertion sort on pairs) */
    for(mat_int_t r = 0; r < m; ++r)
    {
        const mat_int_t row_nnz = m_ptr->csr_row[r + 1] -
            m_ptr->csr_row[r];
        const mat_int_t row_offset = m_ptr->csr_row[r];

        for(mat_int_t o = 0; o < row_nnz - 1; ++o)
        {
            mat_int_t min_col = m_ptr->csr_col[row_offset + o];
            mat_int_t min_id = o;

            for(mat_int_t i = o + 1; i < row_nnz; ++i)
            {
                const mat_int_t col = m_ptr->csr_col[row_offset + i];
                if (col < min_col)
                {
                    min_col = col;
                    min_id = i;
                }
            }

            if(min_id != o)
            {
                std::swap(m_ptr->csr_col[row_offset + o],
                    m_ptr->csr_col[row_offset + min_id]);
                std::swap(m_ptr->csr_val[row_offset + o],
                    m_ptr->csr_val[row_offset + min_id]);
            }
        }
    }

    return m_ptr;
}

/* ************************************************************************** */

template<typename T>
void
Test<T>::
matrix_csr_to_coo(
    const csr_matrix_t<T> * A,
    std::vector<mat_int_t>& ii,
    std::vector<mat_int_t>& jj,
    std::vector<T>& cc)
{
    ii.clear();
    jj.clear();
    cc.clear();

    ii.reserve(A->nnz);
    jj.reserve(A->nnz);
    cc.reserve(A->nnz);

    for(mat_int_t i = 0; i < A->m; ++i)
    {
        for(mat_int_t j = A->csr_row[i]; j < A->csr_row[i + 1]; ++j)
        {
            ii.push_back(i);
            jj.push_back(A->csr_col[j]);
            cc.push_back(A->csr_val[j]);
        }
    }
}

/* ************************************************************************** */

template<typename T>
csr_matrix_ptr<T>
Test<T>::
read_matrix_csr(
    const char* filename,
    const bool transpose)
{
    mat_int_t m, n, nnz;
    std::vector<mat_int_t> ii, jj;
    std::vector<T> cc;

    nnz = read_mm_triplets(m, n, ii, jj, cc, filename);
    return matrix_coo_to_csr(transpose ? n : m, transpose ? m : n, nnz,
        transpose? jj : ii, transpose ? ii : jj, cc);
}

/* ************************************************************************** */

template<typename T>
dense_vector_ptr<T>
Test<T>::
read_dense_vector(
    const char* filename,
    const T shift)
{
    FILE* mm_in = NULL;
    mm_in = fopen(filename , "r");

    if(mm_in != NULL)
    {
        /* read header (get size and check type) */
        MM_typecode type;
        mm_read_banner(mm_in, &type);

        if(!mm_is_array(type))
        {
            std::cerr << "Only arrays supported, exiting..." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        int m, n;
        mm_read_mtx_array_size(mm_in, &m, &n);

        m = std::max(m, n);
        dense_vector_ptr<T> vec = make_managed_dense_vector_ptr<T>(m, false);
        int i = 0;

        int skipped = 0;
        std::ifstream mm_in_s(filename);
        if(mm_in_s.good())
        {
            /* jump first two header lines */
            std::string line;

            for (; std::getline(mm_in_s, line); )
            {
                /* jump first two header lines */
                if(line[0] == '%')
                    continue;

                /* skip line with size information */
                if(skipped == 0)
                {
                    ++skipped;
                    continue;
                }

                std::stringstream ss(line);
                ss >> vec->dense_val[i];
                vec->dense_val[i] -= shift;
                i++;
            }
        }
        mm_in_s.close();
        fclose(mm_in);

        return vec;
    }
    else
    {
        std::cerr << "Couldn't open " << filename << ", exiting..." <<
            std::endl;
        std::exit(EXIT_FAILURE);
    }

    return nullptr;
}

/* ************************************************************************** */

template<typename T>
dense_vector_ptr<T>
Test<T>::
read_dense_vector_from_sparse(
    const char* filename,
    const T shift)
{
    FILE* mm_in = NULL;
    mm_in = fopen(filename , "r");

    if(mm_in != NULL)
    {
        /* read header (get size and check type) */
        MM_typecode type;
        mm_read_banner(mm_in, &type);

        if(!mm_is_sparse(type))
        {
            std::cerr << "Only sparse supported, exiting..." << std::endl;
            std::exit(EXIT_FAILURE);
        }

        mat_int_t m, n, nnz;
        std::vector<mat_int_t> ii, jj;
        std::vector<T> cc;
        nnz = read_mm_triplets(m, n, ii, jj, cc, filename);

        dense_vector_ptr<T> vec = make_managed_dense_vector_ptr<T>(m, false);

        /* initialize with 0 */
        memset(vec->dense_val, 0, m * sizeof(T));

        for(mat_int_t i = 0; i < nnz; ++i)
            vec->dense_val[ii[i]] = cc[i] - shift;

        fclose(mm_in);

        return vec;
    }
    else
    {
        std::cerr << "Couldn't open " << filename << ", exiting..." <<
            std::endl;
        std::exit(EXIT_FAILURE);
    }

    return dense_vector_ptr<T>(nullptr);
}

/* ************************************************************************** */

template<typename T, bool lower_triangular>
struct is_unwanted
{
    __host__
    bool operator()(const thrust::tuple<mat_int_t, mat_int_t, T>& t0)
    {
        return
            ((lower_triangular && thrust::get<1>(t0) > thrust::get<0>(t0))
            ||
            (!lower_triangular && thrust::get<1>(t0) < thrust::get<0>(t0)));
    }
};

template<typename T>
csr_matrix_ptr<T>
Test<T>::
extract_triangular_part(
    const csr_matrix_t<T> * A,
    const bool lower_triangular)
{
    std::vector<mat_int_t> ii, jj;
    std::vector<T> cc;

    /* extract COO triplets */
    matrix_csr_to_coo(A, ii, jj, cc);

    /* remove all unwanted elements */
    auto coo_it_begin =
        thrust::make_zip_iterator(
            thrust::make_tuple(
                ii.begin(),
                jj.begin(),
                cc.begin()));
    auto coo_it_end =
        thrust::make_zip_iterator(
            thrust::make_tuple(
                ii.end(),
                jj.end(),
                cc.end()));

    mat_int_t new_elems;
    if(lower_triangular)
    {
        const auto new_it =
            thrust::remove_if(thrust::host, coo_it_begin, coo_it_end,
            is_unwanted<T, true>());
        new_elems = std::distance(coo_it_begin, new_it);
    }
    else
    {
        const auto new_it =
            thrust::remove_if(thrust::host, coo_it_begin, coo_it_end,
            is_unwanted<T, false>());
        new_elems = std::distance(coo_it_begin, new_it);
    }

    ii.resize(new_elems);
    jj.resize(new_elems);
    cc.resize(new_elems);

    /* create new CSR matrix and return */
    return matrix_coo_to_csr(
        A->m,
        A->n,
        A->nnz,
        ii,
        jj,
        cc);
}

/* ************************************************************************** */

template<typename T>
bool
Test<T>::
compare_csr_matrix(
    const csr_matrix_t<T> * A1,
    const csr_matrix_t<T> * A2,
    const T tolerance)
{
    bool equal = true;
    equal &= (A1->m == A2->m);
    equal &= (A1->n == A2->n);
    equal &= compare_sparse_matrix(A1->csr_row, A1->csr_col, A1->csr_val,
        A2->csr_row, A2->csr_col, A2->csr_val, A1->m, tolerance, true);
    equal &= compare_sparse_matrix(A2->csr_row, A2->csr_col, A2->csr_val,
        A1->csr_row, A1->csr_col, A1->csr_val, A2->m, tolerance, false);

    return equal;
}

/* ************************************************************************** */

template<typename T>
bool
Test<T>::
compare_dense_vector(
    const dense_vector_t<T> * a,
    const dense_vector_t<T> * b,
    const T tolerance)
{
    if(a->m != b->m)
        return false;

    bool equal = true;

    for(mat_int_t i = 0; i < a->m; ++i)
    {
        if(std::abs(a->dense_val[i] - b->dense_val[i]) > tolerance)
        {
            std::cout << "Entries " << i << " exceed tolerance (" <<
                a->dense_val[i] << ", " << b->dense_val[i] <<
                ")" << std::endl;
            equal = false;
        }
    }

    return equal;
}

/* ************************************************************************** */

template<typename T>
bool
Test<T>::
write_csr_matrix(
    const csr_matrix_t<T> * A,
    const char * path)
{
    const int m = A->m;
    const int n = A->n;
    const int nnz = A->nnz;

    /* copy to host, if necessary */
    const csr_matrix_t<T> * B = A;

    csr_matrix_ptr<T> h_A = make_csr_matrix_ptr<T>(false);
    if(A->on_device)
    {
        *h_A = A;
        B = h_A.get();
    }

    /**
     * first step: create COO representation
     */
    std::vector<int> ii(nnz);
    std::vector<int> jj(nnz);
    std::vector<double> cc(nnz);

    int ctr = 0;
    for(int i = 0; i < m; ++i)
    {
        const int B_offset = B->csr_row[i];
        const int B_size = B->csr_row[i + 1] - B_offset;

        for(int j = 0; j < B_size; ++j)
        {
            ii[ctr] = i;
            jj[ctr] = B->csr_col[B_offset + j];
            cc[ctr] = (double) B->csr_val[B_offset + j];

            ++ctr;
        }
    }

    /**
     * second step: write COO form to file
     */
    MM_typecode matcode;

    mm_initialize_typecode(&matcode);
    mm_set_general(&matcode);
    mm_set_matrix(&matcode);
    mm_set_real(&matcode);
    mm_set_sparse(&matcode);

    FILE * f = fopen(path, "w");

    if(f != NULL)
    {
        mm_write_banner(f, matcode);
        mm_write_mtx_crd_size(f, m, n, nnz);

        for(int i = 0; i < nnz; ++i)
            fprintf(f, "%d %d %.12g\n", (ii[i] + 1), (jj[i] + 1), (double) cc[i]);

        fclose(f);

        return true;
    }

    return false;
}

/* ************************************************************************** */

template<typename T>
bool
Test<T>::
write_col_major_matrix(
    const col_major_matrix_t<T> * A,
    const char * path)
{
    /* copy to host if necessary */
    const col_major_matrix_t<T> * B = A;

    col_major_matrix_ptr<T> h_A = make_col_major_matrix_ptr<T>(false);
    if(A->on_device)
    {
        *h_A = A;
        B = h_A.get();
    }

    FILE * f = fopen(path, "w");

    if(f != NULL)
    {
        MM_typecode veccode;
        mm_initialize_typecode(&veccode);
        mm_set_general(&veccode);
        mm_set_matrix(&veccode);
        mm_set_array(&veccode);
        mm_set_real(&veccode);

        mm_write_banner(f, veccode);
        mm_write_mtx_array_size(f, B->m, B->n);

        for(int i = 0; i < B->m * B->n; ++i)
            fprintf(f, "%.12g\n", (double) B->dense_val[i]);

        fclose(f);
        return true;
    }

    return false;
}

/* ************************************************************************** */

template<typename T>
bool
Test<T>::
write_dense_vector(
    const dense_vector_t<T> * v,
    const char * path)
{
    /* copy to host if necessary */
    const dense_vector_t<T> * w = v;

    dense_vector_ptr<T> h_v = make_managed_dense_vector_ptr<T>(false);
    if(v->on_device)
    {
        *h_v = v;
        w = h_v.get();
    }

    FILE * f = fopen(path, "w");

    if(f != NULL)
    {
        MM_typecode veccode;
        mm_initialize_typecode(&veccode);
        mm_set_general(&veccode);
        mm_set_matrix(&veccode);
        mm_set_array(&veccode);
        mm_set_real(&veccode);

        mm_write_banner(f, veccode);
        mm_write_mtx_array_size(f, w->m, 1);

        for(int i = 0; i < w->m; ++i)
            fprintf(f, "%.12g\n", (double) w->dense_val[i]);

        fclose(f);
        return true;
    }

    return false;
}

/* ************************************************************************** */

template<typename T>
bool
Test<T>::
compare_sparse_matrix(
    const mat_int_t * L_p,
    const mat_int_t * L_i,
    const T * L_x,
    const mat_int_t * R_p,
    const mat_int_t * R_i,
    const T * R_x,
    const mat_int_t lead,
    const T tolerance,
    const bool R_right)
{
    bool equal = true;

    for(mat_int_t l = 0; l < lead; ++l)
    {
        const mat_int_t L_size = L_p[l + 1] - L_p[l];
        const mat_int_t R_size = R_p[l + 1] - R_p[l];

        const mat_int_t L_offset = L_p[l];
        const mat_int_t R_offset = R_p[l];

        for(mat_int_t i = 0; i < L_size; ++i)
        {
            const mat_int_t L_o = L_i[L_offset + i];
            const T L_val = std::abs(L_x[L_offset + i]);

            bool found = false;
            if(L_val > tolerance)
            {
                /* find corresponding entry in R matrix */
                for(mat_int_t ii = 0; ii < R_size; ++ii)
                {
                    const mat_int_t R_o = R_i[R_offset + ii];
                    const T R_val = std::abs(R_x[R_offset + ii]);

                    if(R_o == L_o)
                    {
                        const T diff = std::abs(R_val - L_val);
                        found = (diff < tolerance);
                    }
                }

                if(!found)
                {
                    std::cerr << "No corresponding entry in "
                        << (R_right ? "R" : "L")
                        << " found for " <<
                        l << " / " << L_o << " / " << L_val << std::endl;
                }

                equal &= found;
            }
        }
    }

    return equal;
}

/* ************************************************************************** */

template<typename T>
dense_vector_ptr<T>
TestLA<T>::
mat_vec_multiply(
    const csr_matrix_t<T> * A,
    const dense_vector_t<T> * x,
    const gpu_handle_ptr& gpu_handle,
    const bool transpose)
{
    /* upload matrix, vector data to GPU */
    csr_matrix_ptr<T> gpu_A = make_csr_matrix_ptr<T>(true);
    dense_vector_ptr<T> gpu_x = make_managed_dense_vector_ptr<T>(true);
    dense_vector_ptr<T> gpu_y = make_managed_dense_vector_ptr<T>(
        transpose ? A->n : A->m, true);

    *gpu_A = A;
    *gpu_x = x;

    /* save old cublas / cusparse mode */
    gpu_handle->push_scalar_mode();
    gpu_handle->set_scalar_mode(false);

    /* execute matrix-vector multiplication on GPU */
    const T alpha = (T) 1.0;
    const T beta = (T) 0;

    T_csrmv<T>(
        gpu_handle,
        transpose ? CUSPARSE_OPERATION_TRANSPOSE :
            CUSPARSE_OPERATION_NON_TRANSPOSE,
        gpu_A.get(),
        gpu_x.get(),
        gpu_y.get(),
        &alpha,
        &beta);

    CHECK_CUDA(cudaDeviceSynchronize());

    /* download solution and save it as sparse vector */
    dense_vector_ptr<T> y = make_managed_dense_vector_ptr<T>(false);
    *y = gpu_y.get();

    /* restore cusparse mode */
    gpu_handle->pop_scalar_mode();

    return y;
}

template<typename T>
T
TestLA<T>::
norm2(
    const dense_vector_t<T> * x)
{
    return norm2(x->m, x->dense_val);
}

/* ************************************************************************** */

template<typename T>
T
TestLA<T>::
norm2(
    const mat_int_t v_len,
    const T * v_val)
{
    /* copy algorithm from NETLIB's lapack description */
    T absxi, scale, ssq;

    scale = 0;
    ssq = (T) 1;
    for(mat_int_t i = 0; i < v_len; ++i)
    {
        if(v_val[i] == 0)
            continue;

        absxi = std::abs(v_val[i]);
        if(scale < absxi)
        {
            ssq = ((T) 1) + ssq * (scale / absxi) * (scale / absxi);
            scale = absxi;
        }
        else
        {
            ssq += (absxi / scale) * (absxi / scale);
        }
    }

    return (scale * std::sqrt(ssq));
}

NS_TEST_END
NS_CULIP_END