/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/utils/types.cuh>

NS_CULIP_BEGIN


/**
 * *****************************************************************************
 * ****************************** DENSE_VECTOR_T *******************************
 * *****************************************************************************
 */

template<typename T>
dense_vector_t<T>::
~dense_vector_t()
{

}

/* ************************************************************************** */

template<typename T>
T&
dense_vector_t<T>::
operator[](
    const mat_int_t i)
{
    if(on_device)
    {
        std::cerr <<
            "Error accessing memory on device with operator[] is not supported"
            << " at" << __FILE__ << ":" << __LINE__ << ", exiting..." <<
            std::endl;
        std::exit(EXIT_FAILURE);
    }
    else
    {
        return dense_val[i];
    }
}

/* ************************************************************************** */

template<typename T>
const T&
dense_vector_t<T>::
operator[](
    const mat_int_t i)
const
{
    if(on_device) {
        std::cerr <<
            "Error accessing memory on device with operator[] is not supported"
            << " at" << __FILE__ << ":" << __LINE__ << ", exiting..." <<
            std::endl;
        std::exit(EXIT_FAILURE);
    }
    else
    {
        return dense_val[i];
    }
}

/* ************************************************************************** */

template<typename T>
void
dense_vector_t<T>::
print(
    const char *s)
const
{
    T * v_h;

    std::cout << s << std::endl;
    std::cout << "m " << m << " (on device " << on_device << ", managed " <<
        is_managed() << ")" << std::endl;
    if (/*_dnd ||*/ (dense_val == nullptr)) {
        std::cout << "NULL" << std::endl;
        return;
    }


    if (on_device) {
        /* allocate temporary memory */
        v_h = new T[m];
        /* copy data to the host */
        CHECK_CUDA(cudaMemcpy(v_h, dense_val, m*sizeof(T), cudaMemcpyDeviceToHost));
    }
    else{
        /* set pointers */
        v_h = dense_val;
    }

    for(int i=0; i<m; i++) {
        printf("\t%d %g\n", i, (double) v_h[i]);
    }

    /* free temporary memory */
    if (on_device) {
        delete[] v_h;
    }
}

/* ************************************************************************** */

template<typename T>
thrust::device_ptr<T>
dense_vector_t<T>::
dense_val_ptr()
const
{
    return thrust::device_pointer_cast(dense_val);
}

/* ************************************************************************** */

template<typename T>
dense_vector_t<T>::
dense_vector_t(
    const mat_size_t _m,
    const bool _on_device,
    T * _dense_val)
: m(_m),
  on_device(_on_device),
  dense_val(_dense_val)
{

}

/* ************************************************************************** */

template<typename T>
raw_dense_vector_t<T>::
raw_dense_vector_t()
: dense_vector_t<T>(0, true, nullptr)
{

}

/* ************************************************************************** */

template<typename T>
raw_dense_vector_t<T>::
raw_dense_vector_t(
    const mat_size_t _m,
    const bool _on_device,
    T * _dense_val)
: dense_vector_t<T>(_m, _on_device, _dense_val)
{

}

/* ************************************************************************** */

template<typename T>
raw_dense_vector_t<T>::
~raw_dense_vector_t()
{

}

/* ************************************************************************** */

template<typename T>
bool
raw_dense_vector_t<T>::
is_managed()
const
{
    return false;
}

/* ************************************************************************** */

template<typename T>
void
raw_dense_vector_t<T>::
operator=(
    const dense_vector_t<T> * vec)
{
    /* target a raw pointer, thus clamp data to capacity */
    const mat_int_t copy_size = std::min(this->m, vec->m);

    cudaMemcpyKind copy_direction = cudaMemcpyHostToDevice;

    if(!vec->on_device && !this->on_device)
        std::copy(vec->dense_val, vec->dense_val + copy_size, this->dense_val);

    if(vec->on_device)
    {
        if(this->on_device)
            copy_direction = cudaMemcpyDeviceToDevice;
        if(!this->on_device)
            copy_direction = cudaMemcpyDeviceToHost;

        if(vec->dense_val != nullptr)
            CHECK_CUDA(cudaMemcpy(this->dense_val, vec->dense_val,
                copy_size * sizeof(T), copy_direction));
    }
}

/* ************************************************************************** */

template<typename T>
managed_dense_vector_t<T>::
managed_dense_vector_t(
    const mat_size_t _m,
    const bool _on_device)
: dense_vector_t<T>(_m, _on_device, nullptr)
{
    _alloc(_m, _on_device);
}

/* ************************************************************************** */

template<typename T>
managed_dense_vector_t<T>::
managed_dense_vector_t(
    const bool _on_device)
: dense_vector_t<T>(0, _on_device, nullptr)
{

}

/* ************************************************************************** */

template<typename T>
managed_dense_vector_t<T>::
~managed_dense_vector_t()
{
    _free();
}

/* ************************************************************************** */

template<typename T>
bool
managed_dense_vector_t<T>::
is_managed()
const
{
    return true;
}

/* ************************************************************************** */

template<typename T>
void
managed_dense_vector_t<T>::
operator=(
    const dense_vector_t<T> * vec)
{
    /* only (re-)allocate memory if necessary */
    _alloc(vec->m, this->on_device);

    cudaMemcpyKind copy_direction;

    if(!vec->on_device)
    {
        if(!this->on_device)
            std::copy(vec->dense_val, vec->dense_val + vec->m, this->dense_val);
        if(this->on_device)
            CHECK_CUDA(cudaMemcpy(this->dense_val, vec->dense_val,
                vec->m * sizeof(T), cudaMemcpyHostToDevice));
    }

    if(vec->on_device)
    {
        if(this->on_device)
            copy_direction = cudaMemcpyDeviceToDevice;
        if(!this->on_device)
            copy_direction = cudaMemcpyDeviceToHost;

        if(vec->dense_val != nullptr)
            CHECK_CUDA(cudaMemcpy(this->dense_val, vec->dense_val,
                vec->m * sizeof(T), copy_direction));
    }
}

/* ************************************************************************** */

template<typename T>
void
managed_dense_vector_t<T>::
_alloc(
    const mat_size_t _m,
    const bool _on_device)
{
    if(this->dense_val == nullptr || _m > this->m)
    {
        _free();

        this->dense_val = (T *) GlobalMemPool().request_memory(
            _on_device, _m * sizeof(T), sizeof(T));
    }

    this->m = _m;
    this->on_device = _on_device;
}

/* ************************************************************************** */

template<typename T>
void
managed_dense_vector_t<T>::
_free()
{
    if(this->m == 0)
        return;

    if(this->dense_val == nullptr)
        return;

    GlobalMemPool().release_memory(this->on_device,
        (char *) this->dense_val);

    this->m = 0;
    this->dense_val = nullptr;
}

/* ************************************************************************** */

template<typename T>
dense_vector_ptr<T>
make_raw_dense_vector_ptr()
{
    return std::unique_ptr<dense_vector_t<T>>(new raw_dense_vector_t<T>);
}

/* ************************************************************************** */

template<typename T>
dense_vector_ptr<T>
make_raw_dense_vector_ptr(
    const mat_size_t m,
    const bool on_device,
    T * dense_val)
{
    return std::unique_ptr<dense_vector_t<T>>(new raw_dense_vector_t<T>(m,
        on_device, dense_val));
}

/* ************************************************************************** */

template<typename T>
dense_vector_ptr<T>
make_managed_dense_vector_ptr(
    const mat_size_t m,
    const bool on_device)
{
    return std::unique_ptr<dense_vector_t<T>>(new managed_dense_vector_t<T>(m,
        on_device));
}

/* ************************************************************************** */

template<typename T>
dense_vector_ptr<T>
make_managed_dense_vector_ptr(
    const bool on_device)
{
    return std::unique_ptr<dense_vector_t<T>>(new
        managed_dense_vector_t<T>(on_device));
}

/**
 * *****************************************************************************
 * **************************** COL_MAJOR_MATRIX_T *****************************
 * *****************************************************************************
 */

template<typename T>
col_major_matrix_t<T>::
col_major_matrix_t(
    const mat_size_t _m,
    const mat_size_t _n,
    const bool _on_device)
: m(_m),
  n(_n),
  on_device(_on_device),
  dense_val(nullptr),
  _managed(true)
{
    _alloc(_m, _n);
}

/* ************************************************************************** */

template<typename T>
col_major_matrix_t<T>::
col_major_matrix_t(
    const bool _on_device)
: m(0),
  n(0),
  on_device(_on_device),
  dense_val(nullptr),
  _managed(true)
{
}

/* ************************************************************************** */

template<typename T>
col_major_matrix_t<T>::
col_major_matrix_t(
    const mat_size_t _m,
    const mat_size_t _n,
    T * _dense_val,
    const bool _on_device)
: m(_m),
  n(_n),
  on_device(_on_device),
  dense_val(_dense_val),
  _managed(false)
{

}

/* ************************************************************************** */

template<typename T>
col_major_matrix_t<T>::
~col_major_matrix_t()
{
    _free();
}

/* ************************************************************************** */

template<typename T>
void
col_major_matrix_t<T>::
operator=(
    const col_major_matrix_t<T> * mat)
{
    cudaMemcpyKind copy_direction = cudaMemcpyHostToDevice;
    if(mat->on_device && on_device)
        copy_direction = cudaMemcpyDeviceToDevice;
    if(mat->on_device && !on_device)
        copy_direction = cudaMemcpyDeviceToHost;
    if(!mat->on_device && !on_device)
        copy_direction = cudaMemcpyHostToHost;

    /* allocate memory (if necessary) */
    _alloc(mat->m, mat->n);

    /* copy meta data */
    m = mat->m;
    n = mat->n;

    /* copy data */
    if(mat->dense_val != nullptr)
        CHECK_CUDA(cudaMemcpy(dense_val, mat->dense_val, (m * n) *
            sizeof(T), copy_direction));
}

/* ************************************************************************** */

template<typename T>
void
col_major_matrix_t<T>::
print(
    const char * s)
const
{
    T * print_val = dense_val;

    if(on_device)
    {
        print_val = new T[m * n];
        CHECK_CUDA(cudaMemcpy(print_val, dense_val, m * n * sizeof(T),
            cudaMemcpyDeviceToHost));
    }

    /* print dense matrix */
    printf("Col Major Matrix %s [%ld x %ld]:\n", s, m, n);

    for(mat_int_t i = 0; i < m; ++i)
    {
        printf("\t");
        for(mat_int_t j = 0; j < n; ++j)
            printf("%g ", (double) print_val[j * m + i]);
        printf("\n");
    }

    if(on_device)
    {
        delete[] print_val;
    }
}

/* ************************************************************************** */

template<typename T>
dense_vector_ptr<T>
col_major_matrix_t<T>::
col(
    const mat_size_t j)
{
    return make_raw_dense_vector_ptr<T>(m, on_device, &dense_val[m * j]);
}

/* ************************************************************************** */

template<typename T>
T *
col_major_matrix_t<T>::
elem(
    const mat_size_t i,
    const mat_size_t j)
{
    return &dense_val[m * j + i];
}

/* ************************************************************************** */

template<typename T>
T&
col_major_matrix_t<T>::
operator[](
    const mat_size_t i)
{
    if(on_device)
    {
        std::cerr <<
            "Error accessing memory on device with operator[] is not supported"
            << " at" << __FILE__ << ":" << __LINE__ << ", exiting..." <<
            std::endl;
        std::exit(EXIT_FAILURE);
    }
    else
    {
        return dense_val[i];
    }
}

/* ************************************************************************** */

template<typename T>
const T&
col_major_matrix_t<T>::
operator[](
    const mat_size_t i)
const
{
    if(on_device) {
        std::cerr <<
            "Error accessing memory on device with operator[] is not supported"
            << " at" << __FILE__ << ":" << __LINE__ << ", exiting..." <<
            std::endl;
        std::exit(EXIT_FAILURE);
    }
    else
    {
        return dense_val[i];
    }
}

/* ************************************************************************** */

template<typename T>
thrust::device_ptr<T>
col_major_matrix_t<T>::
dense_val_ptr()
const
{
    return thrust::device_pointer_cast(dense_val);
}

/* ************************************************************************** */

template<typename T>
void
col_major_matrix_t<T>::
_alloc(
    const mat_size_t _m,
    const mat_size_t _n)
{
    /* allocate storage */
    if(dense_val == nullptr || m < _m || n < _n)
    {
        /* need more capacity, thus free old memory */
        _free();

        /* use memory manager for allocation */
        _managed = true;

        dense_val = (T *) GlobalMemPool().request_memory(on_device,
            (_m * _n) * sizeof(T), sizeof(T));
    }

    /* update metadata */
    m = _m;
    n = _n;
}

/* ************************************************************************** */

template<typename T>
void
col_major_matrix_t<T>::
_free()
{
    if(m == 0 || n == 0)
        return;

    if(dense_val == nullptr)
        return;

    if(_managed)
        GlobalMemPool().release_memory(on_device, (char *) dense_val);

    if(!_managed && on_device)
        CHECK_CUDA(cudaFree(dense_val));

    if(!_managed && !on_device)
        delete[] dense_val;

    m = 0;
    n = 0;

    dense_val = nullptr;
}

/* ************************************************************************** */

template<typename T>
col_major_matrix_ptr<T>
make_col_major_matrix_ptr(
    const mat_size_t _m,
    const mat_size_t _n,
    const bool _on_device)
{
    return col_major_matrix_ptr<T>(new col_major_matrix_t<T>(_m, _n,
        _on_device));
}

/* ************************************************************************** */

template<typename T>
col_major_matrix_ptr<T>
make_col_major_matrix_ptr(
    const bool _on_device)
{
    return col_major_matrix_ptr<T>(new col_major_matrix_t<T>(_on_device));
}

/* ************************************************************************** */

template<typename T>
col_major_matrix_ptr<T>
make_col_major_matrix_ptr(
    const mat_size_t _m,
    const mat_size_t _n,
    T * _dense_val,
    const bool _on_device)
{
    return col_major_matrix_ptr<T>(new col_major_matrix_t<T>(_m, _n, _dense_val,
        _on_device));
}

/**
 * *****************************************************************************
 * ******************************* CSR_MATRIX_T ********************************
 * *****************************************************************************
 */

template<typename T>
csr_matrix_t<T>::
csr_matrix_t(
    const mat_size_t _m,
    const mat_size_t _n,
    const mat_size_t _nnz,
    const bool _on_device)
: m(_m),
  n(_n),
  nnz(_nnz),
  on_device(_on_device),
  csr_row(nullptr),
  csr_col(nullptr),
  csr_val(nullptr),
  _managed(true)
{
    _alloc(_m, _nnz);
}

/* ************************************************************************** */

template<typename T>
csr_matrix_t<T>::
csr_matrix_t(
    const bool _on_device)
: m(0),
  n(0),
  nnz(0),
  on_device(_on_device),
  csr_row(nullptr),
  csr_col(nullptr),
  csr_val(nullptr),
  _managed(true)
{
}

/* ************************************************************************** */

template<typename T>
csr_matrix_t<T>::
csr_matrix_t(
    const mat_size_t _m,
    const mat_size_t _n,
    const mat_size_t _nnz,
    const mat_int_t * _csr_row,
    const mat_int_t * _csr_col,
    const T * _csr_val,
    const bool _on_device)
: m(_m),
  n(_n),
  nnz(_nnz),
  csr_row((mat_int_t *) _csr_row),
  csr_col((mat_int_t *) _csr_col),
  csr_val((T *) _csr_val),
  on_device(_on_device),
  _managed(false)
{
}

/* ************************************************************************** */

template<typename T>
csr_matrix_t<T>::
~csr_matrix_t()
{
    _free();
}

/* ************************************************************************** */

template<typename T>
void
csr_matrix_t<T>::
operator=(
    const csr_matrix_t<T> * mat)
{
    cudaMemcpyKind copy_direction = cudaMemcpyHostToDevice;
    if(mat->on_device && on_device)
        copy_direction = cudaMemcpyDeviceToDevice;
    if(mat->on_device && !on_device)
        copy_direction = cudaMemcpyDeviceToHost;
    if(!mat->on_device && !on_device)
        copy_direction = cudaMemcpyHostToHost;

    /* allocate memory (if necessary) */
    _alloc(mat->m, mat->nnz);

    /* copy meta data */
    n = mat->n;

    /* copy data */
    if(mat->csr_row != nullptr)
        CHECK_CUDA(cudaMemcpy(csr_row, mat->csr_row, (m + 1) *
            sizeof(mat_int_t), copy_direction));
    if(mat->csr_col != nullptr)
        CHECK_CUDA(cudaMemcpy(csr_col, mat->csr_col, nnz *
            sizeof(mat_int_t), copy_direction));
    if(mat->csr_val != nullptr)
        CHECK_CUDA(cudaMemcpy(csr_val, mat->csr_val, nnz *
            sizeof(T), copy_direction));
}

/* ************************************************************************** */

template<typename T>
void
csr_matrix_t<T>::
print(
    const char *s,
    const bool print_dense)
const
{
    mat_int_t * Ap_h; /* row length table */
    mat_int_t * Ac_h; /* column index table */
    T * Av_h;

    std::cout << s << std::endl;
    std::cout << "m " << m << ", n " << n << ", nnz " << nnz <<
        " (on device " << on_device << ")" << std::endl;
    if ((csr_row == nullptr) || (csr_col == nullptr) || (csr_val == nullptr)) {
        std::cout << "NULL" << std::endl;
        return;
    }

    if (on_device) {
        /* allocate temporary memory */
        Ap_h = new mat_int_t[m + 1];
        Ac_h = new mat_int_t[nnz];
        Av_h = new T[nnz];

        /* copy data to the host */
        CHECK_CUDA(cudaMemcpy(Ap_h, csr_row, (m+1)*sizeof(mat_int_t),
            cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(Ac_h, csr_col, (nnz)*sizeof(mat_int_t),
            cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(Av_h, csr_val, (nnz)*sizeof(T),
            cudaMemcpyDeviceToHost));
    }
    else{
        /* set pointers */
        Ap_h = csr_row;
        Ac_h = csr_col;
        Av_h = csr_val;
    }

    if(print_dense)
    {
        T * dense = new T[m * n];
        memset(dense, 0, m * n * sizeof(T));

        for(int i = 0; i < m; i++) {
            for (int j = Ap_h[i]; j < Ap_h[i+1]; j++) {
                dense[i * n + Ac_h[j]] = Av_h[j];
            }
        }

        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                std::cout << dense[i * n + j] << "\t";
            }
            std::cout << std::endl;
        }

        delete[] dense;
    }
    else
    {
        for(int i=0; i<m; i++) {
            for (int j=Ap_h[i]; j<Ap_h[i+1]; j++) {
                std::cout << i << ", " << Ac_h[j] << ", " << Av_h[j]
                    << std::endl;
            }
        }
    }

    /* free temporary memory */
    if (on_device) {
        delete[] Ap_h;
        delete[] Ac_h;
        delete[] Av_h;
    }
}

/* ************************************************************************** */

template<typename T>
thrust::device_ptr<mat_int_t>
csr_matrix_t<T>::
csr_row_ptr()
const
{
    return thrust::device_pointer_cast(csr_row);
}

/* ************************************************************************** */

template<typename T>
thrust::device_ptr<mat_int_t>
csr_matrix_t<T>::
csr_col_ptr()
const
{
    return thrust::device_pointer_cast(csr_col);
}

/* ************************************************************************** */

template<typename T>
thrust::device_ptr<T>
csr_matrix_t<T>::
csr_val_ptr()
const
{
    return thrust::device_pointer_cast(csr_val);
}

/* ************************************************************************** */

template<typename T>
cusparseMatDescr_t
csr_matrix_t<T>::
get_description()
const
{
    cusparseMatDescr_t A_desc;
    cusparseCreateMatDescr(&A_desc);
    cusparseSetMatType(A_desc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatDiagType(A_desc, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseSetMatIndexBase(A_desc, CUSPARSE_INDEX_BASE_ZERO);

    return A_desc;
}

/* ************************************************************************** */

template<typename T>
void
csr_matrix_t<T>::
_alloc(
    const mat_size_t _m,
    const mat_size_t _nnz)
{
    if(csr_val == nullptr || nnz < _nnz || m < _m)
    {
        /* need more capacity, hence request from memory manager and free old */
        _free();

        /* turn into managed object */
        _managed = true;

        csr_row = (mat_int_t *) GlobalMemPool().request_memory(on_device,
            (_m + 1) * sizeof(mat_int_t), sizeof(mat_int_t));
        csr_col = (mat_int_t *) GlobalMemPool().request_memory(on_device,
            _nnz * sizeof(mat_int_t), sizeof(mat_int_t));
        csr_val = (T *) GlobalMemPool().request_memory(on_device,
            _nnz * sizeof(T), sizeof(T));
    }

    m = _m;
    nnz = _nnz;
}

/* ************************************************************************** */

template<typename T>
void
csr_matrix_t<T>::
_free()
{
    if(nnz == 0)
        return;

    if(csr_val == nullptr)
        return;

    if(_managed)
    {
        GlobalMemPool().release_memory(on_device, (const char *) csr_row);
        GlobalMemPool().release_memory(on_device, (const char *) csr_col);
        GlobalMemPool().release_memory(on_device, (const char *) csr_val);
    }

    if(!_managed && on_device)
    {
        CHECK_CUDA(cudaFree(csr_row));
        CHECK_CUDA(cudaFree(csr_col));
        CHECK_CUDA(cudaFree(csr_val));
    }

    if(!_managed && !on_device)
    {
        delete[] csr_row;
        delete[] csr_col;
        delete[] csr_val;
    }

    m = 0;
    n = 0;
    nnz = 0;

    csr_row = nullptr;
    csr_col = nullptr;
    csr_val = nullptr;
}

/* ************************************************************************** */

template<typename T>
csr_matrix_ptr<T>
make_csr_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const bool on_device)
{
    return std::unique_ptr<csr_matrix_t<T>>(new csr_matrix_t<T>(
        m, n, nnz, on_device));
}

/* ************************************************************************** */

template<typename T>
csr_matrix_ptr<T>
make_csr_matrix_ptr(
    const bool on_device)
{
    return std::unique_ptr<csr_matrix_t<T>>(new csr_matrix_t<T>(on_device));
}

/* ************************************************************************** */

template<typename T>
csr_matrix_ptr<T>
make_csr_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const mat_int_t * csr_row,
    const mat_int_t * csr_col,
    const T * csr_val,
    const bool on_device)
{
    return std::unique_ptr<csr_matrix_t<T>>(new csr_matrix_t<T>(m, n, nnz,
        csr_row, csr_col, csr_val, on_device));
}

/**
 * *****************************************************************************
 * ******************************* COO_MATRIX_T ********************************
 * *****************************************************************************
 */

template<typename T>
coo_matrix_t<T>::
coo_matrix_t(
    const mat_size_t _m,
    const mat_size_t _n,
    const mat_size_t _nnz,
    const bool _on_device)
: m(_m),
  n(_n),
  nnz(_nnz),
  on_device(_on_device),
  coo_row(nullptr),
  coo_col(nullptr),
  coo_val(nullptr),
  _managed(true)
{
    _alloc(_nnz);
}

/* ************************************************************************** */

template<typename T>
coo_matrix_t<T>::
coo_matrix_t(
    const bool _on_device)
: m(0),
  n(0),
  nnz(0),
  on_device(_on_device),
  coo_row(nullptr),
  coo_col(nullptr),
  coo_val(nullptr),
  _managed(true)
{

}

/* ************************************************************************** */

template<typename T>
coo_matrix_t<T>::
coo_matrix_t(
    const mat_size_t _m,
    const mat_size_t _n,
    const mat_size_t _nnz,
    const mat_int_t * _coo_row,
    const mat_int_t * _coo_col,
    const T * _coo_val,
    const bool _on_device)
: m(_m),
  n(_n),
  nnz(_nnz),
  coo_row((mat_int_t *) _coo_row),
  coo_col((mat_int_t *) _coo_col),
  coo_val((T *) _coo_val),
  on_device(_on_device),
  _managed(false)
{

}

/* ************************************************************************** */

template<typename T>
coo_matrix_t<T>::
~coo_matrix_t()
{
   _free();
}

/* ************************************************************************** */

template<typename T>
void
coo_matrix_t<T>::
operator=(
    const coo_matrix_t<T> * mat)
{
    cudaMemcpyKind copy_direction = cudaMemcpyHostToDevice;
    if(mat->on_device && on_device)
        copy_direction = cudaMemcpyDeviceToDevice;
    if(mat->on_device && !on_device)
        copy_direction = cudaMemcpyDeviceToHost;
    if(!mat->on_device && !on_device)
        copy_direction = cudaMemcpyHostToHost;

    /* (re-)allocate memory if necessary */
    _alloc(mat->nnz);

    /* copy meta data */
    m = mat->m;
    n = mat->n;

    /* copy data */
    if(mat->coo_row != nullptr)
        CHECK_CUDA(cudaMemcpy(coo_row, mat->coo_row, nnz *
            sizeof(mat_int_t), copy_direction));
    if(mat->coo_col != nullptr)
        CHECK_CUDA(cudaMemcpy(coo_col, mat->coo_col, nnz *
            sizeof(mat_int_t), copy_direction));
    if(mat->coo_val != nullptr)
        CHECK_CUDA(cudaMemcpy(coo_val, mat->coo_val, nnz *
            sizeof(T), copy_direction));
}

/* ************************************************************************** */

template<typename T>
void
coo_matrix_t<T>::
print(
    const char *s,
    const bool print_dense)
const
{
    mat_int_t * Ar_h; /* row index table */
    mat_int_t * Ac_h; /* col index table */
    T * Av_h;

    std::cout << s << std::endl;
    std::cout << "m " << m << ", n " << n << ", nnz " << nnz <<
        " (on device " << on_device << ")" << std::endl;
    if ((coo_row == nullptr) || (coo_col == nullptr) || (coo_val == nullptr)) {
        std::cout << "NULL" << std::endl;
        return;
    }

    if (on_device) {
        /* allocate temporary memory */
        Ar_h = new mat_int_t[nnz];
        Ac_h = new mat_int_t[nnz];
        Av_h = new T[nnz];

        /* copy data to the host */
        CHECK_CUDA(cudaMemcpy(Ar_h, coo_row, (nnz)*sizeof(Ar_h[0]),
            cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(Ac_h, coo_col, (nnz)*sizeof(Ac_h[0]),
            cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(Av_h, coo_val, (nnz)*sizeof(Av_h[0]),
            cudaMemcpyDeviceToHost));
    }
    else{
        /* set pointers */
        Ar_h = coo_row;
        Ac_h = coo_col;
        Av_h = coo_val;
    }

    if(print_dense)
    {
        T * dense = new T[m * n];
        memset(dense, 0, m * n * sizeof(T));

        for(int i = 0; i < nnz; i++) {
            dense[Ar_h[i] * n + Ac_h[i]] = Av_h[i];
        }

        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                std::cout << dense[i * n + j] << "\t";
            }
            std::cout << std::endl;
        }

        delete[] dense;
    }
    else
    {
        for(int i = 0; i < nnz; i++) {
            std::cout << Ar_h[i] << ", " << Ac_h[i] << ", " <<
                Av_h[i] << std::endl;
        }
    }

    /* free temporary memory */
    if (on_device) {
        delete[] Ar_h;
        delete[] Ac_h;
        delete[] Av_h;
    }
}

/* ************************************************************************** */

template<typename T>
thrust::device_ptr<mat_int_t>
coo_matrix_t<T>::
coo_row_ptr()
const
{
    return thrust::device_pointer_cast(coo_row);
}

/* ************************************************************************** */

template<typename T>
thrust::device_ptr<mat_int_t>
coo_matrix_t<T>::
coo_col_ptr()
const
{
    return thrust::device_pointer_cast(coo_col);
}

/* ************************************************************************** */

template<typename T>
thrust::device_ptr<T>
coo_matrix_t<T>::
coo_val_ptr()
const
{
    return thrust::device_pointer_cast(coo_val);
}

/* ************************************************************************** */

template<typename T>
void
coo_matrix_t<T>::
_alloc(
    const mat_size_t _nnz)
{
    if(coo_val == nullptr || nnz < _nnz)
    {
        _free();

        if(_managed)
        {
            coo_row = (mat_int_t *) GlobalMemPool().request_memory(on_device,
                _nnz * sizeof(mat_int_t), sizeof(mat_int_t));
            coo_col = (mat_int_t *) GlobalMemPool().request_memory(on_device,
                _nnz * sizeof(mat_int_t), sizeof(mat_int_t));
            coo_val = (T *) GlobalMemPool().request_memory(on_device,
                _nnz * sizeof(T), sizeof(T));
        }

        if(!_managed && on_device)
        {
            CHECK_CUDA(cudaMalloc((void **) &coo_row,
                _nnz * sizeof(mat_int_t)));
            CHECK_CUDA(cudaMalloc((void **) &coo_col,
                _nnz * sizeof(mat_int_t)));
            CHECK_CUDA(cudaMalloc((void **) &coo_val,
                _nnz * sizeof(T)));
        }

        if(!_managed && !on_device)
        {
            coo_row = new mat_int_t[_nnz];
            coo_col = new mat_int_t[_nnz];
            coo_val = new T[_nnz];
        }
    }

    nnz = _nnz;
}

/* ************************************************************************** */

template<typename T>
void
coo_matrix_t<T>::
_free()
{
    if(nnz == 0)
        return;

    if(coo_val == nullptr)
        return;

    if(_managed)
    {
        GlobalMemPool().release_memory(on_device, (const char *) coo_row);
        GlobalMemPool().release_memory(on_device, (const char *) coo_col);
        GlobalMemPool().release_memory(on_device, (const char *) coo_val);
    }

    if(!_managed && on_device)
    {
        CHECK_CUDA(cudaFree(coo_row));
        CHECK_CUDA(cudaFree(coo_col));
        CHECK_CUDA(cudaFree(coo_val));
    }

    if(!_managed && !on_device)
    {
        delete[] coo_row;
        delete[] coo_col;
        delete[] coo_val;
    }

    m = 0;
    n = 0;
    nnz = 0;

    coo_row = nullptr;
    coo_col = nullptr;
    coo_val = nullptr;
}

/* ************************************************************************** */

template<typename T>
coo_matrix_ptr<T>
make_coo_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const bool on_device)
{
    return std::unique_ptr<coo_matrix_t<T>>(new coo_matrix_t<T>(m, n, nnz,
        on_device));
}

/* ************************************************************************** */

template<typename T>
coo_matrix_ptr<T>
make_coo_matrix_ptr(
    const bool on_device)
{
    return std::unique_ptr<coo_matrix_t<T>>(new coo_matrix_t<T>(on_device));
}

/* ************************************************************************** */

template<typename T>
coo_matrix_ptr<T>
make_coo_matrix_ptr(
    const mat_size_t m,
    const mat_size_t n,
    const mat_size_t nnz,
    const mat_int_t * coo_row,
    const mat_int_t * coo_col,
    const T * coo_val,
    const bool on_device)
{
    return std::unique_ptr<coo_matrix_t<T>>(new coo_matrix_t<T>(m, n, nnz,
        coo_row, coo_col, coo_val, on_device));
}

NS_CULIP_END
