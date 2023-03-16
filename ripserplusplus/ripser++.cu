/*
 Ripser++: accelerated Vietoris-Rips persistence barcodes computation with GPU

 MIT License

 Copyright (c) 2019, 2020 Simon Zhang, Mengbai Xiao, Hao Wang

 Python Bindings Contributors:
 Birkan Gokbag
 Ryan DeMilt

 Copyright (c) 2015-2019 Ripser codebase, written by Ulrich Bauer

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 You are under no obligation whatsoever to provide any bug fixes, patches, or
 upgrades to the features, functionality or performance of the source code
 ("Enhancements") to anyone; however, if you choose to make your Enhancements
 available either publicly, or directly to the author of this software, without
 imposing a separate written license agreement for such Enhancements, then you
 hereby grant the following license: a non-exclusive, royalty-free perpetual
 license to install, use, modify, prepare derivative works, incorporate into
 other computer software, distribute, and sublicense such enhancements or
 derivative works thereof, in binary and source code form.

*/

#define CUDACHECK(cmd) do {\
    cudaError_t e= cmd;\
    if( e != cudaSuccess ) {\
        printf("Failed: Cuda error %s:%d '%s'\n",\
        __FILE__,__LINE__,cudaGetErrorString(e));\
    exit(EXIT_FAILURE);\
    }\
    } while(0)

//#define INDICATE_PROGRESS//DO NOT UNCOMMENT THIS IF YOU WANT TO LOG PROFILING NUMBERS FROM stderr TO FILE
//#define PRINT_PERSISTENCE_PAIRS//print out all persistence paris to stdout
//#define CPUONLY_ASSEMBLE_REDUCTION_MATRIX//do full matrix reduction on CPU with the sparse coefficient matrix V
#define ASSEMBLE_REDUCTION_SUBMATRIX//do submatrix reduction with the sparse coefficient submatrix of V
//#define PROFILING
//#define COUNTING
#define USE_PHASHMAP//www.github.com/greg7mdp/parallel-hashmap
#define PYTHON_BARCODE_COLLECTION
#ifndef USE_PHASHMAP
#define USE_GOOGLE_HASHMAP
#endif

//#define CPUONLY_SPARSE_HASHMAP//WARNING: MAY NEED LOWER GCC VERSION TO RUN, TESTED ON: NVCC VERSION 9.2 WITH GCC VERSIONS >=5.3.0 AND <=7.3.0

#define MIN_INT64 (-9223372036854775807-1)
#define MAX_INT64 (9223372036854775807)
#define MAX_FLOAT (340282346638528859811704183484516925440.000000)


#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <profiling/stopwatch.h>
#include <sparsehash/dense_hash_map>
#include <phmap_interface/phmap_interface.h>

#include <omp.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#ifdef CPUONLY_SPARSE_HASHMAP
#include <sparsehash/sparse_hash_map>
template <class Key, class T> class hash_map : public google::sparse_hash_map<Key, T> {
public:
    explicit hash_map() : google::sparse_hash_map<Key, T>() {
        }
    inline void reserve(size_t hint) { this->resize(hint); }
};
#endif

#ifndef CPUONLY_SPARSE_HASHMAP
template <class Key, class T> class hash_map : public google::dense_hash_map<Key, T> {
public:
    explicit hash_map() : google::dense_hash_map<Key, T>() {
        this->set_empty_key(-1);
    }
    inline void reserve(size_t hint) { this->resize(hint); }
};
#endif

#ifdef INDICATE_PROGRESS
static const std::chrono::milliseconds time_step(40);
static const std::string clear_line("\r\033[K");
#endif

typedef float value_t;
typedef int64_t index_t;

struct diameter_index_t_struct{
    value_t diameter;
    index_t index;
};

struct index_diameter_t_struct{
    index_t index;
    value_t diameter;
};

struct lowerindex_lowerdiameter_index_t_struct_compare{
    __host__ __device__ bool operator() (struct index_diameter_t_struct a, struct index_diameter_t_struct b){
        return a.index!=b.index ? a.index<b.index : a.diameter<b.diameter;
    }
};

struct greaterdiam_lowerindex_diameter_index_t_struct_compare {
    __host__ __device__ bool operator() (struct diameter_index_t_struct a, struct diameter_index_t_struct b){
        return a.diameter!=b.diameter ? a.diameter>b.diameter : a.index<b.index;
    }
};
struct greaterdiam_lowerindex_diameter_index_t_struct_compare_reverse {
    __host__ __device__ bool operator() (struct diameter_index_t_struct a, struct diameter_index_t_struct b){
        return a.diameter!=b.diameter ? a.diameter<b.diameter : a.index>b.index;
    }
};

struct lowerindex_lowerdiam_diameter_index_t_struct_compare{
    __host__ __device__ bool operator() (struct diameter_index_t_struct a, struct diameter_index_t_struct b){
        return a.index!=b.index ? a.index<b.index : a.diameter<b.diameter;
    }
};

struct index_t_pair_struct{//data type for a pivot in the coboundary matrix: (row,column)
    index_t row_cidx;
    index_t column_idx;
};

typedef struct {
    value_t birth;
    value_t death;
} birth_death_coordinate;
typedef struct{
    index_t num_barcodes;
    birth_death_coordinate* barcodes;
} set_of_barcodes;
typedef struct{
    int num_dimensions;
    set_of_barcodes* all_barcodes;
} ripser_plusplus_result;

ripser_plusplus_result res;


std::vector<std::vector<birth_death_coordinate>> list_of_barcodes = std::vector<std::vector<birth_death_coordinate>>();

struct row_cidx_column_idx_struct_compare{
    __host__ __device__ bool operator()(struct index_t_pair_struct a, struct index_t_pair_struct b){
        //return a.row_cidx!=b.row_cidx ? a.row_cidx<b.row_cidx : a.column_idx<b.column_idx;//the second condition should never happen if sorting pivot pairs since pivots do not conflict on rows or columns
        return a.row_cidx<b.row_cidx || (a.row_cidx==b.row_cidx && a.column_idx<b.column_idx);
    }
};

__host__ __device__ value_t hd_max(value_t a, value_t b){
    return a>b?a:b;
}

void check_overflow(index_t i){
    if(i<0){
        throw std::overflow_error("simplex index "+std::to_string((uint64_t)i)+" in filtration is overflowing past 64 bits signed integer");
    }
}

//assume i>j (lower triangular with i indexing rows and j indexing columns
#define LOWER_DISTANCE_INDEX(i,j,n) (((i)*((i)-1)/2)+(j))
class binomial_coeff_table {
    index_t num_n;
    index_t max_tuple_length;

#define BINOM_TRANSPOSE(i,j) ((j)*(num_n)+(i))
#define BINOM(i,j) ((i)*(max_tuple_length)+(j))
public:
    index_t* binoms;

    binomial_coeff_table(index_t n, index_t k) {
        binoms= (index_t*)malloc(sizeof(index_t)*(n+1)*(k+1));
        if(binoms==NULL){
            std::cerr<<"malloc for binoms failed"<<std::endl;
            exit(1);
        }
        num_n= n+1;
        max_tuple_length= k+1;
        memset(binoms, 0, sizeof(index_t)*num_n*max_tuple_length);

        for (index_t i= 0; i <= n; i++) {
            for (index_t j= 0; j <= std::min(i, k); j++){
                if (j == 0 || j == i) {
                    binoms[BINOM_TRANSPOSE(i,j)]= 1;
                } else {
                    binoms[BINOM_TRANSPOSE(i,j)]= binoms[BINOM_TRANSPOSE(i-1,j-1)]+binoms[BINOM_TRANSPOSE(i-1,j)];
                }
            }
            check_overflow(binoms[BINOM_TRANSPOSE(i,std::min(i>>1,k))]);
        }
    }

    index_t get_num_n() const{
        return num_n;
    }

    index_t get_max_tuple_length() const{
        return max_tuple_length;
    }

    __host__ __device__ index_t operator()(index_t n, index_t k) const{
        assert(n<num_n && k<max_tuple_length);
        return binoms[BINOM_TRANSPOSE(n,k)];
    }
    ~binomial_coeff_table(){
        free(binoms);
    }
};

typedef std::pair<value_t, index_t> diameter_index_t;
value_t get_diameter(const diameter_index_t& i) { return i.first; }
index_t get_index(const diameter_index_t& i) { return i.second; }

template <typename Entry> struct greater_diameter_or_smaller_index {
    bool operator()(const Entry& a, const Entry& b) {
        return (get_diameter(a) > get_diameter(b)) ||
               ((get_diameter(a) == get_diameter(b)) && (get_index(a) < get_index(b)));
    }
};


struct CSR_distance_matrix{
    index_t capacity;
    value_t* entries;
    index_t* offsets;
    index_t* col_indices;
    index_t n;
    index_t num_edges;
    index_t num_entries;
public:
    CSR_distance_matrix(){}//avoid calling malloc in constructor for GPU side

    index_t size(){return n;}
    ~CSR_distance_matrix(){
        free(entries);
        free(offsets);
        free(col_indices);
    }
};

class compressed_lower_distance_matrix {
public:
    std::vector<value_t> distances;
    std::vector<value_t*> rows;

    void init_rows() {
        value_t* pointer= &distances[0];
        for (index_t i= 1; i < size(); ++i) {
            rows[i]= pointer;
            pointer+= i;
        }
    }

    compressed_lower_distance_matrix(std::vector<value_t>&& _distances)
            : distances(std::move(_distances)), rows((1 + std::sqrt(1 + 8 * distances.size())) / 2) {
        assert(distances.size() == size() * (size() - 1) / 2);
        init_rows();
    }

    template <typename DistanceMatrix>
    compressed_lower_distance_matrix(const DistanceMatrix& mat)
            : distances(mat.size() * (mat.size() - 1) / 2), rows(mat.size()) {
        init_rows();

        for (index_t i= 1; i < size(); ++i)
            for (index_t j= 0; j < i; ++j) rows[i][j]= mat(i, j);
    }

    value_t operator()(const index_t i, const index_t j) const {
        return i == j ? 0 : i < j ? rows[j][i] : rows[i][j];
    }

    size_t size() const { return rows.size(); }

};

struct sparse_distance_matrix {
    std::vector<std::vector<index_diameter_t_struct>> neighbors;

    index_t num_entries;

    mutable std::vector<std::vector<index_diameter_t_struct>::const_reverse_iterator> neighbor_it;
    mutable std::vector<std::vector<index_diameter_t_struct>::const_reverse_iterator> neighbor_end;

    sparse_distance_matrix(std::vector<std::vector<index_diameter_t_struct>>&& _neighbors,
                           index_t _num_edges)
            : neighbors(std::move(_neighbors)), num_entries(_num_edges*2) {}

    template <typename DistanceMatrix>
    sparse_distance_matrix(const DistanceMatrix& mat, const value_t threshold)
            : neighbors(mat.size()), num_entries(0) {
#ifdef COUNTING
        std::cerr << "threshold: " << threshold << std::endl;
#endif
        for (index_t i= 0; i < size(); ++i) {
            for (index_t j= 0; j < size(); ++j) {
                if (i != j && mat(i, j) <= threshold) {
                    ++num_entries;
                    neighbors[i].push_back({j, mat(i, j)});
                }
            }
        }
    }
    size_t size() const { return neighbors.size(); }

private:
    //this should only be called from CPU side
    void append_sparse(CSR_distance_matrix& dist, value_t e, index_t j) {

        if (dist.capacity == 0) {
            dist.entries= (value_t *) malloc(sizeof(value_t) * size() * 10);
            if(dist.entries==NULL){
                std::cerr<<"entries could not be malloced"<<std::endl;
                exit(1);
            }
            dist.col_indices= (index_t *) malloc(sizeof(index_t) * size() * 10);
            if(dist.col_indices==NULL){
                std::cerr<<"col_indices could not be malloced"<<std::endl;
                exit(1);
            }
            dist.capacity= size() * 10;
        }

        if (dist.num_entries >= dist.capacity) {
            dist.capacity*= 2;
            dist.entries= (value_t *) realloc(dist.entries, sizeof(value_t) * dist.capacity);
            if(dist.entries==NULL){
                std::cerr<<"col_indices could not be realloced with double memory"<<std::endl;
                exit(1);
            }
            dist.col_indices= (index_t *) realloc(dist.col_indices, sizeof(index_t) * dist.capacity);
            if(dist.col_indices==NULL){
                std::cerr<<"col_indices could not be realloced with double memory"<<std::endl;
                exit(1);
            }
        }
        dist.entries[dist.num_entries]= e;
        dist.col_indices[dist.num_entries++]= j;
    }

    //this should only be called on CPU side
    void update_offsets(CSR_distance_matrix& dist, index_t row_index, index_t offset_increment){
        if(row_index==0){
            dist.offsets[0]= 0;
        }
        dist.offsets[row_index+1]= dist.offsets[row_index]+offset_increment;
    }

public:

    CSR_distance_matrix toCSR(){
        CSR_distance_matrix dist;
        dist.n= size();
        dist.num_entries= 0;
        dist.capacity= num_entries;//this sets the matrix to exactly num_entries memory allocation
        dist.offsets= (index_t*) malloc(sizeof(index_t)*(size()+1));
        if(dist.offsets==NULL){
            std::cerr<<"malloc for offsets failed"<<std::endl;
            exit(1);
        }
        dist.col_indices= (index_t*) malloc(sizeof(index_t)*dist.capacity);
        if(dist.col_indices==NULL){
            std::cerr<<"malloc for col_indices failed"<<std::endl;
            exit(1);
        }
        dist.entries= (value_t*) malloc(sizeof(value_t)*dist.capacity);
        if(dist.entries==NULL){
            std::cerr<<"malloc for entries failed"<<std::endl;
            exit(1);
        }
        for(index_t i= 0; i<size(); i++){
            index_t nnz_inrow= 0;
            for(index_t j=0; j<neighbors[i].size(); j++){
                append_sparse(dist, neighbors[i][j].diameter, neighbors[i][j].index);
                nnz_inrow++;
            }
            update_offsets(dist, i, nnz_inrow);
        }
        dist.num_edges= num_entries/2;
        return dist;
    }

};

class euclidean_distance_matrix {
public:
    std::vector<std::vector<value_t>> points;

    euclidean_distance_matrix(std::vector<std::vector<value_t>>&& _points)
            : points(std::move(_points)) {
        for (auto p : points) { assert(p.size() == points.front().size()); }
    }

    value_t operator()(const index_t i, const index_t j) const {
        assert(i < points.size());
        assert(j < points.size());
        return std::sqrt(std::inner_product(
                points[i].begin(), points[i].end(), points[j].begin(), value_t(), std::plus<value_t>(),
                [](value_t u, value_t v) { return (u - v) * (u - v); }));
    }

    size_t size() const { return points.size(); }
};

class union_find {
    std::vector<index_t> parent;
    std::vector<uint8_t> rank;

public:
    union_find(index_t n) : parent(n), rank(n, 0) {
        for (index_t i= 0; i < n; ++i) parent[i]= i;
    }

    index_t find(index_t x) {
        index_t y= x, z;
        while ((z= parent[y]) != y) y= z;
        while ((z= parent[x]) != y) {
            parent[x]= y;
            x= z;
        }
        return z;
    }
    void link(index_t x, index_t y) {
        if ((x= find(x)) == (y= find(y))) return;
        if (rank[x] > rank[y])
            parent[y]= x;
        else {
            parent[x]= y;
            if (rank[x] == rank[y]) ++rank[y];
        }
    }
};

template <typename Heap> struct diameter_index_t_struct pop_pivot(Heap& column) {
    if(column.empty()) {
        return {0,-1};
    }

    auto pivot= column.top();
    column.pop();
    while(!column.empty() && (column.top()).index == pivot.index) {
        column.pop();
        if (column.empty()) {
            return {0,-1};
        }
        else {
            pivot= column.top();
            column.pop();
        }
    }
    return pivot;
}

template <typename Heap> struct diameter_index_t_struct get_pivot(Heap& column) {
    struct diameter_index_t_struct result= pop_pivot(column);
    if (result.index != -1) column.push(result);
    return result;
}

template <typename T> T begin(std::pair<T, T>& p) { return p.first; }
template <typename T> T end(std::pair<T, T>& p) { return p.second; }
template <typename ValueType> class compressed_sparse_matrix {
    std::vector<size_t> bounds;
    std::vector<ValueType> entries;

    typedef typename std::vector<ValueType>::iterator iterator;
    typedef std::pair<iterator, iterator> iterator_pair;

public:
    size_t size() const { return bounds.size(); }

    iterator_pair subrange(const index_t index) {
        return {entries.begin() + (index == 0 ? 0 : bounds[index - 1]),
                entries.begin() + bounds[index]};
    }

    void append_column() { bounds.push_back(entries.size()); }

    void push_back(const ValueType e) {
        assert(0 < size());
        entries.push_back(e);
        ++bounds.back();
    }
};

template <typename ValueType> class compressed_sparse_submatrix {
    std::vector<size_t> sub_bounds;//the 0-based indices for
    std::vector<ValueType> entries;

    typedef typename std::vector<ValueType>::iterator iterator;
    typedef std::pair<iterator, iterator> iterator_pair;

public:
    size_t size() const { return sub_bounds.size(); }

    //assume we are given a "subindex" for the submatrix
    //allows iteration from sub_bounds[index_to_subindex[index]] to sub_bounds[index_to_subindex[index+1]]-1
    iterator_pair subrange(const index_t subindex) {
        return {entries.begin() + (subindex == 0 ? 0 : sub_bounds[subindex - 1]),
                entries.begin() + sub_bounds[subindex]};
    }

    void append_column() { sub_bounds.push_back(entries.size()); }

    void push_back(const ValueType e) {
        assert(0 < size());
        entries.push_back(e);
        ++sub_bounds.back();
    }
};

template <class Predicate> index_t upper_bound(index_t top, Predicate pred) {
    if (!pred(top)) {
        index_t count= top;
        while (count > 0) {
            index_t step= count >> 1;
            if (!pred(top - step)) {
                top-= step + 1;
                count-= step + 1;
            } else
                count= step;
        }
    }
    return top;
}

__global__ void gpu_insert_pivots_kernel(struct index_t_pair_struct* d_pivot_array, index_t* d_lowest_one_of_apparent_pair, index_t* d_pivot_column_index_OR_nonapparent_cols, index_t num_columns_to_reduce, index_t* d_num_nonapparent){
    index_t tid= (index_t)threadIdx.x+(index_t)blockIdx.x*(index_t)blockDim.x;
    index_t stride= (index_t)blockDim.x*(index_t)gridDim.x;

    //index_t* d_pivot_column_index_OR_nonapparent_cols is being used as d_nonapparent_cols
    for(; tid<num_columns_to_reduce; tid+= stride) {
        int keep_tid= d_lowest_one_of_apparent_pair[tid] == -1;
        if (!keep_tid) {//insert pivot
            d_pivot_array[tid].row_cidx= d_lowest_one_of_apparent_pair[tid];
            d_pivot_array[tid].column_idx= tid;
        }else {//keep track of nonapparent columns
            d_pivot_array[tid].row_cidx= MAX_INT64;
            d_pivot_array[tid].column_idx= MAX_INT64;

            //do standard warp based filtering under the assumption that there are few nonapparent columns
#define FULL_MASK 0xFFFFFFFF
            int lane_id= threadIdx.x % 32;
            int mask= __ballot_sync(FULL_MASK, keep_tid);
            int leader= __ffs(mask) - 1;
            int base;
            if (lane_id == leader)
                base= atomicAdd((unsigned long long int *) d_num_nonapparent, __popc(mask));
            base= __shfl_sync(mask, base, leader);
            int pos= base + __popc(mask & ((1 << lane_id) - 1));

            if (keep_tid) {
                d_pivot_column_index_OR_nonapparent_cols[pos]= tid;//being used as d_nonapparent_cols
            }
        }
    }
}
__global__ void populate_edges_warpfiltering(struct diameter_index_t_struct* d_columns_to_reduce, value_t threshold, value_t* d_distance_matrix, index_t max_num_simplices, index_t num_points, binomial_coeff_table* d_binomial_coeff, index_t* d_num_columns_to_reduce){
    index_t tid= (index_t)threadIdx.x+(index_t)blockIdx.x*(index_t)blockDim.x;
    index_t stride= (index_t)blockDim.x*(index_t)gridDim.x;

    __shared__ index_t shared_vertices[256][3];//eliminate bank conflicts (that's what the 3 is for)
    for(; tid<max_num_simplices; tid+= stride) {
        index_t offset= 0;
        index_t v= num_points - 1;
        index_t idx= tid;

        for (index_t k= 2; k > 0; --k) {
            if (!((*d_binomial_coeff)(v, k) <= idx)) {
                index_t count= v;
                while (count > 0) {
                    index_t step= count >> 1;
                    if (!((*d_binomial_coeff)(v - step, k) <= idx)) {
                        v-= step + 1;
                        count-= step + 1;//+1 is here to preserve the induction hypothesis (check v=4, k=4)
                    } else
                        count= step;//went too far, need to try a smaller step size to subtract from top
                }
            }

            shared_vertices[threadIdx.x][offset++]= v;

            idx -= (*d_binomial_coeff)(v, k);
        }

        //shared_vertices is always sorted in decreasing order
        value_t diam= d_distance_matrix[LOWER_DISTANCE_INDEX(shared_vertices[threadIdx.x][0], shared_vertices[threadIdx.x][1], num_points)];
#define FULL_MASK 0xFFFFFFFF
        int lane_id= threadIdx.x % 32;
        int keep_tid= diam<=threshold;
        int mask= __ballot_sync(FULL_MASK, keep_tid);
        int leader= __ffs(mask) - 1;
        int base;
        if (lane_id == leader)
            base= atomicAdd((unsigned long long int *)d_num_columns_to_reduce, __popc(mask));
        base= __shfl_sync(mask, base, leader);
        int pos= base + __popc(mask & ((1 << lane_id) - 1));

        if(keep_tid){
            d_columns_to_reduce[pos].diameter= diam;
            d_columns_to_reduce[pos].index= tid;
        }
    }
}

template <typename T> __global__ void populate_edges(T* d_flagarray, struct diameter_index_t_struct* d_columns_to_reduce, value_t threshold, value_t* d_distance_matrix, index_t max_num_simplices, index_t num_points, binomial_coeff_table* d_binomial_coeff){
    index_t tid= (index_t)threadIdx.x+(index_t)blockIdx.x*(index_t)blockDim.x;
    index_t stride= (index_t)blockDim.x*(index_t)gridDim.x;

    __shared__ index_t shared_vertices[256][3];//designed to eliminate bank conflicts (that's what the 3 is for)
    for(; tid<max_num_simplices; tid+= stride) {
        index_t offset= 0;
        index_t v= num_points - 1;
        index_t idx= tid;

        for (index_t k= 2; k > 0; --k) {

            if (!((*d_binomial_coeff)(v, k) <= idx)) {
                index_t count= v;
                while (count > 0) {
                    index_t step= count >> 1;
                    if (!((*d_binomial_coeff)(v - step, k) <= idx)) {
                        v-= step + 1;
                        count-= step + 1;//+1 is here to preserve the induction hypothesis (check v=4, k=4)
                    } else
                        count= step;//went too far, need to try a smaller step size to subtract from top
                }
            }

            shared_vertices[threadIdx.x][offset++]= v;

            idx-= (*d_binomial_coeff)(v, k);
        }
        //shared_vertices is sorted in decreasing order
        value_t diam= d_distance_matrix[LOWER_DISTANCE_INDEX(shared_vertices[threadIdx.x][0], shared_vertices[threadIdx.x][1], num_points)];
        if(diam<=threshold){
            d_columns_to_reduce[tid].diameter= diam;
            d_columns_to_reduce[tid].index= tid;
            d_flagarray[tid]= 1;
        }else{
            d_columns_to_reduce[tid].diameter= MAX_FLOAT;//the sorting is in boundary matrix filtration order
            d_columns_to_reduce[tid].index= MIN_INT64;
            d_flagarray[tid]= 0;
        }
    }
}

__global__ void populate_columns_to_reduce_warpfiltering(struct diameter_index_t_struct* d_columns_to_reduce, index_t* d_num_columns_to_reduce, index_t* d_pivot_column_index, value_t* d_distance_matrix, index_t num_points, index_t max_num_simplices, index_t dim, value_t threshold, binomial_coeff_table* d_binomial_coeff) {
    index_t tid= (index_t)threadIdx.x + (index_t)blockIdx.x * (index_t)blockDim.x;
    index_t stride= (index_t)blockDim.x * (index_t)gridDim.x;

    extern __shared__ index_t shared_vertices[];//a 256x(dim+1) matrix; shared_vertices[threadIdx.x*(dim+1)+j]=the jth vertex for threadIdx.x thread in the thread block
    for (; tid < max_num_simplices; tid+= stride) {
        index_t offset= 0;
        index_t v= num_points - 1;
        index_t idx= tid;

        for (index_t k= dim + 1; k > 0; --k) {

            if (!((*d_binomial_coeff)(v, k) <= idx)) {
                index_t count= v;
                while (count > 0) {
                    index_t step= count >> 1;
                    if (!((*d_binomial_coeff)(v - step, k) <= idx)) {
                        v-= step + 1;
                        count-= step + 1;//+1 is here to preserve the induction hypothesis (check v=4, k=4)
                    } else
                        count= step;//went too far, need to try a smaller step size to subtract from top
                }
            }

            shared_vertices[threadIdx.x * (dim + 1) + offset++]= v;

            idx-= (*d_binomial_coeff)(v, k);
        }

        value_t diam= -MAX_FLOAT;
        for (index_t j= 0; j <= dim; ++j) {
            for (index_t i= 0; i < j; ++i) {
                diam= hd_max(diam, d_distance_matrix[LOWER_DISTANCE_INDEX(shared_vertices[threadIdx.x * (dim + 1) + i], shared_vertices[threadIdx.x *(dim + 1) + j], num_points)]);
            }
        }

#define FULL_MASK 0xFFFFFFFF
        int lane_id= threadIdx.x % 32;
        int keep_tid= d_pivot_column_index[tid] == -1 && diam<=threshold;
        int mask= __ballot_sync(FULL_MASK, keep_tid);
        int leader= __ffs(mask) - 1;
        int base;
        if (lane_id == leader)
            base= atomicAdd((unsigned long long int *)d_num_columns_to_reduce, __popc(mask));
        base= __shfl_sync(mask, base, leader);
        int pos= base + __popc(mask & ((1 << lane_id) - 1));

        if(keep_tid){
            d_columns_to_reduce[pos].diameter= diam;
            d_columns_to_reduce[pos].index= tid;
        }
    }
}

__global__ void populate_sparse_edges_preparingcount(int* d_num, CSR_distance_matrix* d_CSR_distance_matrix, index_t num_points, index_t* d_num_simplices){
    index_t tid= (index_t)threadIdx.x+(index_t)blockIdx.x*(index_t)blockDim.x;
    index_t stride= (index_t)blockDim.x * (index_t)gridDim.x;
    index_t* offsets= d_CSR_distance_matrix->offsets;
    index_t* col_indices= d_CSR_distance_matrix->col_indices;

    for(; tid<num_points; tid+= stride){
        int _num=0;
        index_t col_start= offsets[tid];
        index_t col_end= offsets[tid+1];
        for(index_t entry_idx= col_start; entry_idx<col_end; entry_idx++){
            index_t neighbor_of_tid= col_indices[entry_idx];
            if(tid>neighbor_of_tid)_num++;
        }
        d_num[tid]= _num;
    }
}

__global__ void populate_sparse_edges_prefixsum(struct diameter_index_t_struct* d_simplices, int* d_num, CSR_distance_matrix* d_CSR_distance_matrix, binomial_coeff_table* d_binomial_coeff, index_t num_points, index_t* d_num_simplices){
    index_t tid= (index_t)threadIdx.x+(index_t)blockIdx.x*(index_t)blockDim.x;
    index_t stride= (index_t)blockDim.x * (index_t)gridDim.x;

    value_t* entries= d_CSR_distance_matrix->entries;
    index_t* offsets= d_CSR_distance_matrix->offsets;
    index_t* col_indices= d_CSR_distance_matrix->col_indices;

    for(; tid<num_points; tid+= stride){
        int _pos=0;
        index_t col_start= offsets[tid];
        index_t col_end= offsets[tid+1];
        for(index_t entry_idx= col_start; entry_idx<col_end; entry_idx++){
            index_t neighbor_of_tid= col_indices[entry_idx];
            if(tid>neighbor_of_tid){
                d_simplices[d_num[tid]+_pos].diameter= entries[entry_idx];
                d_simplices[d_num[tid]+_pos++].index= (*d_binomial_coeff)(tid,2) + neighbor_of_tid;
            }
        }
        if(tid==num_points-1){
            *d_num_simplices= d_num[tid]+_pos;
        }
    }
}
__global__ void populate_sparse_simplices_warpfiltering(struct diameter_index_t_struct* d_simplices, index_t* d_num_simplices, struct diameter_index_t_struct* d_columns_to_reduce, index_t* d_num_columns_to_reduce, CSR_distance_matrix* d_CSR_distance_matrix, index_t num_points, index_t dim, value_t threshold, binomial_coeff_table* d_binomial_coeff){
    index_t tid= (index_t)threadIdx.x + (index_t)blockIdx.x * (index_t)blockDim.x;
    index_t stride= (index_t)blockDim.x * (index_t)gridDim.x;

    dim--;//keep dim in terms of the dimension of the simplices
    extern __shared__ index_t shared_vertices[];//a 256x(dim+1) matrix; shared_vertices[threadIdx.x*(dim+1)+j]=the jth vertex for threadIdx.x thread in the thread block

    for (; tid < *d_num_simplices; tid += stride) {
        index_t offset= 0;
        index_t v= num_points - 1;
        index_t idx= d_simplices[tid].index;

        for (index_t k= dim + 1; k > 0; --k) {

            if (!((*d_binomial_coeff)(v, k) <= idx)) {
                index_t count= v;
                while (count > 0) {
                    index_t step= count >> 1;
                    if (!((*d_binomial_coeff)(v - step, k) <= idx)) {
                        v-= step + 1;
                        count-= step + 1;//+1 is here to preserve the induction hypothesis (check v=4, k=4)
                    } else
                        count= step;//went too far, need to try a smaller step size to subtract from top
                }
            }

            shared_vertices[threadIdx.x * (dim + 1) + offset++]= v;

            idx-= (*d_binomial_coeff)(v, k);
        }

        index_t k= dim+1;

        bool next_cofacet= false;
        value_t nbr_diameter= -1;
        index_t nbr_index= -1;
        index_t idx_below= d_simplices[tid].index;
        index_t idx_above= 0;


        index_t base_vertex_index= shared_vertices[threadIdx.x * (dim + 1)]; //shared_vertices[threadIdx.x][0];
        //this gives the entry indices of the right and left ends of the row indexed by base_vertex_index in the CSR distance matrix
        index_t base_vertex_nbr_itr= d_CSR_distance_matrix->offsets[base_vertex_index+1]-1;
        index_t base_vertex_nbr_end= d_CSR_distance_matrix->offsets[base_vertex_index];

        for(; base_vertex_nbr_itr>=base_vertex_nbr_end; base_vertex_nbr_itr--){
            //nbr is the neighboring vertex to the simplex corresponding to this tid
            nbr_diameter= d_CSR_distance_matrix->entries[base_vertex_nbr_itr];
            nbr_index= d_CSR_distance_matrix->col_indices[base_vertex_nbr_itr];

            //there are dim other vertices along with the base_vertex
            for(index_t other_vertex_idx=1; other_vertex_idx<dim+1; other_vertex_idx++){

                index_t other_vertex= shared_vertices[threadIdx.x * (dim + 1) + other_vertex_idx];
                index_t other_vertex_nbr_itr= d_CSR_distance_matrix->offsets[other_vertex+1]-1;
                index_t other_vertex_nbr_end= d_CSR_distance_matrix->offsets[other_vertex];
                index_t other_vertex_nbr_index= d_CSR_distance_matrix->col_indices[other_vertex_nbr_itr];
                while(other_vertex_nbr_index>nbr_index){
                    if(other_vertex_nbr_itr==other_vertex_nbr_end) {
                        next_cofacet= false;
                        goto end_search;
                    }
                    other_vertex_nbr_itr--;
                    other_vertex_nbr_index= d_CSR_distance_matrix->col_indices[other_vertex_nbr_itr];
                }
                if(other_vertex_nbr_index!=nbr_index){
                    goto try_next_vertex;
                }else{
                    nbr_diameter= hd_max(nbr_diameter, d_CSR_distance_matrix->entries[other_vertex_nbr_itr]);
                }
            }

            //this simply says we only consider nbr_index (the appending point) to be of larger index than the largest of shared_vertices (the vertices of the current simplex)
            if(shared_vertices[threadIdx.x * (dim + 1)]>nbr_index){
                next_cofacet= false;
                goto end_search;
            }
            next_cofacet= true;
            goto end_search;
            try_next_vertex:;
        }
        next_cofacet= false;
        end_search:;


        //end of search for next cofacet (sparse version)
        while(next_cofacet){
            base_vertex_nbr_itr--;
            value_t cofacet_diameter= hd_max(d_simplices[tid].diameter, nbr_diameter);
            index_t cofacet_index= idx_above + (*d_binomial_coeff)(nbr_index, k + 1) + idx_below;

#define FULL_MASK 0xFFFFFFFF
            int lane_id= threadIdx.x % 32;
            int keep_cofacet= cofacet_diameter<=threshold;
            int mask= __ballot_sync(FULL_MASK, keep_cofacet);
            int leader= __ffs(mask) - 1;
            int base;
            if (lane_id == leader)
                base= atomicAdd((unsigned long long int *)d_num_columns_to_reduce, __popc(mask));
            base= __shfl_sync(mask, base, leader);
            int pos= base + __popc(mask & ((1 << lane_id) - 1));

            if(keep_cofacet){
                d_columns_to_reduce[pos].diameter= cofacet_diameter;
                d_columns_to_reduce[pos].index= cofacet_index;
            }

//isn't a way to represent the hash table on gpu in a cheap way, so we ignore the hash table for assembling columns to reduce

            next_cofacet= false;
            for(; base_vertex_nbr_itr>=base_vertex_nbr_end; base_vertex_nbr_itr--){
                //nbr is the neighboring vertex to the simplex corresponding to this tid
                nbr_diameter= d_CSR_distance_matrix->entries[base_vertex_nbr_itr];
                nbr_index= d_CSR_distance_matrix->col_indices[base_vertex_nbr_itr];

                //there are dim other vertices, in addition to the base_vertex
                for(index_t other_vertex_index= 1; other_vertex_index<dim+1; other_vertex_index++){

                    index_t other_vertex= shared_vertices[threadIdx.x * (dim + 1) + other_vertex_index];
                    index_t other_vertex_nbr_itr= d_CSR_distance_matrix->offsets[other_vertex+1]-1;
                    index_t other_vertex_nbr_end= d_CSR_distance_matrix->offsets[other_vertex];
                    index_t other_vertex_nbr_index= d_CSR_distance_matrix->col_indices[other_vertex_nbr_itr];
                    while(other_vertex_nbr_index>nbr_index){
                        if(other_vertex_nbr_itr==other_vertex_nbr_end) {
                            next_cofacet= false;
                            goto end_search_inloop;
                        }
                        other_vertex_nbr_itr--;
                        other_vertex_nbr_index= d_CSR_distance_matrix->col_indices[other_vertex_nbr_itr];
                    }
                    if(other_vertex_nbr_index!=nbr_index){
                        goto try_next_vertex_inloop;
                    }else{
                        nbr_diameter= hd_max(nbr_diameter, d_CSR_distance_matrix->entries[other_vertex_nbr_itr]);
                    }
                }
                //notice we must reverse the shared_vertices in the original ripser code since they are sorted in decreasing order
                if(shared_vertices[threadIdx.x * (dim + 1)]>nbr_index){
                    next_cofacet= false;
                    goto end_search_inloop;
                }
                next_cofacet= true;
                goto end_search_inloop;
                try_next_vertex_inloop:;
            }
            next_cofacet= false;
            end_search_inloop:;
        }
    }
}

//the hope is that this is concurrency-bug free, however this is very bad for sparse graph performance
__global__ void populate_sparse_simplices_pairedfiltering(struct diameter_index_t_struct* d_simplices, index_t* d_num_simplices, struct diameter_index_t_struct* d_columns_to_reduce, index_t* d_num_columns_to_reduce, CSR_distance_matrix* d_CSR_distance_matrix, index_t num_points, index_t dim, value_t threshold, binomial_coeff_table* d_binomial_coeff){
    //a thread per (simplex , point) pair
    //if the point is a "neighbor" of the simplex, then include that cofacet in d_columns_to_reduce (a filtering of d_simplices),

    index_t tid= (index_t)threadIdx.x + (index_t)blockIdx.x * (index_t)blockDim.x;
    index_t stride= (index_t)blockDim.x * (index_t)gridDim.x;

    dim--;
    extern __shared__ index_t shared_vertices[];//a 256x(dim+1) matrix; shared_vertices[threadIdx.x*(dim+1)+j]=the jth vertex for threadIdx.x thread in the thread block
    for (; tid < *d_num_simplices*num_points; tid+= stride) {
        index_t vertex= tid%num_points;
        index_t simplex= tid/num_points;

        index_t offset= 0;
        index_t v= num_points-1;
        index_t idx= d_simplices[simplex].index;

        for (index_t k= dim +1; k > 0; --k) {

            if (!((*d_binomial_coeff)(v, k) <= idx)) {
                index_t count= v;
                while (count > 0) {
                    index_t step= count >> 1;
                    if (!((*d_binomial_coeff)(v - step, k) <= idx)) {
                        v-= step + 1;
                        count-= step + 1;//+1 is here to preserve the induction hypothesis (check v=4, k=4)
                    } else
                        count= step;//went too far, need to try a smaller step size to subtract from top
                }
            }
            shared_vertices[threadIdx.x * (dim + 1) + offset++]= v;
            idx-= (*d_binomial_coeff)(v, k);
        }

        index_t* offsets= d_CSR_distance_matrix->offsets;
        index_t* col_indices= d_CSR_distance_matrix->col_indices;
        value_t* entries= d_CSR_distance_matrix->entries;

        bool alledges_exist= true;

        index_t start_idx= offsets[vertex];
        index_t end_idx= offsets[vertex+1];
        value_t cofacet_diameter= d_simplices[simplex].diameter;
        for(index_t vidx= 0; vidx<dim+1; vidx++) {
            index_t v= shared_vertices[threadIdx.x * (dim + 1) + vidx];

            index_t left= start_idx;
            index_t right= end_idx-1;

            //binary search for v in row vertex with start and end start_idx and end_idx respectively
            while(left<=right){
                index_t mid= left+(right-left)/2;
                if(col_indices[mid]==v){
                    cofacet_diameter= hd_max(cofacet_diameter, entries[mid]);
                    goto next_vertex;
                }
                if(col_indices[mid]<v){
                    left= mid+1;
                }else{
                    right= mid-1;
                }
            }
            alledges_exist= false;
            break;
            next_vertex:;
        }
        if(!alledges_exist){
            cofacet_diameter= threshold+1;
        }

        if(shared_vertices[threadIdx.x * (dim + 1)]>vertex){
            alledges_exist= false;//we only include this vertex "vertex" if "vertex" has a  strictly larger value than all other vertices in simplex
        }

        index_t cofacet_index= (*d_binomial_coeff)(vertex, dim+2) + d_simplices[simplex].index;

#define FULL_MASK 0xFFFFFFFF
        int lane_id= threadIdx.x % 32;
        int keep_cofacet= cofacet_diameter<=threshold && alledges_exist;
        int mask= __ballot_sync(FULL_MASK, keep_cofacet);
        int leader= __ffs(mask) - 1;
        int base;
        if (lane_id == leader)
            base= atomicAdd((unsigned long long int *)d_num_columns_to_reduce, __popc(mask));
        base= __shfl_sync(mask, base, leader);
        int pos= base + __popc(mask & ((1 << lane_id) - 1));

        if(keep_cofacet){
            d_columns_to_reduce[pos].diameter= cofacet_diameter;
            d_columns_to_reduce[pos].index= cofacet_index;
        }
    }
}

template <typename T>__global__ void populate_columns_to_reduce(T* d_flagarray, struct diameter_index_t_struct* d_columns_to_reduce, index_t* d_pivot_column_index,
                                                                value_t* d_distance_matrix, index_t num_points, index_t max_num_simplices, index_t dim, value_t threshold, binomial_coeff_table* d_binomial_coeff) {
    index_t tid= (index_t)threadIdx.x + (index_t)blockIdx.x * (index_t)blockDim.x;
    index_t stride= (index_t)blockDim.x * (index_t)gridDim.x;

    extern __shared__ index_t shared_vertices[];//a 256x(dim+1) matrix; shared_vertices[threadIdx.x*(dim+1)+j]=the jth vertex for threadIdx.x thread in the thread block
    for (; tid < max_num_simplices; tid+= stride) {

        index_t offset= 0;
        index_t v= num_points - 1;
        index_t idx= tid;

        for (index_t k= dim + 1; k > 0; --k) {

            if (!((*d_binomial_coeff)(v, k) <= idx)) {
                index_t count= v;
                while (count > 0) {
                    index_t step= count >> 1;
                    if (!((*d_binomial_coeff)(v - step, k) <= idx)) {
                        v-= step + 1;
                        count-= step + 1;//+1 is here to preserve the induction hypothesis (check v=4, k=4)
                    } else
                        count= step;//went too far, need to try a smaller step size to subtract from top
                }
            }

            shared_vertices[threadIdx.x * (dim + 1) + offset++]= v;
            idx-= (*d_binomial_coeff)(v, k);
        }

        value_t diam= -MAX_FLOAT;

        for(index_t i= 0; i<=dim; i++){
            for(index_t j= i+1; j<=dim; j++){
                diam= hd_max(diam, d_distance_matrix[LOWER_DISTANCE_INDEX(shared_vertices[threadIdx.x * (dim + 1) + i], shared_vertices[threadIdx.x * (dim + 1) + j], num_points)]);
            }
        }

        if(d_pivot_column_index[tid]==-1 && diam<=threshold){
            d_columns_to_reduce[tid].diameter= diam;
            d_columns_to_reduce[tid].index= tid;
            d_flagarray[tid]= 1;
        }else{
            d_columns_to_reduce[tid].diameter= -MAX_FLOAT;
            d_columns_to_reduce[tid].index= MAX_INT64;
            d_flagarray[tid]= 0;
        }
    }
}

__global__ void init_cidx_to_diam(value_t* d_cidx_to_diameter, struct diameter_index_t_struct* d_columns_to_reduce, index_t num_columns_to_reduce){
    index_t tid= (index_t) threadIdx.x + (index_t) blockIdx.x * (index_t) blockDim.x;
    index_t stride= (index_t) blockDim.x * (index_t) gridDim.x;

    for (; tid < num_columns_to_reduce; tid += stride) {
        d_cidx_to_diameter[d_columns_to_reduce[tid].index]= d_columns_to_reduce[tid].diameter;
    }
}

//scatter operation
__global__ void init_index_to_subindex(index_t* d_index_to_subindex, index_t* d_nonapparent_columns, index_t num_nonapparent){
    index_t tid= (index_t) threadIdx.x + (index_t) blockIdx.x * (index_t) blockDim.x;
    index_t stride= (index_t) blockDim.x * (index_t) gridDim.x;

    for (; tid < num_nonapparent; tid += stride) {
        d_index_to_subindex[d_nonapparent_columns[tid]]= tid;
    }
}

//THIS IS THE GPU SCAN KERNEL for the dense case!!
__global__ void coboundary_findapparent_single_kernel(value_t* d_cidx_to_diameter, struct diameter_index_t_struct * d_columns_to_reduce, index_t* d_lowest_one_of_apparent_pair,  const index_t dim, index_t num_simplices, const index_t num_points, binomial_coeff_table* d_binomial_coeff, index_t num_columns_to_reduce, value_t* d_distance_matrix, value_t threshold) {

    index_t tid= (index_t) threadIdx.x + (index_t) blockIdx.x * (index_t) blockDim.x;
    index_t stride= (index_t) blockDim.x * (index_t) gridDim.x;

    extern __shared__ index_t shared_vertices[];//a 256x(dim+1) matrix; shared_vertices[threadIdx.x*(dim+1)+j]=the jth vertex for threadIdx.x thread in the thread block

    for (; tid < num_columns_to_reduce; tid += stride) {

        //populate the shared_vertices[][] matrix with vertex indices of the column index= shared_vertices[threadIdx.x][-];
        //shared_vertices[][] matrix has row index threadIdx.x and col index offset, represented by: shared_vertices[threadIdx.x * (dim + 1) + offset]=
        index_t offset= 0;

        index_t v= num_points - 1;
        index_t idx= d_columns_to_reduce[tid].index;

        for (index_t k= dim + 1; k > 0; --k) {

            if (!((*d_binomial_coeff)(v, k) <= idx)) {
                index_t count= v;
                while (count > 0) {
                    index_t step= count >> 1;
                    if (!((*d_binomial_coeff)(v - step, k) <= idx)) {
                        v-= step + 1;
                        count-= step + 1;//+1 is here to preserve the induction hypothesis (check v=4, k=4)
                    } else
                        count= step;//went too far, need to try a smaller step size to subtract from top
                }
            }

            shared_vertices[threadIdx.x * (dim + 1) + offset++]= v;//set v to the largest possible vertex index given idx as a combinatorial index

            idx-= (*d_binomial_coeff)(v, k);
        }
        v= num_points-1;//this keeps track of the newly added vertex to the set of vertices stored in shared_vertices[threadIdx.x][-] to form a cofacet of the columns
        index_t k= dim+1;
        index_t idx_below= d_columns_to_reduce[tid].index;
        index_t idx_above= 0;
        while ((v != -1) && ((*d_binomial_coeff)(v, k) <= idx_below)) {
            idx_below -= (*d_binomial_coeff)(v, k);
            idx_above += (*d_binomial_coeff)(v, k + 1);
            --v;
            --k;
            assert(k != -1);
        }
        while(v!=-1) {//need to enumerate cofacet combinatorial index in reverse lexicographic order (largest cidx down to lowest cidx)
            index_t row_combinatorial_index= idx_above + (*d_binomial_coeff)(v--, k + 1) + idx_below;

            //find the cofacet diameter
            value_t cofacet_diam= d_columns_to_reduce[tid].diameter;
            for(index_t j=0; j<dim+1; j++){
                index_t last_v= v+1;
                index_t simplex_v= shared_vertices[threadIdx.x * (dim + 1) + j];
                if(last_v>simplex_v){
                    cofacet_diam= hd_max(cofacet_diam, d_distance_matrix[LOWER_DISTANCE_INDEX(last_v, shared_vertices[threadIdx.x * (dim + 1) + j], num_points)]);
                }else{
                    cofacet_diam= hd_max(cofacet_diam, d_distance_matrix[LOWER_DISTANCE_INDEX(shared_vertices[threadIdx.x * (dim + 1) + j], last_v, num_points)]);
                }
            }
            if(d_columns_to_reduce[tid].diameter==cofacet_diam) {//this is a sufficient condition to finding a lowest one

                //check if there is a nonzero to the left of (row_combinatorial_index, tid) in the coboundary matrix

                //extra_vertex is the "added" vertex to shared_vertices
                //FACT: {shared_vertices[threadIdx.x*(dim+1)+0]... threadIdx.x*(dim+1)+dim] union extra_vertex} equals cofacet vertices

                index_t prev_remove_v= -1;
                index_t s_v= shared_vertices[threadIdx.x * (dim + 1)];//the largest indexed vertex, shared_vertices is sorted in decreasing orders
                bool passed_extra_v= false;
                index_t remove_v;//this is the vertex to remove from the cofacet
                index_t extra_vertex= v+1;//the +1 is here to counteract the last v-- line of code
                if(s_v>extra_vertex){
                    remove_v= s_v;
                }else{
                    remove_v= extra_vertex;
                    passed_extra_v= true;
                }
                prev_remove_v= remove_v;

                index_t facet_of_row_combinatorial_index= row_combinatorial_index;
                facet_of_row_combinatorial_index-= (*d_binomial_coeff)(remove_v, dim+2);//subtract the largest binomial coefficient to get the new cidx

                index_t col_cidx= d_columns_to_reduce[tid].index;
                value_t facet_of_row_diameter= d_cidx_to_diameter[facet_of_row_combinatorial_index];
                value_t col_diameter= d_columns_to_reduce[tid].diameter;

                if(facet_of_row_combinatorial_index==col_cidx && facet_of_row_diameter== col_diameter){//if there is an exact match of the tid column and the face of the row, then all subsequent faces to search will be to the right of column tid
                    //coboundary column tid has an apparent pair, record it
                    d_lowest_one_of_apparent_pair[tid]= row_combinatorial_index;
                    break;
                }

                    //else if(d_cidx_to_diameter[facet_of_row_combinatorial_index]<= threshold && (
                    //        d_cidx_to_diameter[facet_of_row_combinatorial_index]>d_columns_to_reduce[tid].diameter
                    //        || (d_cidx_to_diameter[facet_of_row_combinatorial_index]==d_columns_to_reduce[tid].diameter && facet_of_row_combinatorial_index<d_columns_to_reduce[tid].index)
                    //        || facet_of_row_combinatorial_index> d_columns_to_reduce[tid].index)){
                    //FACT: it turns out we actually only need to check facet_of_row_diameter<= threshold &&(facet_of_row_diameter==col_diameter && facet_of_row_combinatorial_index<col_cidx)
                    //since we should never have a facet of the cofacet with diameter larger than the cofacet's diameter= column's diameter
                    //in fact, we don't even need to check facet_of_row_diameter<=threshold since diam(face(cofacet(simplex)))<=diam(cofacet(simplex))=diam(simplex)<=threshold
                    //furthermore, we don't even need to check facet_of_row_combinatorial_index<col_cidx since we will exit upon col_cidx while iterating in increasing combinatorial index
                else if(facet_of_row_diameter==col_diameter){
                    assert(facet_of_row_diameter<= threshold && (facet_of_row_diameter==col_diameter && facet_of_row_combinatorial_index<col_cidx));
                    d_lowest_one_of_apparent_pair[tid]= -1;
                    break;
                }
                bool found_apparent_or_found_nonzero_to_left= false;

                //need to remove the last vertex: extra_v during searches
                //there are dim+2 total number of vertices, the largest vertex was already checked so that is why k starts at dim+1
                //j is the col. index e.g. shared_vertices[threadIdx.x][j]=shared_vertices[threadIdx.x*(dim+1)+j]
                for(index_t k= dim+1, j=passed_extra_v?0:1; k>=1; k--){//start the loop after checking the lexicographically smallest facet boundary case
                    if(passed_extra_v) {
                        remove_v= shared_vertices[threadIdx.x * (dim + 1) + j];
                        j++;
                    }
                    else if(j<dim+1) {
                        //compare s_v in shared_vertices with v
                        index_t s_v= shared_vertices[threadIdx.x * (dim + 1) + j];
                        if (s_v > extra_vertex) {
                            remove_v= s_v;
                            j++;
                        } else {
                            remove_v= extra_vertex;//recall: extra_vertex= v+1
                            passed_extra_v= true;
                        }
                        //this last else says: if j==dim+1 and we never passed extra vertex, then we must remove extra_vertex as the last vertex to remove to form a facet.
                    }else {//there is no need to check s_v>extra_vertex, we never passed extra_vertex, so we need to remove extra_vertex for the last check
                        remove_v= extra_vertex;//recall; v+1 since there is a v-- before this
                        passed_extra_v= true;
                    }

                    //exchange remove_v choose k with prev_remove_v choose k
                    facet_of_row_combinatorial_index-=(*d_binomial_coeff)(remove_v,k);
                    facet_of_row_combinatorial_index+= (*d_binomial_coeff)(prev_remove_v,k);

                    value_t facet_of_row_diameter= d_cidx_to_diameter[facet_of_row_combinatorial_index];

                    if(facet_of_row_combinatorial_index==col_cidx && facet_of_row_diameter==col_diameter){
                        //coboundary column tid has an apparent pair, record it
                        d_lowest_one_of_apparent_pair[tid]= row_combinatorial_index;
                        found_apparent_or_found_nonzero_to_left= true;
                        break;///need to break out the while(v!=-1) loop
                    }

                        //else if(d_cidx_to_diameter[facet_of_row_combinatorial_index]<=threshold &&
                        //( d_cidx_to_diameter[facet_of_row_combinatorial_index]>d_columns_to_reduce[tid].diameter
                        //|| (d_cidx_to_diameter[facet_of_row_combinatorial_index]==d_columns_to_reduce[tid].diameter && facet_of_row_combinatorial_index<d_columns_to_reduce[tid].index)
                        //|| facet_of_row_combinatorial_index>d_columns_to_reduce[tid].index)){
                    else if(facet_of_row_diameter==col_diameter){
                        assert(facet_of_row_diameter<= threshold && (facet_of_row_diameter==col_diameter && facet_of_row_combinatorial_index<col_cidx));
                        //d_lowest_one_of_apparent_pair[] is set to -1's already though...
                        d_lowest_one_of_apparent_pair[tid]= -1;
                        found_apparent_or_found_nonzero_to_left= true;
                        break;
                    }

                    prev_remove_v= remove_v;
                }
                //we must exit early if we have a nonzero to left or the column is apparent
                if(found_apparent_or_found_nonzero_to_left){
                    break;
                }


                //end check for nonzero to left

                //need to record the found pairs in the global hash_map for pairs (post processing)
                //see post processing section in gpuscan method
            }
            while ((v != -1) && ((*d_binomial_coeff)(v, k) <= idx_below)) {
                idx_below -= (*d_binomial_coeff)(v, k);
                idx_above += (*d_binomial_coeff)(v, k + 1);
                --v;
                --k;
                assert(k != -1);
            }
        }
    }
}

//gpuscan for sparse case
__global__ void coboundary_findapparent_sparse_single_kernel(struct diameter_index_t_struct* d_cidx_diameter_sorted_list, struct diameter_index_t_struct * d_columns_to_reduce, index_t* d_lowest_one_of_apparent_pair,  const index_t dim, const index_t num_points, binomial_coeff_table* d_binomial_coeff, index_t num_columns_to_reduce, CSR_distance_matrix* d_CSR_distance_matrix, value_t threshold){//(this was for debugging), index_t* d_leftmostnz_inrow) {
    index_t tid= (index_t) threadIdx.x + (index_t) blockIdx.x * (index_t) blockDim.x;
    index_t stride= (index_t) blockDim.x * (index_t) gridDim.x;

    extern __shared__ index_t shared_vertices[];//a 256x(dim+1) matrix; shared_vertices[threadIdx.x*(dim+1)+j]=the jth vertex for threadIdx.x thread in the thread block
    //vertices sorted in reverse order

    for (; tid < num_columns_to_reduce; tid += stride) {
        //populate the shared_vertices[][] matrix with vertex indices of the column tid;
        //row index of the shared_vertices matrix is threadIdx.x, col index of the shared_vertices matrix is offset
        index_t offset= 0;

        index_t v= num_points - 1;
        index_t idx= d_columns_to_reduce[tid].index;

        for (index_t k= dim + 1; k > 0; --k) {

            if (!((*d_binomial_coeff)(v, k) <= idx)) {
                index_t count= v;
                while (count > 0) {
                    index_t step= count >> 1;
                    if (!((*d_binomial_coeff)(v - step, k) <= idx)) {
                        v-= step + 1;
                        count-= step + 1;//+1 is here to preserve the induction hypothesis (check v=4, k=4)
                    } else
                        count= step;//went too far, need to try a smaller step size to subtract from top
                }
            }

            shared_vertices[threadIdx.x * (dim + 1) + offset++]= v;//set v to the largest possible vertex index given idx as a combinatorial index

            idx-= (*d_binomial_coeff)(v, k);
        }
        index_t k= dim+1;

        bool next_cofacet= false;
        value_t nbr_diameter= -1;
        index_t nbr_index= -1;
        index_t idx_below= d_columns_to_reduce[tid].index;
        index_t idx_above= 0;

        index_t base_vertex_index= shared_vertices[threadIdx.x * (dim + 1)];
        //this gives the entry indices of the right and left ends of the row indexed by base_vertex_index in the CSR distance matrix
        index_t base_vertex_nbr_itr= d_CSR_distance_matrix->offsets[base_vertex_index+1]-1;
        index_t base_vertex_nbr_end= d_CSR_distance_matrix->offsets[base_vertex_index];

        for(; base_vertex_nbr_itr>=base_vertex_nbr_end; base_vertex_nbr_itr--){
            //nbr is the neighboring vertex to the simplex corresponding to this tid
            nbr_diameter= d_CSR_distance_matrix->entries[base_vertex_nbr_itr];
            nbr_index= d_CSR_distance_matrix->col_indices[base_vertex_nbr_itr];
            //there are dim other vertices besides the base_vertex
            for(index_t other_vertex_idx=1; other_vertex_idx<dim+1; other_vertex_idx++){

                index_t other_vertex= shared_vertices[threadIdx.x * (dim + 1) + other_vertex_idx];
                index_t other_vertex_nbr_itr= d_CSR_distance_matrix->offsets[other_vertex+1]-1;
                index_t other_vertex_nbr_end= d_CSR_distance_matrix->offsets[other_vertex];
                index_t other_vertex_nbr_index= d_CSR_distance_matrix->col_indices[other_vertex_nbr_itr];
                while(other_vertex_nbr_index>nbr_index){
                    if(other_vertex_nbr_itr==other_vertex_nbr_end) {
                        next_cofacet= false;
                        goto end_search;
                    }
                    other_vertex_nbr_itr--;
                    other_vertex_nbr_index= d_CSR_distance_matrix->col_indices[other_vertex_nbr_itr];
                }
                if(other_vertex_nbr_index!=nbr_index){
                    goto try_next_vertex;
                }else{
                    nbr_diameter= hd_max(nbr_diameter, d_CSR_distance_matrix->entries[other_vertex_nbr_itr]);
                }
            }
            while (k > 0 && shared_vertices[threadIdx.x * (dim + 1) + dim- (k - 1)] > nbr_index) {
                idx_below -= (*d_binomial_coeff)(shared_vertices[threadIdx.x * (dim + 1) + dim- (k - 1)], k);
                idx_above += (*d_binomial_coeff)(shared_vertices[threadIdx.x * (dim + 1) + dim- (k - 1)], k + 1);
                --k;
            }
            next_cofacet= true;
            goto end_search;
            try_next_vertex:;
        }
        next_cofacet= false;
        end_search:;

        //end of search for next cofacet (sparse version)
        while(next_cofacet) {
            base_vertex_nbr_itr--;
            value_t cofacet_diameter= hd_max(d_columns_to_reduce[tid].diameter, nbr_diameter);
            index_t row_combinatorial_index= idx_above + (*d_binomial_coeff)(nbr_index, k + 1) + idx_below;
            if(d_columns_to_reduce[tid].diameter==cofacet_diameter) {//this is a sufficient condition to finding a lowest one

                //check if there is a nonzero to the left of (row_combinatorial_index, tid) in the coboundary matrix

                //extra_vertex is the "added" vertex to shared_verticess
                //FACT: {shared_vertices[threadIdx.x*(dim+1)+0]... shared_vertices[threadIdx.x*(dim+1)+dim] union extra_vertex} equals cofacet vertices

                index_t prev_remove_v= -1;
                index_t s_v= shared_vertices[threadIdx.x * (dim + 1)];//the largest indexed vertex, shared_vertices is sorted in decreasing orders
                bool passed_extra_v= false;
                index_t remove_v;//this is the vertex to remove from the cofacet
                index_t extra_vertex= nbr_index;//the +1 is here to counteract the last v-- line of code
                if (s_v > extra_vertex) {
                    remove_v= s_v;
                } else {
                    remove_v= extra_vertex;
                    passed_extra_v= true;
                }
                prev_remove_v= remove_v;
                index_t facet_of_row_combinatorial_index= row_combinatorial_index;
                facet_of_row_combinatorial_index-= (*d_binomial_coeff)(remove_v, dim + 2);//subtract the largest binomial coefficient to get the new cidx
                index_t col_cidx= d_columns_to_reduce[tid].index;
                value_t col_diameter= d_columns_to_reduce[tid].diameter;
                //binary search d_columns_to_reduce to get face_of_row_diameter
                value_t facet_of_row_diameter= -1;// there is no direct mapping: d_cidx_to_diameter[facet_of_row_combinatorial_index];

                ///binary search goes here on d_cidx_diameter_sorted_list
                index_t left= 0;
                index_t right= num_columns_to_reduce-1;

                while(left<=right){
                    index_t mid= left + (right-left)/2;
                    if(d_cidx_diameter_sorted_list[mid].index==facet_of_row_combinatorial_index){
                        facet_of_row_diameter= d_cidx_diameter_sorted_list[mid].diameter;
                        break;
                    }
                    if(d_cidx_diameter_sorted_list[mid].index<facet_of_row_combinatorial_index){
                        left= mid+1;
                    }else{
                        right= mid-1;
                    }
                }
                if (facet_of_row_combinatorial_index == col_cidx && facet_of_row_diameter == col_diameter) {//if there is an exact match of the tid column and the face of the row, then all subsequent faces to search will be to the right of column tid
                    //coboundary column tid has an apparent pair, record it
                    d_lowest_one_of_apparent_pair[tid]= row_combinatorial_index;
                    break;
                }

                    //else if(d_cidx_to_diameter[facet_of_row_combinatorial_index]<= threshold && (
                    //        d_cidx_to_diameter[facet_of_row_combinatorial_index]>d_columns_to_reduce[tid].diameter
                    //        || (d_cidx_to_diameter[facet_of_row_combinatorial_index]==d_columns_to_reduce[tid].diameter && facet_of_row_combinatorial_index<d_columns_to_reduce[tid].index)
                    //        || facet_of_row_combinatorial_index> d_columns_to_reduce[tid].index)){
                    //FACT: it turns out we actually only need to check facet_of_row_diameter<= threshold &&(facet_of_row_diameter==col_diameter && facet_of_row_combinatorial_index<col_cidx)
                    //since we should never have a face of the cofacet with diameter larger than the cofacet's diameter= column's diameter
                    //in fact, we don't even need to check facet_of_row_diameter<=threshold since diam(face(cofacet(simplex)))<=diam(cofacet(simplex))=diam(simplex)<=threshold
                    //furthremore, we don't even need to check facet_of_row_combinatorial_index<col_cidx since we will exit upon col_cidx while iterating in increasing combinatorial index
                else if (facet_of_row_diameter == col_diameter) {
                    assert(facet_of_row_diameter <= threshold &&
                           (facet_of_row_diameter == col_diameter && facet_of_row_combinatorial_index < col_cidx));
                    d_lowest_one_of_apparent_pair[tid]= -1;
                    break;
                }
                bool found_apparent_or_found_nonzero_to_left= false;
                //need to remove the last vertex: extra_v during searches
                //there are dim+2 total number of vertices, the largest vertex was already checked so that is why k starts at dim+1
                //j is the col. index, e.g. shared_vertices[threadIdx.x][j]=shared_vertices[threadIdx.x*(dim+1)+j]
                for (index_t k= dim + 1, j= passed_extra_v ? 0 : 1;
                     k >= 1; k--) {//start the loop after checking the lexicographically smallest facet boundary case
                    if (passed_extra_v) {
                        remove_v= shared_vertices[threadIdx.x * (dim + 1) + j];
                        j++;
                    } else if (j < dim + 1) {
                        //compare s_v in shared_vertices with v
                        index_t s_v= shared_vertices[threadIdx.x * (dim + 1) + j];
                        if (s_v > extra_vertex) {
                            remove_v= s_v;
                            j++;
                        } else {
                            remove_v= extra_vertex;//recall: extra_vertex=nbr_index;
                            passed_extra_v= true;
                        }
                        //this last else says: if j==dim+1 and we never passed extra vertex, then we must remove extra_vertex as the last vertex to remove to form a face.
                    } else {//there is no need to check s_v>extra_vertex, we never passed extra_vertex, so we need to remove extra_vertex for the last check
                        remove_v= extra_vertex;//recall; extra_vertex= nbr_index
                        passed_extra_v= true;
                    }

                    //exchange remove_v choose k with prev_remove_v choose k
                    facet_of_row_combinatorial_index -= (*d_binomial_coeff)(remove_v, k);
                    facet_of_row_combinatorial_index += (*d_binomial_coeff)(prev_remove_v, k);

                    //replace d_cidx_to_diameter with d_cidx_diameter_sorted_list;
                    value_t facet_of_row_diameter= -1;// replacing direct map:: d_cidx_to_diameter[facet_of_row_combinatorial_index];

                    ///binary search goes here on d_cidx_diameter_sorted_list
                    index_t left= 0;
                    index_t right= num_columns_to_reduce-1;

                    while(left<=right){
                        index_t mid= left + (right-left)/2;
                        if(d_cidx_diameter_sorted_list[mid].index==facet_of_row_combinatorial_index){
                            facet_of_row_diameter= d_cidx_diameter_sorted_list[mid].diameter;
                            break;
                        }
                        if(d_cidx_diameter_sorted_list[mid].index<facet_of_row_combinatorial_index){
                            left= mid+1;
                        }else{
                            right= mid-1;
                        }
                    }

                    if (facet_of_row_combinatorial_index == col_cidx && facet_of_row_diameter == col_diameter) {
                        //coboundary column tid has an apparent pair, record it
                        d_lowest_one_of_apparent_pair[tid]= row_combinatorial_index;
                        found_apparent_or_found_nonzero_to_left= true;
                        break;///need to break out the while(v!=-1) loop
                    }

                        //else if(d_cidx_to_diameter[facet_of_row_combinatorial_index]<=threshold &&
                        //( d_cidx_to_diameter[facet_of_row_combinatorial_index]>d_columns_to_reduce[tid].diameter
                        //|| (d_cidx_to_diameter[facet_of_row_combinatorial_index]==d_columns_to_reduce[tid].diameter && facet_of_row_combinatorial_index<d_columns_to_reduce[tid].index)
                        //|| facet_of_row_combinatorial_index>d_columns_to_reduce[tid].index)){
                    else if (facet_of_row_diameter == col_diameter) {
                        assert(facet_of_row_diameter <= threshold &&
                               (facet_of_row_diameter == col_diameter && facet_of_row_combinatorial_index < col_cidx));

                        //d_lowest_one_of_apparent_pair[tid]= -1;
                        found_apparent_or_found_nonzero_to_left= true;
                        break;
                    }

                    prev_remove_v= remove_v;
                }
                //we must exit early if we have a nonzero to left or the column is apparent
                if (found_apparent_or_found_nonzero_to_left) {
                    break;
                }


                //end check for nonzero to left
            }


            next_cofacet= false;
            for(; base_vertex_nbr_itr>=base_vertex_nbr_end; base_vertex_nbr_itr--){
                //nbr is the neighboring vertex to the simplex corresponding to this tid
                nbr_diameter= d_CSR_distance_matrix->entries[base_vertex_nbr_itr];
                nbr_index= d_CSR_distance_matrix->col_indices[base_vertex_nbr_itr];
                //there are dim other vertices besides the base_vertex
                for(index_t other_vertex_index=1; other_vertex_index<dim+1; other_vertex_index++){

                    index_t other_vertex= shared_vertices[threadIdx.x * (dim + 1) + other_vertex_index];
                    index_t other_vertex_nbr_itr= d_CSR_distance_matrix->offsets[other_vertex+1]-1;
                    index_t other_vertex_nbr_end= d_CSR_distance_matrix->offsets[other_vertex];
                    index_t other_vertex_nbr_index= d_CSR_distance_matrix->col_indices[other_vertex_nbr_itr];
                    while(other_vertex_nbr_index>nbr_index){
                        if(other_vertex_nbr_itr==other_vertex_nbr_end) {
                            next_cofacet= false;
                            goto end_search_inloop;
                        }
                        other_vertex_nbr_itr--;
                        other_vertex_nbr_index= d_CSR_distance_matrix->col_indices[other_vertex_nbr_itr];
                    }
                    if(other_vertex_nbr_index!=nbr_index){
                        goto try_next_vertex_inloop;
                    }else{
                        nbr_diameter= hd_max(nbr_diameter, d_CSR_distance_matrix->entries[other_vertex_nbr_itr]);
                    }
                }
                //notice we must reverse the shared_vertices since they are sorted in decreasing order
                while (k > 0 && shared_vertices[threadIdx.x * (dim + 1) + dim- (k - 1)] > nbr_index) {
                    idx_below -= (*d_binomial_coeff)(shared_vertices[threadIdx.x * (dim + 1) + dim- (k - 1)], k);
                    idx_above += (*d_binomial_coeff)(shared_vertices[threadIdx.x * (dim + 1) + dim- (k - 1)], k + 1);
                    --k;
                }
                next_cofacet= true;
                goto end_search_inloop;
                try_next_vertex_inloop:;
            }
            next_cofacet= false;
            end_search_inloop:;
        }
    }
}

template <typename DistanceMatrix> class ripser {
    DistanceMatrix dist;//this can be either sparse or compressed

    index_t n, dim_max;//n is the number of points, dim_max is the max dimension to compute PH
    value_t threshold;//this truncates the filtration by removing simplices too large. low values of threshold should use --sparse option
    float ratio;
    const binomial_coeff_table binomial_coeff;
    mutable std::vector<index_t> vertices;
    mutable std::vector<diameter_index_t_struct> cofacet_entries;
private:
    size_t freeMem, totalMem;
    cudaDeviceProp deviceProp;
    int grid_size;
    hash_map<index_t, index_t> pivot_column_index;//small hash map for matrix reduction

    //we are removing d_flagarray for a more general array: d_flagarray_OR_index_to_subindex
    //char* type is 3x faster for thrust::count than index_t*
#ifndef ASSEMBLE_REDUCTION_SUBMATRIX
    char* d_flagarray;//an array where d_flagarray[i]= 1 if i satisfies some property and d_flagarray[i]=0 otherwise
#endif
    index_t* h_pivot_column_index_array_OR_nonapparent_cols;//the pivot column index hashmap represented by an array OR the set of nonapparent column indices

    value_t* d_distance_matrix;//GPU copy of the distance matrix
    CSR_distance_matrix* d_CSR_distance_matrix;
    index_t *h_d_offsets;
    value_t *h_d_entries;
    index_t *h_d_col_indices;

    //d_pivot_column_index_OR_nonapparent_cols is d_nonapparent_cols when used in gpuscan() and compute_pairs() and is d_pivot_column_index when in gpu_assemble_columns()
    index_t* d_pivot_column_index_OR_nonapparent_cols;//the pivot column index hashmap represented on GPU as an array OR the set of nonapparent columns on GPU

    index_t max_num_simplices_forall_dims;//the total number of simplices of dimension dim_max possible (this assumes no threshold condition to sparsify the simplicial complex)
    //the total number of simplices in the dim_max+1 dimension (a factor n larger than max_num_simplices_forall_dims), infeasible to allocate with this number if max_num_simplices_forall_dims is already pushing the memory limits.

    struct diameter_index_t_struct* d_columns_to_reduce;//GPU copy of the columns to reduce depending on the current dimension
    struct diameter_index_t_struct* h_columns_to_reduce;//columns to reduce depending on the current dimension

    binomial_coeff_table* d_binomial_coeff;//GPU copy of the binomial coefficient table
    index_t* h_d_binoms;

    index_t* d_num_columns_to_reduce=NULL;//use d_num_columns_to_reduce to keep track of the number of columns to reduce
    index_t* h_num_columns_to_reduce;//h_num_columns_to_reduce is tied to d_num_columns_to_reduce in pinned memory?

    index_t* d_num_nonapparent= NULL;//the number of nonapparent columns. *d_num_columns_to_reduce-*d_num_nonapparent= number of apparent columns
    index_t* h_num_nonapparent;//h_num_nonapparent is tied to d_num_nonapparent in pinned memory?

    index_t num_apparent;//the number of apparent pairs found

    value_t* d_cidx_to_diameter;//GPU side mapping from cidx to diameters for gpuscan faces of a given row of a "lowest one" search
    struct diameter_index_t_struct* d_cidx_diameter_pairs_sortedlist;//used as a sorted list of cidx,diameter pairs for lookup in gpuscan kernel for sparse case

#if defined(ASSEMBLE_REDUCTION_SUBMATRIX)//assemble reduction submatrix
    index_t* d_flagarray_OR_index_to_subindex;//GPU data structure that maps index to subindex
    index_t* h_flagarray_OR_index_to_subindex;//copy of index_to_subindex data structure that acts as a map for matrix index to reduction submatrix indexing on CPU side
#endif

    //for GPU-scan (finding apparent pairs)
    index_t* d_lowest_one_of_apparent_pair;//GPU copy of the lowest ones, d_lowest_one_of_apparent_pair[col]= lowest one row of column col
    //index_t* h_lowest_one_of_apparent_pair;//the lowest ones, d_lowest_one_of_apparent_pair[col]= lowest one row of column col
    struct index_t_pair_struct* d_pivot_array;//sorted array of all pivots, substitute for a structured hashmap with lookup done by log(n) binary search
    struct index_t_pair_struct* h_pivot_array;//sorted array of all pivots
    std::vector<struct diameter_index_t_struct> columns_to_reduce;

    //used for sparse_distance_matrix ONLY:
    struct diameter_index_t_struct* d_simplices;//GPU copy of h_simplices
    struct diameter_index_t_struct* h_simplices;//the simplices filtered by diameter that need to be considered for the next dimension's simplices
    index_t* d_num_simplices=NULL;//use d_num_simplices to keep track of the number of simplices in h_ or d_ simplices
    index_t* h_num_simplices;//h_num_simplices is tied to d_num_simplices in pinned memory
public:

    ripser(DistanceMatrix&& _dist, index_t _dim_max, value_t _threshold, float _ratio)
            : dist(std::move(_dist)), n(dist.size()),
              dim_max(std::min(_dim_max, index_t(dist.size() - 2))), threshold(_threshold),
              ratio(_ratio), binomial_coeff(n, dim_max + 2) {}

    void free_gpumem_dense_computation() {
        if (n>=10) {//this fixes a bug for single point persistence being called repeatedly
            cudaFree(d_columns_to_reduce);
#ifndef ASSEMBLE_REDUCTION_SUBMATRIX
            cudaFree(d_flagarray);
#endif
            cudaFree(d_cidx_to_diameter);
//            if (n >= 10) {
                cudaFree(d_distance_matrix);
//            }
            cudaFree(d_pivot_column_index_OR_nonapparent_cols);
#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
            cudaFree(d_flagarray_OR_index_to_subindex);
#endif
//            if (binomial_coeff.get_num_n() * binomial_coeff.get_max_tuple_length() > 0) {
                cudaFree(h_d_binoms);
//            }
            cudaFree(d_binomial_coeff);
            cudaFree(d_lowest_one_of_apparent_pair);
            cudaFree(d_pivot_array);
        }
    }
    void free_gpumem_sparse_computation() {
        if (n >= 10) {
            cudaFree(d_columns_to_reduce);
#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
            cudaFree(d_flagarray_OR_index_to_subindex);
#endif
//            if (CSR_distance_matrix .num_entries > 0) {
                cudaFree(h_d_entries);
//            }
            cudaFree(h_d_offsets);

            cudaFree(h_d_col_indices);
            cudaFree(d_CSR_distance_matrix);
            cudaFree(d_cidx_diameter_pairs_sortedlist);
            cudaFree(d_pivot_column_index_OR_nonapparent_cols);
            //if (binomial_coeff.get_num_n() * binomial_coeff.get_max_tuple_length() > 0) {
                cudaFree(h_d_binoms);
            //}
            cudaFree(d_binomial_coeff);
            cudaFree(d_lowest_one_of_apparent_pair);
            cudaFree(d_pivot_array);
            cudaFree(d_simplices);
        }
    }

    void free_init_cpumem() {
        free(h_pivot_column_index_array_OR_nonapparent_cols);
    }

    void free_remaining_cpumem(){
        free(h_columns_to_reduce);
        free(h_pivot_array);
        pivot_column_index.resize(0);
    }

    //calulate gpu_num_simplices_forall_dims based on GPU memory limit
    index_t calculate_gpu_max_columns_for_sparserips_computation_from_memory(){
        cudaGetDeviceProperties(&deviceProp, 0);

        cudaMemGetInfo(&freeMem,&totalMem);
#ifdef PROFILING
        std::cerr<<"before calculation, sparse: total mem, free mem: "<<totalMem <<" bytes, "<<freeMem<<" bytes"<<std::endl;
#endif
        index_t gpumem_char_array_bytes_factor= sizeof(char);
        index_t gpumem_index_t_array_bytes_factor= sizeof(index_t);
        index_t gpumem_value_t_array_bytes_factor= sizeof(value_t);
        index_t gpumem_index_t_pairs_array_bytes_factor= sizeof(index_t_pair_struct);
        index_t gpumem_diameter_index_t_array_bytes_factor= sizeof(diameter_index_t_struct);
        index_t gpumem_CSR_dist_matrix_bytes= sizeof(index_t)*(n+1+4)+(sizeof(index_t)+sizeof(value_t))*dist.num_entries;//sizeof(value_t)*(n*(n-1))/2;
        index_t gpumem_binomial_coeff_table_bytes= sizeof(index_t)*binomial_coeff.get_num_n()*binomial_coeff.get_max_tuple_length() +sizeof(binomial_coeff_table);
        index_t gpumem_index_t_bytes= sizeof(index_t);
        index_t padding= 1024*1024*1024;//1GB padding

        index_t fixedmemory= gpumem_index_t_bytes*4+gpumem_binomial_coeff_table_bytes+gpumem_CSR_dist_matrix_bytes+padding;
//this can be larger but not smaller than actual sizeof(-) sum
        index_t sizeof_factor_sum=
                gpumem_diameter_index_t_array_bytes_factor
                #ifdef ASSEMBLE_REDUCTION_SUBMATRIX
                +gpumem_index_t_array_bytes_factor
                #endif
                +gpumem_diameter_index_t_array_bytes_factor
                +gpumem_index_t_array_bytes_factor
                +gpumem_index_t_array_bytes_factor
                +gpumem_index_t_pairs_array_bytes_factor
                +gpumem_diameter_index_t_array_bytes_factor
                +gpumem_index_t_pairs_array_bytes_factor;

#ifdef PROFILING
        std::cerr<<"sparse final calculation for memory, free memory: "<<freeMem <<" bytes, sizeof_factor_sum: "<<sizeof_factor_sum<<" bytes"<<std::endl;
#endif
        return (freeMem*0.7-fixedmemory)/sizeof_factor_sum;
    }

    index_t calculate_gpu_dim_max_for_fullrips_computation_from_memory(const index_t dim_max, const bool isfullrips){

        if(dim_max==0)return 0;
        index_t gpu_dim_max= dim_max;
        index_t gpu_alloc_memory_in_bytes= 0;
        cudaGetDeviceProperties(&deviceProp, 0);

        cudaMemGetInfo(&freeMem,&totalMem);
#ifdef PROFILING
        std::cerr<<"GPU memory before full rips memory calculation, total mem: "<< totalMem<<" bytes, free mem: "<<freeMem<<" bytes"<<std::endl;
#endif
        do{
            index_t gpu_num_simplices_forall_dims= gpu_dim_max<n/2?get_num_simplices_for_dim(gpu_dim_max): get_num_simplices_for_dim(n/2);
            index_t gpumem_char_array_bytes= sizeof(char)*gpu_num_simplices_forall_dims;
            index_t gpumem_index_t_array_bytes= sizeof(index_t)*gpu_num_simplices_forall_dims;
            index_t gpumem_value_t_array_bytes= sizeof(value_t)*gpu_num_simplices_forall_dims;
            index_t gpumem_index_t_pairs_array_bytes= sizeof(index_t_pair_struct)*gpu_num_simplices_forall_dims;
            index_t gpumem_diameter_index_t_array_bytes= sizeof(diameter_index_t_struct)*gpu_num_simplices_forall_dims;
            index_t gpumem_dist_matrix_bytes= sizeof(value_t)*(n*(n-1))/2;
            index_t gpumem_binomial_coeff_table_bytes= sizeof(index_t)*binomial_coeff.get_num_n()*binomial_coeff.get_max_tuple_length() +sizeof(binomial_coeff_table);
            index_t gpumem_index_t_bytes= sizeof(index_t);
            //gpumem_CSR_dist_matrix_bytes is estimated to have n*(n-1)/2 number of nonzeros as an upper bound
            index_t gpumem_CSR_dist_matrix_bytes= sizeof(index_t)*(n+1+4)+(sizeof(index_t)+sizeof(value_t))*n*(n-1)/2;//dist.num_entries;//sizeof(value_t)*(n*(n-1))/2;

            if(isfullrips) {//count the allocated memory for dense case
                gpu_alloc_memory_in_bytes= gpumem_diameter_index_t_array_bytes +
                                           #ifndef ASSEMBLE_REDUCTION_SUBMATRIX
                                           gpumem_char_array_bytes +
                                           #endif
                                           gpumem_value_t_array_bytes +
                                           #ifdef ASSEMBLE_REDUCTION_SUBMATRIX
                                           gpumem_index_t_array_bytes+
                                           #endif
                                           gpumem_dist_matrix_bytes +
                                           gpumem_index_t_array_bytes +
                                           gpumem_binomial_coeff_table_bytes +
                                           gpumem_index_t_bytes * 2 +
                                           gpumem_index_t_array_bytes +
                                           gpumem_index_t_pairs_array_bytes +
                                           gpumem_index_t_pairs_array_bytes;//this last one is for thrust radix sorting buffer

#ifdef PROFILING
                //std::cerr<<"free gpu memory for full rips by calculation in bytes for gpu dim: "<<gpu_dim_max<<": "<<freeMem-gpu_alloc_memory_in_bytes<<std::endl;
                std::cerr<<"gpu memory needed for full rips by calculation in bytes for dim: "<<gpu_dim_max<<": "<<gpu_alloc_memory_in_bytes<<" bytes"<<std::endl;
#endif
                if (gpu_alloc_memory_in_bytes <= freeMem){
                    return gpu_dim_max;
                }
            }else{//count the alloced memory for sparse case
                //includes the d_simplices array used in sparse computation for an approximation for both sparse and full rips compelexes?
                gpu_alloc_memory_in_bytes= gpumem_diameter_index_t_array_bytes
                                           #ifdef ASSEMBlE_REDUCTION_SUBMATRIX
                                           + gpumem_index_t_array_bytes
                                           #endif
                                           + gpumem_CSR_dist_matrix_bytes
                                           + gpumem_diameter_index_t_array_bytes
                                           + gpumem_index_t_array_bytes
                                           + gpumem_binomial_coeff_table_bytes
                                           + gpumem_index_t_array_bytes
                                           + gpumem_index_t_pairs_array_bytes
                                           + gpumem_index_t_bytes*4
                                           + gpumem_diameter_index_t_array_bytes
                                           + gpumem_index_t_pairs_array_bytes;//last one is for buffer needed for sorting
#ifdef PROFILING
                //std::cerr<<"(sparse) free gpu memory for full rips by calculation in bytes for gpu dim: "<<gpu_dim_max<<": "<<freeMem-gpu_alloc_memory_in_bytes<<std::endl;
                std::cerr<<"(sparse) gpu memory needed for full rips by calculation in bytes for dim: "<<gpu_dim_max<<": "<<gpu_alloc_memory_in_bytes<<" bytes"<<std::endl;
#endif
                if (gpu_alloc_memory_in_bytes <= freeMem){
                    return gpu_dim_max;
                }
            }
            gpu_dim_max--;
        }while(gpu_dim_max>=0);
        return 0;
    }

    index_t get_num_simplices_for_dim(index_t dim){
        //beware if dim+1>n and where dim is negative
        assert(dim+1<=n && dim+1>=0);
        return binomial_coeff(n, dim + 1);
    }

    index_t get_next_vertex(index_t& v, const index_t idx, const index_t k) const {
        return v= upper_bound(
                v, [&](const index_t& w) -> bool { return (binomial_coeff(w, k) <= idx); });
    }

    index_t get_edge_index(const index_t i, const index_t j) const {
        return binomial_coeff(i, 2) + j;
    }

    template <typename OutputIterator>
    OutputIterator get_simplex_vertices(index_t idx, const index_t dim, index_t v,
                                        OutputIterator out) const {
        --v;
        for (index_t k= dim + 1; k > 0; --k) {
            get_next_vertex(v, idx, k);
            *out++= v;
            idx-= binomial_coeff(v, k);
        }
        return out;
    }

    value_t compute_diameter(const index_t index, index_t dim) const {
        value_t diam= -std::numeric_limits<value_t>::infinity();

        vertices.clear();
        get_simplex_vertices(index, dim, dist.size(), std::back_inserter(vertices));

        for (index_t i= 0; i <= dim; ++i)
            for (index_t j= 0; j < i; ++j) {
                diam= std::max(diam, dist(vertices[i], vertices[j]));
            }
        return diam;
    }

    class simplex_coboundary_enumerator;

    void gpu_assemble_columns_to_reduce_plusplus(const index_t dim);

    void cpu_byneighbor_assemble_columns_to_reduce(std::vector<struct diameter_index_t_struct>& simplices, std::vector<struct diameter_index_t_struct>& columns_to_reduce,
                                                   hash_map<index_t, index_t>& pivot_column_index, index_t dim);

    void cpu_assemble_columns_to_reduce(std::vector<struct diameter_index_t_struct>& columns_to_reduce,
                                        hash_map<index_t, index_t>& pivot_column_index, index_t dim);

    void assemble_columns_gpu_accel_transition_to_cpu_only(const bool& more_than_one_dim_cpu_only, std::vector<diameter_index_t_struct>& simplices, std::vector<diameter_index_t_struct>& columns_to_reduce, hash_map<index_t,index_t>& cpu_pivot_column_index, index_t dim);

    index_t get_value_pivot_array_hashmap(index_t row_cidx, struct row_cidx_column_idx_struct_compare cmp){
#ifdef USE_PHASHMAP
        index_t col_idx= phmap_get_value(row_cidx);
        if(col_idx==-1){
#endif
#ifdef USE_GOOGLE_HASHMAP
            auto pair= pivot_column_index.find(row_cidx);
        if(pair==pivot_column_index.end()){
#endif
            index_t first= 0;
            index_t last= num_apparent- 1;

            while(first<=last){
                index_t mid= first + (last-first)/2;
                if(h_pivot_array[mid].row_cidx==row_cidx){
                    return h_pivot_array[mid].column_idx;
                }
                if(h_pivot_array[mid].row_cidx<row_cidx){
                    first= mid+1;
                }else{
                    last= mid-1;
                }
            }
            return -1;

        }else{

#ifdef USE_PHASHMAP
            return col_idx;
#endif
#ifdef USE_GOOGLE_HASHMAP
            return pair->second;
#endif
        }
    }


    void compute_dim_0_pairs(std::vector<diameter_index_t_struct>& edges,
                             std::vector<diameter_index_t_struct>& columns_to_reduce) {
#ifdef PRINT_PERSISTENCE_PAIRS
        std::cout << "persistence intervals in dim 0:" << std::endl;
#endif

        union_find dset(n);

        edges= get_edges();
        struct greaterdiam_lowerindex_diameter_index_t_struct_compare cmp;

        std::sort(edges.rbegin(), edges.rend(), cmp);

        std::vector<index_t> vertices_of_edge(2);
        for (auto e : edges) {
            get_simplex_vertices(e.index, 1, n, vertices_of_edge.rbegin());
            index_t u= dset.find(vertices_of_edge[0]), v= dset.find(vertices_of_edge[1]);

            if (u != v) {
#if defined(PRINT_PERSISTENCE_PAIRS) || defined(PYTHON_BARCODE_COLLECTION)
                if(e.diameter!=0) {
#ifdef PRINT_PERSISTENCE_PAIRS
                    std::cout << " [0," << e.diameter << ")" << std::endl;
#endif
                    //Collect persistence pair
                    birth_death_coordinate barcode = {0,e.diameter};
                    list_of_barcodes[0].push_back(barcode);
                }
#endif
                dset.link(u, v);
            } else {
                columns_to_reduce.push_back(e);
            }
        }
        std::reverse(columns_to_reduce.begin(), columns_to_reduce.end());

#ifdef PRINT_PERSISTENCE_PAIRS
        for (index_t i= 0; i < n; ++i)
            if (dset.find(i) == i) std::cout << " [0, )" << std::endl;
#endif
    }
    void gpu_compute_dim_0_pairs(std::vector<struct diameter_index_t_struct>& columns_to_reduce);

    void gpuscan(const index_t dim);

    template <typename Column>
    diameter_index_t_struct init_coboundary_and_get_pivot_fullmatrix(const diameter_index_t_struct simplex,
                                                                     Column& working_coboundary, const index_t& dim
            , hash_map<index_t, index_t>& pivot_column_index) {
        bool check_for_emergent_pair= true;
        cofacet_entries.clear();
        simplex_coboundary_enumerator cofacets(simplex, dim, *this);
        while (cofacets.has_next()) {
            diameter_index_t_struct cofacet= cofacets.next();
            if (cofacet.diameter <= threshold) {
                cofacet_entries.push_back(cofacet);
                if (check_for_emergent_pair && (simplex.diameter == cofacet.diameter)) {
                    if (pivot_column_index.find(cofacet.index) == pivot_column_index.end()){
                        return cofacet;
                    }
                    check_for_emergent_pair= false;
                }
            }
        }
        for (auto cofacet : cofacet_entries) working_coboundary.push(cofacet);
        return get_pivot(working_coboundary);
    }

    template <typename Column>
    diameter_index_t_struct init_coboundary_and_get_pivot_submatrix(const diameter_index_t_struct simplex,
                                                                    Column& working_coboundary, index_t dim, struct row_cidx_column_idx_struct_compare cmp) {
        bool check_for_emergent_pair= true;
        cofacet_entries.clear();
        simplex_coboundary_enumerator cofacets(simplex, dim, *this);
        while (cofacets.has_next()) {
            diameter_index_t_struct cofacet= cofacets.next();
            if (cofacet.diameter <= threshold) {
                cofacet_entries.push_back(cofacet);
                if (check_for_emergent_pair && (simplex.diameter == cofacet.diameter)) {
                    if(get_value_pivot_array_hashmap(cofacet.index, cmp)==-1) {
                        return cofacet;
                    }
                    check_for_emergent_pair= false;
                }
            }
        }
        for (auto cofacet : cofacet_entries) working_coboundary.push(cofacet);
        return get_pivot(working_coboundary);
    }

    template <typename Column>
    void add_simplex_coboundary_oblivious(const diameter_index_t_struct simplex, const index_t& dim,
                                          Column& working_coboundary) {
        simplex_coboundary_enumerator cofacets(simplex, dim, *this);
        while (cofacets.has_next()) {
            diameter_index_t_struct cofacet= cofacets.next();
            if (cofacet.diameter <= threshold) working_coboundary.push(cofacet);
        }
    }

    template <typename Column>
    void add_simplex_coboundary_use_reduction_column(const diameter_index_t_struct simplex, const index_t& dim,
                                                     Column& working_reduction_column, Column& working_coboundary) {
        working_reduction_column.push(simplex);
        simplex_coboundary_enumerator cofacets(simplex, dim, *this);
        while (cofacets.has_next()) {
            diameter_index_t_struct cofacet= cofacets.next();
            if (cofacet.diameter <= threshold) working_coboundary.push(cofacet);
        }
    }

    //THIS IS THE METHOD TO CALL FOR CPU SIDE FULL MATRIX REDUCTION
    template <typename Column>
    void add_coboundary_fullmatrix(compressed_sparse_matrix<diameter_index_t_struct>& reduction_matrix,
                                   const std::vector<diameter_index_t_struct>& columns_to_reduce,
                                   const size_t index_column_to_add, const size_t& dim,
                                   Column& working_reduction_column, Column& working_coboundary) {
        diameter_index_t_struct column_to_add= columns_to_reduce[index_column_to_add];
        add_simplex_coboundary_use_reduction_column(column_to_add, dim, working_reduction_column, working_coboundary);

        for (diameter_index_t_struct simplex : reduction_matrix.subrange(index_column_to_add)) {
            add_simplex_coboundary_use_reduction_column(simplex, dim, working_reduction_column, working_coboundary);
        }
    }

    //THIS IS THE METHOD TO CALL FOR SUBMATRIX REDUCTION ON CPU SIDE
#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
    template <typename Column>
    void add_coboundary_reduction_submatrix(compressed_sparse_submatrix<diameter_index_t_struct>& reduction_submatrix,
                                            const size_t index_column_to_add, const size_t& dim,
                                            Column& working_reduction_column, Column& working_coboundary) {
        diameter_index_t_struct column_to_add= h_columns_to_reduce[index_column_to_add];
        add_simplex_coboundary_use_reduction_column(column_to_add, dim, working_reduction_column, working_coboundary);
        index_t subindex= h_flagarray_OR_index_to_subindex[index_column_to_add];//this is only defined when ASSEMBLE_REDUCTION_SUBMATRIX is defined
        if(subindex>-1) {
            for (diameter_index_t_struct simplex : reduction_submatrix.subrange(subindex)) {
                add_simplex_coboundary_use_reduction_column(simplex, dim, working_reduction_column, working_coboundary);
            }
        }
    }
#endif

    void compute_pairs(std::vector<diameter_index_t_struct>& columns_to_reduce,
                       hash_map<index_t, index_t>& pivot_column_index, index_t dim) {

#ifdef PRINT_PERSISTENCE_PAIRS
        std::cout << "persistence intervals in dim " << dim << ":" << std::endl;
#endif
#ifdef CPUONLY_ASSEMBLE_REDUCTION_MATRIX
        compressed_sparse_matrix<diameter_index_t_struct> reduction_matrix;
#endif
        for (index_t index_column_to_reduce= 0; index_column_to_reduce < columns_to_reduce.size();
             ++index_column_to_reduce) {

            auto column_to_reduce= columns_to_reduce[index_column_to_reduce];

            std::priority_queue<diameter_index_t_struct, std::vector<diameter_index_t_struct>,
                    greaterdiam_lowerindex_diameter_index_t_struct_compare>
#ifdef CPUONLY_ASSEMBLE_REDUCTION_MATRIX
            working_reduction_column,
#endif
                    working_coboundary;

            value_t diameter= column_to_reduce.diameter;

#ifdef INDICATE_PROGRESS
            if ((index_column_to_reduce + 1) % 1000000 == 0)
				std::cerr << "\033[K"
				          << "reducing column " << index_column_to_reduce + 1 << "/"
				          << columns_to_reduce.size() << " (diameter " << diameter << ")"
				          << std::flush << "\r";
#endif

            index_t index_column_to_add= index_column_to_reduce;

            diameter_index_t_struct pivot;

            // initialize index bounds of reduction matrix
#ifdef CPUONLY_ASSEMBLE_REDUCTION_MATRIX
            reduction_matrix.append_column();
#endif
            pivot= init_coboundary_and_get_pivot_fullmatrix(columns_to_reduce[index_column_to_add], working_coboundary, dim, pivot_column_index);

            while (true) {
                if(pivot.index!=-1){
                    auto left_pair= pivot_column_index.find(pivot.index);
                    if (left_pair != pivot_column_index.end()) {
                        index_column_to_add= left_pair->second;
#ifdef CPUONLY_ASSEMBLE_REDUCTION_MATRIX
                        add_coboundary_fullmatrix(reduction_matrix, columns_to_reduce, index_column_to_add, dim, working_reduction_column, working_coboundary);
                        pivot= get_pivot(working_coboundary);
#else
                        add_simplex_coboundary_oblivious(columns_to_reduce[index_column_to_add], dim, working_coboundary);
                        pivot= get_pivot(working_coboundary);
#endif
                    } else {
#if defined(PRINT_PERSISTENCE_PAIRS) || defined(PYTHON_BARCODE_COLLECTION)
                        value_t death= pivot.diameter;
                        if (death > diameter * ratio) {
#ifdef INDICATE_PROGRESS
                            std::cerr << "\033[K";
#endif
#ifdef PRINT_PERSISTENCE_PAIRS
                            std::cout << " [" << diameter << "," << death << ")" << std::endl
                                      << std::flush;
#endif
                            birth_death_coordinate barcode = {diameter,death};
                            list_of_barcodes[dim].push_back(barcode);
                        }
#endif
                        pivot_column_index[pivot.index]= index_column_to_reduce;

                        break;
                    }
                } else {
#ifdef PRINT_PERSISTENCE_PAIRS
                    std::cout << " [" << diameter << ", )" << std::endl << std::flush;
#endif
                    break;
                }
            }
        }
    }

    void compute_pairs_plusplus(
            index_t dim,
            index_t gpuscan_startingdim) {

#ifdef PRINT_PERSISTENCE_PAIRS
        std::cout << "persistence intervals in dim " << dim << ":" << std::endl;
#endif
#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
        compressed_sparse_submatrix<diameter_index_t_struct> reduction_submatrix;
#endif
#ifdef INDICATE_PROGRESS
        std::chrono::steady_clock::time_point next= std::chrono::steady_clock::now() + time_step;
#endif
        struct row_cidx_column_idx_struct_compare cmp_pivots;
        index_t num_columns_to_iterate= *h_num_columns_to_reduce;
        if(dim>=gpuscan_startingdim){
            num_columns_to_iterate= *h_num_nonapparent;
        }
        for (index_t sub_index_column_to_reduce= 0; sub_index_column_to_reduce < num_columns_to_iterate;
             ++sub_index_column_to_reduce) {
            index_t index_column_to_reduce =sub_index_column_to_reduce;
            if(dim>=gpuscan_startingdim) {
                index_column_to_reduce= h_pivot_column_index_array_OR_nonapparent_cols[sub_index_column_to_reduce];//h_nonapparent_cols
            }
            auto column_to_reduce= h_columns_to_reduce[index_column_to_reduce];

            std::priority_queue<diameter_index_t_struct, std::vector<diameter_index_t_struct>,
                    greaterdiam_lowerindex_diameter_index_t_struct_compare>
#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
                    working_reduction_column,
#endif
                    working_coboundary;

            value_t diameter= column_to_reduce.diameter;

            index_t index_column_to_add= index_column_to_reduce;

            struct diameter_index_t_struct pivot;
#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
            reduction_submatrix.append_column();
#endif
            pivot= init_coboundary_and_get_pivot_submatrix(column_to_reduce, working_coboundary, dim, cmp_pivots);

            while (true) {
#ifdef INDICATE_PROGRESS
                //if(sub_index_column_to_reduce%2==0){
                if (std::chrono::steady_clock::now() > next) {
                    std::cerr<< clear_line << "reducing column " << index_column_to_reduce + 1
                              << "/" << *h_num_columns_to_reduce << " (diameter " << diameter << ")"
                              << std::flush;
                    next= std::chrono::steady_clock::now() + time_step;
                }
#endif
                if(pivot.index!=-1){

                    index_column_to_add= get_value_pivot_array_hashmap(pivot.index,cmp_pivots);
                    if(index_column_to_add!=-1) {
#ifdef ASSEMBLE_REDUCTION_SUBMATRIX

                        add_coboundary_reduction_submatrix(reduction_submatrix, index_column_to_add,
                                                           dim, working_reduction_column, working_coboundary);

                        pivot= get_pivot(working_coboundary);
#else
                        add_simplex_coboundary_oblivious(h_columns_to_reduce[index_column_to_add], dim, working_coboundary);
                        pivot= get_pivot(working_coboundary);
#endif

                    }else{
#if defined(PRINT_PERSISTENCE_PAIRS) || defined(PYTHON_BARCODE_COLLECTION)
                        value_t death= pivot.diameter;
                        if (death > diameter * ratio) {
#ifdef INDICATE_PROGRESS
                            std::cerr << clear_line << std::flush;
#endif

#ifdef PRINT_PERSISTENCE_PAIRS
                            std::cout << " [" << diameter << "," << death << ")" << std::endl
                                      << std::flush;
#endif
                            birth_death_coordinate barcode = {diameter,death};
                            list_of_barcodes[dim].push_back(barcode);
                        }
#endif

#ifdef USE_PHASHMAP
                        phmap_put(pivot.index, index_column_to_reduce);
#endif
#ifdef USE_GOOGLE_HASHMAP
                        pivot_column_index[pivot.index]= index_column_to_reduce;
#endif
#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
                        while (true) {
                            diameter_index_t_struct e= pop_pivot(working_reduction_column);
                            if (e.index == -1) break;
                            reduction_submatrix.push_back(e);
                        }
#endif
                        break;
                    }
                } else {

#ifdef PRINT_PERSISTENCE_PAIRS
                    #ifdef INDICATE_PROGRESS
                    std::cerr << clear_line << std::flush;
#endif
                    std::cout << " [" << diameter << ", )" << std::endl << std::flush;
#endif
                    break;
                }
            }
        }
#ifdef INDICATE_PROGRESS
        std::cerr << clear_line << std::flush;
#endif
    }


    std::vector<diameter_index_t_struct> get_edges();
    void compute_barcodes();
};


template<>
class ripser<compressed_lower_distance_matrix>::simplex_coboundary_enumerator {
private:
    index_t idx_below, idx_above, v, k;
    std::vector<index_t> vertices;
    ///const diameter_index_t simplex;
    const struct diameter_index_t_struct simplex;
    const compressed_lower_distance_matrix& dist;
    const binomial_coeff_table& binomial_coeff;

public:

    simplex_coboundary_enumerator(
            const struct diameter_index_t_struct _simplex, index_t _dim,
            const ripser<compressed_lower_distance_matrix>& parent)
            : idx_below(_simplex.index),
              idx_above(0), v(parent.n - 1), k(_dim + 1),
              vertices(_dim + 1), simplex(_simplex), dist(parent.dist),
              binomial_coeff(parent.binomial_coeff) {
        parent.get_simplex_vertices(_simplex.index, _dim, parent.n, vertices.begin());

    }

    bool has_next(bool all_cofacets= true) {
        return (v >= k && (all_cofacets || binomial_coeff(v, k) > idx_below));//second condition after the || is to ensure iteration of cofacets with no need to adjust
    }

    struct diameter_index_t_struct next() {
        while ((binomial_coeff(v, k) <= idx_below)) {
            idx_below -= binomial_coeff(v, k);
            idx_above += binomial_coeff(v, k + 1);
            --v;
            --k;
            assert(k != -1);
        }
        value_t cofacet_diameter= simplex.diameter;
        for (index_t w : vertices) cofacet_diameter= std::max(cofacet_diameter, dist(v, w));
        index_t cofacet_index= idx_above + binomial_coeff(v--, k + 1) + idx_below;
        return {cofacet_diameter, cofacet_index};
    }
};

template <> class ripser<sparse_distance_matrix>::simplex_coboundary_enumerator {
    const ripser& parent;
    index_t idx_below, idx_above, k;
    std::vector<index_t> vertices;
    const diameter_index_t_struct simplex;
    const sparse_distance_matrix& dist;
    std::vector<std::vector<index_diameter_t_struct>::const_reverse_iterator>& neighbor_it;
    std::vector<std::vector<index_diameter_t_struct>::const_reverse_iterator>& neighbor_end;
    index_diameter_t_struct neighbor;

public:
    simplex_coboundary_enumerator(const diameter_index_t_struct _simplex, const index_t _dim,
                                  const ripser& _parent)
            : parent(_parent), idx_below(_simplex.index), idx_above(0), k(_dim + 1),
              vertices(_dim + 1), simplex(_simplex),
              dist(parent.dist),
              neighbor_it(dist.neighbor_it),
              neighbor_end(dist.neighbor_end) {
        neighbor_it.clear();
        neighbor_end.clear();

        parent.get_simplex_vertices(idx_below, _dim, parent.n, vertices.rbegin());
        for (auto v : vertices) {
            neighbor_it.push_back(dist.neighbors[v].rbegin());
            neighbor_end.push_back(dist.neighbors[v].rend());
        }
    }

    bool has_next(bool all_cofacets= true) {
        //auto& x will permanently change upon updates to it.
        for (auto &it0= neighbor_it[0], &end0= neighbor_end[0]; it0 != end0; ++it0) {
            neighbor= *it0;//neighbor is a pair: diameter_index_t_struct
            for (size_t idx= 1; idx < neighbor_it.size(); ++idx) {
                auto &it= neighbor_it[idx], end= neighbor_end[idx];
                //enforce the invariant that get_index(*it)<=get_index(neighbor)
                while(it->index > neighbor.index)
                    if (++it == end) return false;
                if(it->index != neighbor.index)
                    goto continue_outer;//try the next number in neighbor_it[0]
                else
                    //update neighbor to the max of matching vertices of "neighbors" of each vertex in simplex
                    neighbor= (neighbor.diameter>it->diameter)?neighbor:*it;
            }
            while(k>0 && vertices[k-1]>neighbor.index){
                if (!all_cofacets) return false;
                idx_below -= parent.binomial_coeff(vertices[k - 1], k);
                idx_above += parent.binomial_coeff(vertices[k - 1], k + 1);
                --k;
            }
            return true;
            continue_outer:;
        }
        return false;
    }

    diameter_index_t_struct next() {
        ++neighbor_it[0];
        value_t cofacet_diameter= std::max(simplex.diameter, neighbor.diameter);
        index_t cofacet_index= idx_above+parent.binomial_coeff(neighbor.index,k+1)+idx_below;

        return {cofacet_diameter,cofacet_index};
    }
};

template<> std::vector<diameter_index_t_struct> ripser<compressed_lower_distance_matrix>::get_edges() {
    std::vector<diameter_index_t_struct> edges;
    for (index_t index= binomial_coeff(n, 2); index-- > 0;) {
        value_t diameter= compute_diameter(index, 1);
        if (diameter <= threshold) edges.push_back({diameter, index});
    }
    return edges;
}

template <> std::vector<diameter_index_t_struct> ripser<sparse_distance_matrix>::get_edges() {
    std::vector<diameter_index_t_struct> edges;
    for (index_t i= 0; i < n; ++i)
        for (auto nbr : dist.neighbors[i]) {
            index_t j= nbr.index;
            //(i choose 2) + (j choose 1) is the combinatorial index of nbr
            if (i > j) edges.push_back({nbr.diameter, binomial_coeff(i, 2) + j});
        }
    return edges;
}

template <>
void ripser<compressed_lower_distance_matrix>::gpu_compute_dim_0_pairs(std::vector<struct diameter_index_t_struct>& columns_to_reduce
){
    union_find dset(n);

    index_t max_num_edges= binomial_coeff(n, 2);
    struct greaterdiam_lowerindex_diameter_index_t_struct_compare_reverse cmp_reverse;
#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
    cudaMemset(d_flagarray_OR_index_to_subindex, 0, sizeof(index_t)*max_num_edges);
    CUDACHECK(cudaDeviceSynchronize());
#else
    cudaMemset(d_flagarray, 0, sizeof(char)*max_num_edges);
    CUDACHECK(cudaDeviceSynchronize());
#endif

#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, populate_edges<index_t>, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;
    populate_edges<<<grid_size, 256>>>(d_flagarray_OR_index_to_subindex, d_columns_to_reduce, threshold, d_distance_matrix, max_num_edges, n, d_binomial_coeff);
    CUDACHECK(cudaDeviceSynchronize());

    *h_num_columns_to_reduce= thrust::count(thrust::device , d_flagarray_OR_index_to_subindex, d_flagarray_OR_index_to_subindex+max_num_edges, 1);
    CUDACHECK(cudaDeviceSynchronize());
    thrust::sort(thrust::device, d_columns_to_reduce, d_columns_to_reduce+ max_num_edges, cmp_reverse);
#else
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, populate_edges<char>, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;
    populate_edges<<<grid_size, 256>>>(d_flagarray, d_columns_to_reduce, threshold, d_distance_matrix, max_num_edges, n, d_binomial_coeff);
    CUDACHECK(cudaDeviceSynchronize());

    *h_num_columns_to_reduce= thrust::count(thrust::device , d_flagarray, d_flagarray+max_num_edges, 1);
    CUDACHECK(cudaDeviceSynchronize());
    thrust::sort(thrust::device, d_columns_to_reduce, d_columns_to_reduce+ max_num_edges, cmp_reverse);
#endif
#ifdef COUNTING
    std::cerr<<"num edges filtered by diameter: "<<*h_num_columns_to_reduce<<std::endl;
#endif

    cudaMemcpy(h_columns_to_reduce, d_columns_to_reduce, sizeof(struct diameter_index_t_struct)*(*h_num_columns_to_reduce), cudaMemcpyDeviceToHost);

#ifdef PRINT_PERSISTENCE_PAIRS
    std::cout << "persistence intervals in dim 0:" << std::endl;
#endif

    std::vector<index_t> vertices_of_edge(2);
    for(index_t idx=0; idx<*h_num_columns_to_reduce; idx++){
        struct diameter_index_t_struct e= h_columns_to_reduce[idx];
        vertices_of_edge.clear();
        get_simplex_vertices(e.index, 1, n, std::back_inserter(vertices_of_edge));
        index_t u= dset.find(vertices_of_edge[0]), v= dset.find(vertices_of_edge[1]);

        if (u != v) {
#if defined(PRINT_PERSISTENCE_PAIRS) || defined(PYTHON_BARCODE_COLLECTION)
            //remove paired destroyer columns (we compute cohomology)
            if(e.diameter!=0) {
#ifdef PRINT_PERSISTENCE_PAIRS
                std::cout << " [0," << e.diameter << ")" << std::endl;
#endif
                birth_death_coordinate barcode = {0,e.diameter};
                list_of_barcodes[0].push_back(barcode);
            }
#endif
            dset.link(u, v);
        } else {
            columns_to_reduce.push_back(e);
        }
    }
    std::reverse(columns_to_reduce.begin(), columns_to_reduce.end());
    //don't want to reverse the h_columns_to_reduce so just put into vector and copy later
#pragma omp parallel for schedule(guided,1)
    for(index_t i=0; i<columns_to_reduce.size(); i++){
        h_columns_to_reduce[i]= columns_to_reduce[i];
    }
    *h_num_columns_to_reduce= columns_to_reduce.size();
    *h_num_nonapparent= *h_num_columns_to_reduce;//we haven't found any apparent columns yet, so set all columns to nonapparent

#ifdef PRINT_PERSISTENCE_PAIRS
    for (index_t i= 0; i < n; ++i)
        if (dset.find(i) == i) std::cout << " [0, )" << std::endl << std::flush;
#endif
#ifdef COUNTING
    std::cerr<<"num cols to reduce: dim 1, "<<*h_num_columns_to_reduce<<std::endl;
#endif
}

template <>
void ripser<sparse_distance_matrix>::gpu_compute_dim_0_pairs(std::vector<struct diameter_index_t_struct>& columns_to_reduce
){
    union_find dset(n);

    struct greaterdiam_lowerindex_diameter_index_t_struct_compare_reverse cmp_reverse;
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, populate_sparse_edges_preparingcount, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;

    //grid_size will return 0 if we have CPU-only code inside d_CSR_distance_matrix
    *h_num_simplices= 0;

    //populate edges kernel cannot have some threads iterating in the inner for loop, preventing shfl_sync() from runnning

    int* d_num;
    CUDACHECK(cudaMalloc((void **) & d_num, sizeof(int)*(n+1)));
    cudaMemset(d_num, 0, sizeof(int)*(n+1));

    populate_sparse_edges_preparingcount<<<grid_size, 256>>>(d_num, d_CSR_distance_matrix, n, d_num_simplices);
    CUDACHECK(cudaDeviceSynchronize());
    thrust::exclusive_scan(thrust::device, d_num, d_num+n+1, d_num, 0);

    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, populate_sparse_edges_prefixsum, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;

    populate_sparse_edges_prefixsum<<<grid_size,256>>>(d_simplices, d_num, d_CSR_distance_matrix, d_binomial_coeff, n, d_num_simplices);
    CUDACHECK(cudaDeviceSynchronize());

    thrust::sort(thrust::device, d_simplices, d_simplices+ *h_num_simplices, cmp_reverse);
    CUDACHECK(cudaDeviceSynchronize());

#ifdef COUNTING
    std::cerr<<"num (sparse) edges filtered: "<<*h_num_simplices<<std::endl;
#endif

    cudaMemcpy(h_simplices, d_simplices, sizeof(struct diameter_index_t_struct)*(*h_num_simplices), cudaMemcpyDeviceToHost);

#ifdef PRINT_PERSISTENCE_PAIRS
    std::cout << "persistence intervals in dim 0:" << std::endl;
#endif


    std::vector<index_t> vertices_of_edge(2);
    for(index_t idx=0; idx<*h_num_simplices; idx++){
        struct diameter_index_t_struct e= h_simplices[idx];
        vertices_of_edge.clear();
        get_simplex_vertices(e.index, 1, n, std::back_inserter(vertices_of_edge));
        index_t u= dset.find(vertices_of_edge[0]), v= dset.find(vertices_of_edge[1]);

        if (u != v) {
#if defined(PRINT_PERSISTENCE_PAIRS) || defined(PYTHON_BARCODE_COLLECTION)
            if(e.diameter!=0) {
#ifdef PRINT_PERSISTENCE_PAIRS
                std::cout << " [0," << e.diameter << ")" << std::endl;
#endif
                birth_death_coordinate barcode = {0,e.diameter};
                list_of_barcodes[0].push_back(barcode);
            }
#endif
            dset.link(u, v);
        } else {
            columns_to_reduce.push_back(e);
        }
    }
    std::reverse(columns_to_reduce.begin(), columns_to_reduce.end());
    //don't want to reverse the h_columns_to_reduce so just put into vector and copy later
#pragma omp parallel for schedule(guided,1)
    for(index_t i=0; i<columns_to_reduce.size(); i++){
        h_columns_to_reduce[i]= columns_to_reduce[i];
    }
    *h_num_columns_to_reduce= columns_to_reduce.size();
    *h_num_nonapparent= *h_num_columns_to_reduce;//we haven't found any apparent columns yet, so set all columns to nonapparent
#ifdef PRINT_PERSISTENCE_PAIRS
    for (index_t i= 0; i < n; ++i)
        if (dset.find(i) == i) std::cout << " [0, )" << std::endl << std::flush;
#endif
#ifdef COUNTING
    std::cerr<<"num cols to reduce: dim 1, "<<*h_num_columns_to_reduce<<std::endl;
#endif
}
//finding apparent pairs
template <>
void ripser<compressed_lower_distance_matrix>::gpuscan(const index_t dim){
    //(need to sort for filtration order before gpuscan first, then apply gpu scan then sort again)
    //note: scan kernel can eliminate high percentage of columns in little time.
    //filter by fully reduced columns (apparent pairs) found by gpu scan

    //need this to prevent 0-blocks kernels from executing
    if(*h_num_columns_to_reduce==0){
        return;
    }
    index_t num_simplices= binomial_coeff(n,dim+1);
#ifdef COUNTING
    std::cerr<<"max possible num simplices: "<<num_simplices<<std::endl;
#endif


    cudaMemcpy(d_columns_to_reduce, h_columns_to_reduce,
               sizeof(struct diameter_index_t_struct) * *h_num_columns_to_reduce, cudaMemcpyHostToDevice);

    CUDACHECK(cudaDeviceSynchronize());

    thrust::fill(thrust::device, d_cidx_to_diameter, d_cidx_to_diameter + num_simplices, -MAX_FLOAT);
    CUDACHECK(cudaDeviceSynchronize());

    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, init_cidx_to_diam, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;
    //there will be kernel launch errors if columns_to_reduce.size()==0; it causes thrust to complain later in the code execution

    init_cidx_to_diam << < grid_size, 256 >> >
    (d_cidx_to_diameter, d_columns_to_reduce, *h_num_columns_to_reduce);

    CUDACHECK(cudaDeviceSynchronize());

    cudaMemset(d_lowest_one_of_apparent_pair, -1, sizeof(index_t) * *h_num_columns_to_reduce);
    CUDACHECK(cudaDeviceSynchronize());

    Stopwatch sw;
    sw.start();
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, coboundary_findapparent_single_kernel, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;

    coboundary_findapparent_single_kernel << < grid_size, 256, 256 * (dim + 1) * sizeof(index_t) >> >
    (d_cidx_to_diameter, d_columns_to_reduce, d_lowest_one_of_apparent_pair, dim, num_simplices, n, d_binomial_coeff, *h_num_columns_to_reduce, d_distance_matrix, threshold);

    CUDACHECK(cudaDeviceSynchronize());
    sw.stop();
#ifdef PROFILING
    std::cerr<<"gpu scan kernel time for dim: "<<dim<<": "<<sw.ms()/1000.0<<"s"<<std::endl;
#endif

    CUDACHECK(cudaDeviceSynchronize());

    //post processing (inserting appararent pairs into a "hash map": 2 level data structure) now on GPU
    Stopwatch postprocessing;
    postprocessing.start();
    struct row_cidx_column_idx_struct_compare cmp_pivots;

    //put pairs into an array

    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, gpu_insert_pivots_kernel, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;

    gpu_insert_pivots_kernel<< < grid_size, 256 >> >(d_pivot_array, d_lowest_one_of_apparent_pair, d_pivot_column_index_OR_nonapparent_cols, *h_num_columns_to_reduce, d_num_nonapparent);
    CUDACHECK(cudaDeviceSynchronize());

    thrust::sort(thrust::device, d_pivot_array, d_pivot_array+*h_num_columns_to_reduce, cmp_pivots);
    thrust::sort(thrust::device, d_pivot_column_index_OR_nonapparent_cols, d_pivot_column_index_OR_nonapparent_cols+*h_num_nonapparent);

    num_apparent= *h_num_columns_to_reduce-*h_num_nonapparent;
#ifdef COUNTING
    std::cerr<<"num apparent for dim: "<<dim<<" is: " <<num_apparent<<std::endl;
#endif
    //transfer to CPU side all GPU data structures
    cudaMemcpy(h_pivot_array, d_pivot_array, sizeof(index_t_pair_struct)*(num_apparent), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pivot_column_index_array_OR_nonapparent_cols, d_pivot_column_index_OR_nonapparent_cols, sizeof(index_t)*(*h_num_nonapparent), cudaMemcpyDeviceToHost);


#ifdef ASSEMBLE_REDUCTION_SUBMATRIX

    cudaMemset(d_flagarray_OR_index_to_subindex, -1, sizeof(index_t)* *h_num_columns_to_reduce);
    //perform the scatter operation
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, init_index_to_subindex, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;
    init_index_to_subindex<< < grid_size, 256 >> >
    (d_flagarray_OR_index_to_subindex, d_pivot_column_index_OR_nonapparent_cols, *h_num_nonapparent);
    cudaMemcpy(h_flagarray_OR_index_to_subindex, d_flagarray_OR_index_to_subindex, sizeof(index_t)*(*h_num_columns_to_reduce), cudaMemcpyDeviceToHost);
#endif
    postprocessing.stop();
#ifdef PROFILING
    std::cerr<<"INSERTION POSTPROCESSING FOR GPU IN DIM "<<dim<<": "<<postprocessing.ms()/1000.0<<"s"<<std::endl;
#endif
}

//finding apparent pairs
template <>
void ripser<sparse_distance_matrix>::gpuscan(const index_t dim){
    //(need to sort for filtration order before gpuscan first, then apply gpu scan then sort again)
    //note: scan kernel can eliminate high percentage of columns in little time.
    //filter by fully reduced columns (apparent pairs) found by gpu scan

    //need this to prevent 0-blocks kernels from executing
    if(*h_num_columns_to_reduce==0){
        return;
    }
    index_t num_simplices= binomial_coeff(n,dim+1);
#ifdef COUNTING
    std::cerr<<"max possible num simplices: "<<num_simplices<<std::endl;
#endif

    cudaMemcpy(d_columns_to_reduce, h_columns_to_reduce,
               sizeof(struct diameter_index_t_struct) * *h_num_columns_to_reduce, cudaMemcpyHostToDevice);

    CUDACHECK(cudaDeviceSynchronize());

    //use binary search on d_columns_to_reduce as retrival process

    cudaMemcpy(d_cidx_diameter_pairs_sortedlist, d_columns_to_reduce, sizeof(struct diameter_index_t_struct)*(*h_num_columns_to_reduce), cudaMemcpyDeviceToDevice);
    struct lowerindex_lowerdiam_diameter_index_t_struct_compare cmp_cidx_diameter;
    thrust::sort(thrust::device, d_cidx_diameter_pairs_sortedlist, d_cidx_diameter_pairs_sortedlist+*h_num_columns_to_reduce, cmp_cidx_diameter);
    CUDACHECK(cudaDeviceSynchronize());

    cudaMemset(d_lowest_one_of_apparent_pair, -1, sizeof(index_t) * *h_num_columns_to_reduce);
    CUDACHECK(cudaDeviceSynchronize());

    Stopwatch sw;
    sw.start();
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, coboundary_findapparent_sparse_single_kernel, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;
    coboundary_findapparent_sparse_single_kernel << < grid_size, 256, 256 * (dim + 1) * sizeof(index_t) >> >
    (d_cidx_diameter_pairs_sortedlist, d_columns_to_reduce, d_lowest_one_of_apparent_pair, dim, n, d_binomial_coeff, *h_num_columns_to_reduce, d_CSR_distance_matrix, threshold);

    CUDACHECK(cudaDeviceSynchronize());
    sw.stop();
#ifdef PROFILING
    std::cerr<<"gpu scan kernel time for dim: "<<dim<<": "<<sw.ms()/1000.0<<"s"<<std::endl;
#endif
    CUDACHECK(cudaDeviceSynchronize());

    //post processing (inserting appararent pairs into a "hash map": 2 level data structure) now on GPU
    Stopwatch postprocessing;
    postprocessing.start();
    struct row_cidx_column_idx_struct_compare cmp_pivots;

    //put pairs into an array

    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, gpu_insert_pivots_kernel, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;
    gpu_insert_pivots_kernel<< < grid_size, 256 >> >(d_pivot_array, d_lowest_one_of_apparent_pair, d_pivot_column_index_OR_nonapparent_cols, *h_num_columns_to_reduce, d_num_nonapparent);
    CUDACHECK(cudaDeviceSynchronize());
    thrust::sort(thrust::device, d_pivot_array, d_pivot_array+*h_num_columns_to_reduce, cmp_pivots);
    thrust::sort(thrust::device, d_pivot_column_index_OR_nonapparent_cols, d_pivot_column_index_OR_nonapparent_cols+*h_num_nonapparent);

    num_apparent= *h_num_columns_to_reduce-*h_num_nonapparent;
#ifdef COUNTING
    std::cerr<<"num apparent for dim: "<<dim<<" is: "<<num_apparent<<std::endl;
#endif
    //transfer to CPU side all GPU data structures
    cudaMemcpy(h_pivot_array, d_pivot_array, sizeof(index_t_pair_struct)*(num_apparent), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pivot_column_index_array_OR_nonapparent_cols, d_pivot_column_index_OR_nonapparent_cols, sizeof(index_t)*(*h_num_nonapparent), cudaMemcpyDeviceToHost);

#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
    cudaMemset(d_flagarray_OR_index_to_subindex, -1, sizeof(index_t)* *h_num_columns_to_reduce);
    //perform the scatter operation
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, init_index_to_subindex, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;
    init_index_to_subindex<< < grid_size, 256 >> >
    (d_flagarray_OR_index_to_subindex, d_pivot_column_index_OR_nonapparent_cols, *h_num_nonapparent);
    cudaMemcpy(h_flagarray_OR_index_to_subindex, d_flagarray_OR_index_to_subindex, sizeof(index_t)*(*h_num_columns_to_reduce), cudaMemcpyDeviceToHost);
#endif
    postprocessing.stop();
#ifdef PROFILING
    std::cerr<<"INSERTION POSTPROCESSING FOR GPU IN DIM "<<dim<<": "<<postprocessing.ms()/1000.0<<"s"<<std::endl;
#endif
}

template <>
void ripser<compressed_lower_distance_matrix>::gpu_assemble_columns_to_reduce_plusplus(const index_t dim) {

    index_t max_num_simplices= binomial_coeff(n, dim + 1);

    Stopwatch sw;
    sw.start();

#pragma omp parallel for schedule(guided,1)
    for (index_t i= 0; i < max_num_simplices; i++) {
#ifdef USE_PHASHMAP
        h_pivot_column_index_array_OR_nonapparent_cols[i]= phmap_get_value(i);
#endif
#ifdef USE_GOOGLE_HASHMAP
        auto pair= pivot_column_index.find(i);
        if(pair!=pivot_column_index.end()){
            h_pivot_column_index_array_OR_nonapparent_cols[i]= pair->second;
        }else{
            h_pivot_column_index_array_OR_nonapparent_cols[i]= -1;
        }
#endif
    }
    num_apparent= *h_num_columns_to_reduce-*h_num_nonapparent;
    if(num_apparent>0) {
#pragma omp parallel for schedule(guided, 1)
        for (index_t i= 0; i < num_apparent; i++) {
            index_t row_cidx= h_pivot_array[i].row_cidx;
            h_pivot_column_index_array_OR_nonapparent_cols[row_cidx]= h_pivot_array[i].column_idx;
        }
    }
    *h_num_columns_to_reduce= 0;
    cudaMemcpy(d_pivot_column_index_OR_nonapparent_cols, h_pivot_column_index_array_OR_nonapparent_cols, sizeof(index_t)*max_num_simplices, cudaMemcpyHostToDevice);

    sw.stop();
#ifdef PROFILING
    std::cerr<<"time to copy hash map for dim "<<dim<<": "<<sw.ms()/1000.0<<"s"<<std::endl;
#endif
#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
    cudaMemset(d_flagarray_OR_index_to_subindex, 0, sizeof(index_t)*max_num_simplices);
    CUDACHECK(cudaDeviceSynchronize());
#else
    cudaMemset(d_flagarray, 0, sizeof(char)*max_num_simplices);
    CUDACHECK(cudaDeviceSynchronize());
#endif
    Stopwatch pop_cols_timer;
    pop_cols_timer.start();

#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, populate_columns_to_reduce<index_t>, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;
    populate_columns_to_reduce<<<grid_size, 256, 256 * (dim + 1) * sizeof(index_t)>>>(d_flagarray_OR_index_to_subindex, d_columns_to_reduce, d_pivot_column_index_OR_nonapparent_cols, d_distance_matrix, n, max_num_simplices, dim, threshold, d_binomial_coeff);
#else
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, populate_columns_to_reduce<char>, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;
    populate_columns_to_reduce<<<grid_size, 256, 256 * (dim + 1) * sizeof(index_t)>>>(d_flagarray, d_columns_to_reduce, d_pivot_column_index_OR_nonapparent_cols, d_distance_matrix, n, max_num_simplices, dim, threshold, d_binomial_coeff);
#endif
    CUDACHECK(cudaDeviceSynchronize());
    pop_cols_timer.stop();

    struct greaterdiam_lowerindex_diameter_index_t_struct_compare cmp;

#ifdef ASSEMBLE_REDUCTION_SUBMATRIX
    *h_num_columns_to_reduce= thrust::count(thrust::device , d_flagarray_OR_index_to_subindex, d_flagarray_OR_index_to_subindex+max_num_simplices, 1);
    CUDACHECK(cudaDeviceSynchronize());
    thrust::sort(thrust::device, d_columns_to_reduce, d_columns_to_reduce+ max_num_simplices, cmp);
#else
    *h_num_columns_to_reduce= thrust::count(thrust::device , d_flagarray, d_flagarray+max_num_simplices, 1);
    CUDACHECK(cudaDeviceSynchronize());
    thrust::sort(thrust::device, d_columns_to_reduce, d_columns_to_reduce+ max_num_simplices, cmp);
#endif

#ifdef COUNTING
    std::cerr<<"num cols to reduce for dim "<<dim<<": "<<*h_num_columns_to_reduce<<std::endl;
#endif
    cudaMemcpy(h_columns_to_reduce, d_columns_to_reduce, sizeof(struct diameter_index_t_struct)*(*h_num_columns_to_reduce), cudaMemcpyDeviceToHost);

}

template <>
void ripser<sparse_distance_matrix>::gpu_assemble_columns_to_reduce_plusplus(const index_t dim) {
    index_t max_num_simplices= binomial_coeff(n,dim+1);
#ifdef COUNTING
    std::cerr<<"max possible num simplices: "<<max_num_simplices<<std::endl;
#endif
    *h_num_columns_to_reduce= 0;

    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, populate_sparse_simplices_warpfiltering, 256, 0));
    grid_size  *= deviceProp.multiProcessorCount;
    //columns_to_reduce contains the "new set" of simplices
#ifdef COUNTING
    std::cerr<<"(sparse) num simplices before kernel call: "<<*h_num_simplices<<std::endl;
#endif

    populate_sparse_simplices_warpfiltering<<<grid_size, 256, 256 * dim * sizeof(index_t)>>>(d_simplices, d_num_simplices, d_columns_to_reduce, d_num_columns_to_reduce, d_CSR_distance_matrix, n, dim, threshold, d_binomial_coeff);
    CUDACHECK(cudaDeviceSynchronize());

    cudaMemcpy(d_simplices, d_columns_to_reduce, sizeof(struct diameter_index_t_struct)*(*h_num_columns_to_reduce),cudaMemcpyDeviceToDevice);
    *h_num_simplices= *h_num_columns_to_reduce;

#ifdef COUNTING
    std::cerr<<"(sparse) num simplices for dim "<<dim<<": "<<*h_num_simplices<<std::endl;
#endif
    struct greaterdiam_lowerindex_diameter_index_t_struct_compare cmp;

    thrust::sort(thrust::device, d_simplices, d_simplices+*h_num_simplices, cmp);
    CUDACHECK(cudaDeviceSynchronize());
    cudaMemcpy(h_simplices, d_simplices, sizeof(struct diameter_index_t_struct)*(*h_num_simplices),cudaMemcpyDeviceToHost);

    //populate the columns_to_reduce vector on CPU side
    struct row_cidx_column_idx_struct_compare pair_cmp;
    columns_to_reduce.clear();
    for(index_t i=0; i<*h_num_simplices; i++){
        struct diameter_index_t_struct s= h_simplices[i];

        if(s.diameter<=threshold &&
           get_value_pivot_array_hashmap(s.index, pair_cmp)==-1){
            columns_to_reduce.push_back(s);
        }
    }
#ifdef COUNTING
    std::cerr<<"columns to reduce for dim: "<<dim<<": "<<columns_to_reduce.size()<<std::endl;
#endif
    *h_num_columns_to_reduce= columns_to_reduce.size();

#pragma omp parallel for schedule(guided,1)
    for(index_t i=0; i<columns_to_reduce.size(); i++){
        h_columns_to_reduce[i]= columns_to_reduce[i];
    }
}

template <>
void ripser<compressed_lower_distance_matrix>::cpu_byneighbor_assemble_columns_to_reduce(std::vector<diameter_index_t_struct>& simplices,
                                                                                         std::vector<diameter_index_t_struct>& columns_to_reduce,
                                                                                         hash_map<index_t,index_t>& pivot_column_index, index_t dim){
#ifdef INDICATE_PROGRESS
    std::cerr << clear_line << "assembling columns on CPU" << std::flush;
    std::chrono::steady_clock::time_point next= std::chrono::steady_clock::now() + time_step;
#endif
    --dim;
    columns_to_reduce.clear();
    std::vector<struct diameter_index_t_struct> next_simplices;

    for (struct diameter_index_t_struct& simplex : simplices) {

        simplex_coboundary_enumerator cofacets(simplex, dim, *this);

        while (cofacets.has_next(false)) {
#ifdef INDICATE_PROGRESS
            if (std::chrono::steady_clock::now() > next) {
                       std::cerr << clear_line << "assembling " << next_simplices.size()
                                 << " columns (processing " << std::distance(&simplices[0], &simplex)
                                 << "/" << simplices.size() << " simplices)" << std::flush;
                       next= std::chrono::steady_clock::now() + time_step;
                   }
#endif
            auto cofacet= cofacets.next();
            if (cofacet.diameter <= threshold) {

                next_simplices.push_back(cofacet);

                if (pivot_column_index.find(cofacet.index) == pivot_column_index.end()) {
                    columns_to_reduce.push_back(cofacet);
                }
            }
        }
    }

    simplices.swap(next_simplices);

#ifdef INDICATE_PROGRESS
    std::cerr << clear_line << "sorting " << columns_to_reduce.size() << " columns"
                     << std::flush;
#endif
    struct greaterdiam_lowerindex_diameter_index_t_struct_compare cmp;
    std::sort(columns_to_reduce.begin(), columns_to_reduce.end(),
              cmp);
#ifdef INDICATE_PROGRESS
    std::cerr << clear_line << std::flush;
#endif

}

template <>
void ripser<sparse_distance_matrix>::cpu_byneighbor_assemble_columns_to_reduce(std::vector<diameter_index_t_struct>& simplices,
                                                                               std::vector<diameter_index_t_struct>& columns_to_reduce,
                                                                               hash_map<index_t,index_t>& pivot_column_index, index_t dim){
#ifdef INDICATE_PROGRESS
    std::cerr << clear_line << "assembling columns" << std::flush;
    std::chrono::steady_clock::time_point next= std::chrono::steady_clock::now() + time_step;
#endif
    --dim;
    columns_to_reduce.clear();
    std::vector<struct diameter_index_t_struct> next_simplices;

    for (struct diameter_index_t_struct& simplex : simplices) {

        simplex_coboundary_enumerator cofacets(simplex, dim, *this);

        while (cofacets.has_next(false)) {
#ifdef INDICATE_PROGRESS
            if (std::chrono::steady_clock::now() > next) {
                       std::cerr << clear_line << "assembling " << next_simplices.size()
                                 << " columns (processing " << std::distance(&simplices[0], &simplex)
                                 << "/" << simplices.size() << " simplices)" << std::flush;
                       next= std::chrono::steady_clock::now() + time_step;
                   }
#endif
            auto cofacet= cofacets.next();
            if (cofacet.diameter <= threshold) {

                next_simplices.push_back(cofacet);

                if (pivot_column_index.find(cofacet.index) == pivot_column_index.end()) { //|| pivot_column_index[cofacet.index]==-1)
                    columns_to_reduce.push_back(cofacet);
                }
            }
        }
    }

    simplices.swap(next_simplices);

#ifdef INDICATE_PROGRESS
    std::cerr << clear_line << "sorting " << columns_to_reduce.size() << " columns"
                     << std::flush;
#endif
    struct greaterdiam_lowerindex_diameter_index_t_struct_compare cmp;
    std::sort(columns_to_reduce.begin(), columns_to_reduce.end(),
              cmp);
#ifdef INDICATE_PROGRESS
    std::cerr << clear_line << std::flush;
#endif

}

template <>
void ripser<compressed_lower_distance_matrix>::cpu_assemble_columns_to_reduce(std::vector<diameter_index_t_struct>& columns_to_reduce,
                                                                              hash_map<index_t, index_t>& pivot_column_index,
                                                                              index_t dim) {
    index_t num_simplices= binomial_coeff(n, dim + 1);
#ifdef COUNTING
    std::cerr<<"max num possible simplices: "<<num_simplices<<std::endl;
#endif
    columns_to_reduce.clear();

#ifdef INDICATE_PROGRESS
    std::cerr << "\033[K"
	          << "assembling " << num_simplices << " columns" << std::flush << "\r";
#endif
    index_t count= 0;
    for (index_t index= 0; index < num_simplices; ++index) {
        if (pivot_column_index.find(index) == pivot_column_index.end()) {
            value_t diameter= compute_diameter(index, dim);
            if (diameter <= threshold){
                columns_to_reduce.push_back({diameter,index});
                count++;
            }
#ifdef INDICATE_PROGRESS
            if ((index + 1) % 1000000 == 0)
				std::cerr << "\033[K"
				          << "assembled " << columns_to_reduce.size() << " out of " << (index + 1)
				          << "/" << num_simplices << " columns" << std::flush << "\r";
#endif
        }
    }

#ifdef INDICATE_PROGRESS
    std::cerr << "\033[K"
	          << "sorting " << num_simplices << " columns" << std::flush << "\r";
#endif
    struct greaterdiam_lowerindex_diameter_index_t_struct_compare cmp;
    std::sort(columns_to_reduce.begin(), columns_to_reduce.end(),
              cmp);
#ifdef INDICATE_PROGRESS
    std::cerr << "\033[K";
#endif
}

template <>
void ripser<compressed_lower_distance_matrix>::assemble_columns_gpu_accel_transition_to_cpu_only(const bool& more_than_one_dim_cpu_only,std::vector<diameter_index_t_struct>& simplices, std::vector<diameter_index_t_struct>& columns_to_reduce, hash_map<index_t,index_t>& cpu_pivot_column_index,
                                                                                                 index_t dim){

    index_t max_num_simplices= binomial_coeff(n,dim+1);
    //insert all pivots from the two gpu pivot data structures into cpu_pivot_column_index, cannot parallelize this for loop due to concurrency issues of hashmaps
    for (index_t i= 0; i < max_num_simplices; i++) {
#ifdef USE_PHASHMAP
        index_t col_idx= phmap_get_value(i);
        if(col_idx!=-1) {
            cpu_pivot_column_index[i]= col_idx;
        }
#endif
#ifdef USE_GOOGLE_HASHMAP
        auto pair= pivot_column_index.find(i);
        if(pair!=pivot_column_index.end()) {
            cpu_pivot_column_index[i]= pair->second;
        }
        //}else{
            //h_pivot_column_index_array_OR_nonapparent_cols[i]= -1;
        //}
#endif
    }

    num_apparent= *h_num_columns_to_reduce-*h_num_nonapparent;
    if(num_apparent>0) {
        //we can't insert into the hashmap in parallel
        for (index_t i= 0; i < num_apparent; i++) {
            index_t row_cidx= h_pivot_array[i].row_cidx;
            index_t column_idx= h_pivot_array[i].column_idx;
            if(column_idx!=-1) {
                cpu_pivot_column_index[row_cidx]= column_idx;
            }
        }
    }

#ifdef INDICATE_PROGRESS
    std::cerr << clear_line << "assembling columns" << std::flush;
		std::chrono::steady_clock::time_point next= std::chrono::steady_clock::now() + time_step;
#endif
    columns_to_reduce.clear();
    simplices.clear();
    index_t count_simplices= 0;
//cpu_pivot_column_index can't be parallelized for lookup
    for (index_t index= 0; index < max_num_simplices; ++index) {
        value_t diameter= -MAX_FLOAT;

        //the second condition after the || should never happen, since we never insert such pairs into cpu_pivot_column_index
        if (cpu_pivot_column_index.find(index) == cpu_pivot_column_index.end() || cpu_pivot_column_index[index]==-1) {
            diameter= compute_diameter(index, dim);
            if (diameter <= threshold) {
                columns_to_reduce.push_back({diameter, index});
            }
#ifdef INDICATE_PROGRESS
            if ((index + 1) % 1000000 == 0)
				std::cerr << "\033[K"
				          << "assembled " << columns_to_reduce.size() << " out of " << (index + 1)
				          << "/" << max_num_simplices << " columns" << std::flush << "\r";
#endif
        }

        if(more_than_one_dim_cpu_only){
            if(diameter==-MAX_FLOAT){
                diameter= compute_diameter(index, dim);
            }
            if(diameter<=threshold){
                simplices.push_back({diameter,index});
                count_simplices++;
            }
        }
    }
#ifdef COUNTING
    if(more_than_one_dim_cpu_only){
        std::cerr<<"(if there are multiple dimensions needed to compute) num simplices for dim: "<<dim<<" is: "<<count_simplices<<std::endl;
    }
#endif
#ifdef INDICATE_PROGRESS
    std::cerr << "\033[K"
	          << "sorting " << columns_to_reduce.size() << " columns" << std::flush << "\r";
#endif
    greaterdiam_lowerindex_diameter_index_t_struct_compare cmp;
    std::sort(columns_to_reduce.begin(), columns_to_reduce.end(), cmp);

#ifdef COUNTING
    std::cerr<<"NUM COLS to reduce for CPU: "<<columns_to_reduce.size()<<std::endl;
#endif

#ifdef INDICATE_PROGRESS
    std::cerr << clear_line << std::flush;
#endif

}

template <>
void ripser<compressed_lower_distance_matrix>::compute_barcodes() {

    Stopwatch sw, gpu_accel_timer;
    gpu_accel_timer.start();
    sw.start();

    index_t gpu_dim_max= calculate_gpu_dim_max_for_fullrips_computation_from_memory(dim_max, true);
#ifdef PROFILING
    std::cerr<<"recalculated dim_max based on GPU free DRAM capacity: "<<gpu_dim_max<<std::endl;
#endif
    max_num_simplices_forall_dims= gpu_dim_max<(n/2)-1?get_num_simplices_for_dim(gpu_dim_max): get_num_simplices_for_dim((n/2)-1);
    if(gpu_dim_max>=1){
#ifdef COUNTING
        std::cerr<<"max possible num simplices over all dim<=dim_max (without clearing) for memory allocation: "<<max_num_simplices_forall_dims<<std::endl;
#endif
        CUDACHECK(cudaMalloc((void **) &d_columns_to_reduce, sizeof(struct diameter_index_t_struct) * max_num_simplices_forall_dims));
        h_columns_to_reduce= (struct diameter_index_t_struct*) malloc(sizeof(struct diameter_index_t_struct)* max_num_simplices_forall_dims);

        if(h_columns_to_reduce==NULL){
            std::cerr<<"malloc for h_columns_to_reduce failed"<<std::endl;
            exit(1);
        }

#ifndef ASSEMBLE_REDUCTION_SUBMATRIX
            CUDACHECK(cudaMalloc((void**) &d_flagarray, sizeof(char)*max_num_simplices_forall_dims));
#endif

        CUDACHECK(cudaMalloc((void **) &d_cidx_to_diameter, sizeof(value_t)*max_num_simplices_forall_dims));
#if defined(ASSEMBLE_REDUCTION_SUBMATRIX)
        CUDACHECK(cudaMalloc((void **) &d_flagarray_OR_index_to_subindex, sizeof(index_t)*max_num_simplices_forall_dims));

        h_flagarray_OR_index_to_subindex= (index_t*) malloc(sizeof(index_t)*max_num_simplices_forall_dims);
        if(h_flagarray_OR_index_to_subindex==NULL) {
            std::cerr<<"malloc for h_index_to_subindex failed"<<std::endl;
        }
#endif
        CUDACHECK(cudaMalloc((void **) &d_distance_matrix, sizeof(value_t)*dist.size()*(dist.size()-1)/2));
        cudaMemcpy(d_distance_matrix, dist.distances.data(), sizeof(value_t)*dist.size()*(dist.size()-1)/2, cudaMemcpyHostToDevice);

        CUDACHECK(cudaMalloc((void **) &d_pivot_column_index_OR_nonapparent_cols, sizeof(index_t)*max_num_simplices_forall_dims));

        //this array is used for both the pivot column index hash table array as well as the nonapparent cols array as an unstructured hashmap
        h_pivot_column_index_array_OR_nonapparent_cols= (index_t*) malloc(sizeof(index_t)*max_num_simplices_forall_dims);

        if(h_pivot_column_index_array_OR_nonapparent_cols==NULL){
            std::cerr<<"malloc for h_pivot_column_index_array_OR_nonapparent_cols failed"<<std::endl;
            exit(1);
        }

        //copy object over to GPU
        CUDACHECK(cudaMalloc((void**) &d_binomial_coeff, sizeof(binomial_coeff_table)));
        cudaMemcpy(d_binomial_coeff, &binomial_coeff, sizeof(binomial_coeff_table), cudaMemcpyHostToDevice);

        index_t num_binoms= binomial_coeff.get_num_n()*binomial_coeff.get_max_tuple_length();

        CUDACHECK(cudaMalloc((void **) &h_d_binoms, sizeof(index_t)*num_binoms));
        cudaMemcpy(h_d_binoms, binomial_coeff.binoms, sizeof(index_t)*num_binoms, cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_binomial_coeff->binoms), &h_d_binoms, sizeof(index_t*), cudaMemcpyHostToDevice);

        cudaHostAlloc((void **)&h_num_columns_to_reduce, sizeof(index_t), cudaHostAllocPortable | cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_num_columns_to_reduce, h_num_columns_to_reduce,0);
        cudaHostAlloc((void **)&h_num_nonapparent, sizeof(index_t), cudaHostAllocPortable | cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_num_nonapparent, h_num_nonapparent,0);

        CUDACHECK(cudaMalloc((void**) &d_lowest_one_of_apparent_pair, sizeof(index_t)*max_num_simplices_forall_dims));
        CUDACHECK(cudaMalloc((void**) &d_pivot_array, sizeof(struct index_t_pair_struct)*max_num_simplices_forall_dims));
        h_pivot_array= (struct index_t_pair_struct*) malloc(sizeof(struct index_t_pair_struct)*max_num_simplices_forall_dims);
        if(h_pivot_array==NULL){
            std::cerr<<"malloc for h_pivot_array failed"<<std::endl;
            exit(1);
        }
#ifdef PROFILING
        cudaMemGetInfo(&freeMem,&totalMem);
        std::cerr<<"GPU memory after full rips memory calculation and allocation, total mem: "<< totalMem<<" bytes, free mem: "<<freeMem<<" bytes"<<std::endl;
#endif
    }
    sw.stop();
#ifdef PROFILING
    std::cerr<<"CUDA PREPROCESSING TIME (e.g. memory allocation time): "<<sw.ms()/1000.0<<"s"<<std::endl;
#endif
    sw.start();

    columns_to_reduce.clear();
    std::vector<diameter_index_t_struct> simplices;
    if(gpu_dim_max>=1) {
        gpu_compute_dim_0_pairs(columns_to_reduce);
        sw.stop();
#ifdef PROFILING
        std::cerr<<"0-dimensional persistence total computation time with GPU: "<<sw.ms()/1000.0<<"s"<<std::endl;
#endif
    }else{

        compute_dim_0_pairs(simplices, columns_to_reduce);
        sw.stop();
#ifdef PROFILING
        std::cerr<<"0-dimensional persistence total computation time with CPU alone: "<<sw.ms()/1000.0<<"s"<<std::endl;
#endif
    }

    //index_t dim_forgpuscan= MAX_INT64;//never do gpu scan
    index_t dim_forgpuscan= 1;
    for (index_t dim= 1; dim <= gpu_dim_max; ++dim) {
        Stopwatch sw;
        sw.start();
#ifdef USE_PHASHMAP
        phmap_clear();
#endif
#ifdef USE_GOOGLE_HASHMAP
        pivot_column_index.clear();
            pivot_column_index.resize(*h_num_columns_to_reduce);
#endif
        *h_num_nonapparent= 0;

        //search for apparent pairs
        gpuscan(dim);
        //dim_forgpuscan= dim;//update dim_forgpuscan to the dimension that gpuscan was just done at
        sw.stop();
#ifdef PROFILING
        std::cerr<<"-SUM OF GPU MATRIX SCAN and post processing time for dim "<<dim<<": "<<sw.ms()/1000.0<<"s"<<std::endl;
#endif
        sw.start();

        compute_pairs_plusplus(
                dim, dim_forgpuscan);
        sw.stop();
#ifdef PROFILING
        std::cerr<<"SUBMATRIX REDUCTION TIME for dim "<< dim<<": "<<sw.ms()/1000.0<<"s"<<"\n"<<std::endl;
#endif
        if (dim < gpu_dim_max) {
            sw.start();
            gpu_assemble_columns_to_reduce_plusplus(dim+1);
            sw.stop();

#ifdef PROFILING
            std::cerr << "ASSEMBLE COLS TIME for dim " << dim + 1 << ": " << sw.ms() / 1000.0
                      << "s" << std::endl;
#endif
        }
    }
    gpu_accel_timer.stop();
#ifdef PROFILING
    if(gpu_dim_max>=1)
        std::cerr<<"GPU ACCELERATED COMPUTATION from dim 0 to dim "<<gpu_dim_max<<": "<<gpu_accel_timer.ms()/1000.0<<"s"<<std::endl;
#endif
    if(dim_max>gpu_dim_max){//do cpu only computation from this point on
#ifdef CPUONLY_SPARSE_HASHMAP
        std::cerr<<"MEMORY EFFICIENT/BUT TIME INEFFICIENT CPU-ONLY MODE FOR REMAINDER OF HIGH DIMENSIONAL COMPUTATION (NOT ENOUGH GPU DEVICE MEMORY)"<<std::endl;
#endif
#ifndef CPUONLY_SPARSE_HASHMAP
        std::cerr<<"CPU-ONLY MODE FOR REMAINDER OF HIGH DIMENSIONAL COMPUTATION (NOT ENOUGH GPU DEVICE MEMORY)"<<std::endl;
#endif
        free_init_cpumem();
        hash_map<index_t,index_t> cpu_pivot_column_index;
        cpu_pivot_column_index.reserve(*h_num_columns_to_reduce);
        bool more_than_one_dim_to_compute= dim_max>gpu_dim_max+1;
        assemble_columns_gpu_accel_transition_to_cpu_only(more_than_one_dim_to_compute, simplices, columns_to_reduce, cpu_pivot_column_index, gpu_dim_max+1);
        free_remaining_cpumem();
        for (index_t dim= gpu_dim_max+1; dim <= dim_max; ++dim) {
            cpu_pivot_column_index.clear();
            cpu_pivot_column_index.reserve(columns_to_reduce.size());
            compute_pairs(columns_to_reduce, cpu_pivot_column_index, dim);
            if(dim<dim_max){
                sw.start();
                //cpu_byneighbor_assemble_columns is a little faster?
                cpu_byneighbor_assemble_columns_to_reduce(simplices, columns_to_reduce, cpu_pivot_column_index, dim+1);
                //cpu_assemble_columns_to_reduce(columns_to_reduce,cpu_pivot_column_index, dim+1);
                sw.stop();
#ifdef PROFILING
                std::cerr<<"TIME FOR CPU ASSEMBLE: "<<sw.ms()/1000.0<<"s"<<std::endl;
#endif
            }
        }
    }else {
        if (n >= 10) {
            free_init_cpumem();
            free_remaining_cpumem();
        }
    }
    if(gpu_dim_max>=1 && n>=10) {
        free_gpumem_dense_computation();
//        free CPU memory:

        cudaFreeHost(h_num_columns_to_reduce);
        cudaFreeHost(h_num_nonapparent);
        //free(h_pivot_array)
        //free(h_columns_to_reduce)
        //free(h_pivot_column_index_array_OR_nonapparent_cols)
#if defined(ASSEMBLE_REDUCTION_SUBMATRIX)
        free(h_flagarray_OR_index_to_subindex);
#endif
    }
}

template <>
void ripser<sparse_distance_matrix>::compute_barcodes() {
    Stopwatch sw, gpu_accel_timer;
    gpu_accel_timer.start();
    sw.start();

    index_t maxgpu_dim = calculate_gpu_dim_max_for_fullrips_computation_from_memory(dim_max, false);
    if (maxgpu_dim < dim_max) {
        max_num_simplices_forall_dims = calculate_gpu_max_columns_for_sparserips_computation_from_memory();
#ifdef COUNTING
        std::cerr<<"(sparse) max possible num simplices for memory allocation forall dims: "<<max_num_simplices_forall_dims<<std::endl;
#endif
    } else {
        max_num_simplices_forall_dims =
                dim_max < (n / 2) - 1 ? get_num_simplices_for_dim(dim_max) : get_num_simplices_for_dim((n / 2) - 1);
#ifdef COUNTING
        std::cerr<<"(dense case used in sparse computation) max possible num simplices for memory allocation forall dims: "<<max_num_simplices_forall_dims<<std::endl;
#endif
    }

    //we assume that we have enough memory to last up to dim_max (should be fine with a >=32GB GPU); growth of num simplices can be very slow for sparse case
    if (dim_max >= 1) {

        CUDACHECK(cudaMalloc((void **) &d_columns_to_reduce,
                             sizeof(struct diameter_index_t_struct) * max_num_simplices_forall_dims));//46000000
        h_columns_to_reduce = (struct diameter_index_t_struct *) malloc(
                sizeof(struct diameter_index_t_struct) * max_num_simplices_forall_dims);

        if (h_columns_to_reduce == NULL) {
            std::cerr << "malloc for h_columns_to_reduce failed" << std::endl;
            exit(1);
        }

#if defined(ASSEMBLE_REDUCTION_SUBMATRIX)
        CUDACHECK(cudaMalloc((void **) &d_flagarray_OR_index_to_subindex,
                             sizeof(index_t) * max_num_simplices_forall_dims));

        h_flagarray_OR_index_to_subindex = (index_t *) malloc(sizeof(index_t) * max_num_simplices_forall_dims);
        if (h_flagarray_OR_index_to_subindex == NULL) {
            std::cerr << "malloc for h_index_to_subindex failed" << std::endl;
        }
#endif

        CSR_distance_matrix CSR_distance_matrix = dist.toCSR();
        //copy CSR_distance_matrix object over to GPU
        CUDACHECK(cudaMalloc((void **) &d_CSR_distance_matrix, sizeof(CSR_distance_matrix)));
        cudaMemcpy(d_CSR_distance_matrix, &CSR_distance_matrix, sizeof(CSR_distance_matrix), cudaMemcpyHostToDevice);

        CUDACHECK(cudaMalloc((void **) &h_d_offsets, sizeof(index_t) * (CSR_distance_matrix.n + 1)));
        cudaMemcpy(h_d_offsets, CSR_distance_matrix.offsets, sizeof(index_t) * (CSR_distance_matrix.n + 1),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_CSR_distance_matrix->offsets), &h_d_offsets, sizeof(index_t *), cudaMemcpyHostToDevice);

        CUDACHECK(cudaMalloc((void **) &h_d_entries, sizeof(value_t) * CSR_distance_matrix.num_entries));
        cudaMemcpy(h_d_entries, CSR_distance_matrix.entries, sizeof(value_t) * CSR_distance_matrix.num_entries,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_CSR_distance_matrix->entries), &h_d_entries, sizeof(value_t *), cudaMemcpyHostToDevice);

        CUDACHECK(cudaMalloc((void **) &h_d_col_indices, sizeof(index_t) * CSR_distance_matrix.num_entries));
        cudaMemcpy(h_d_col_indices, CSR_distance_matrix .col_indices, sizeof(index_t) * CSR_distance_matrix .num_entries,
                   cudaMemcpyHostToDevice);
        
        cudaMemcpy(&(d_CSR_distance_matrix->col_indices), &h_d_col_indices, sizeof(index_t *), cudaMemcpyHostToDevice);

        //this replaces d_cidx_to_diameter
        CUDACHECK(cudaMalloc((void **) &d_cidx_diameter_pairs_sortedlist,
                             sizeof(struct diameter_index_t_struct) * max_num_simplices_forall_dims));

        CUDACHECK(cudaMalloc((void **) &d_pivot_column_index_OR_nonapparent_cols,
                             sizeof(index_t) * max_num_simplices_forall_dims));

        //this array is used for both the pivot column index hash table array as well as the nonapparent cols array as an unstructured hashmap
        h_pivot_column_index_array_OR_nonapparent_cols = (index_t *) malloc(
                sizeof(index_t) * max_num_simplices_forall_dims);
        if (h_pivot_column_index_array_OR_nonapparent_cols == NULL) {
            std::cerr << "malloc for h_pivot_column_index_array_OR_nonapparent_cols failed" << std::endl;
            exit(1);
        }

        //copy object over to GPU
        CUDACHECK(cudaMalloc((void **) &d_binomial_coeff, sizeof(binomial_coeff_table)));
        cudaMemcpy(d_binomial_coeff, &binomial_coeff, sizeof(binomial_coeff_table), cudaMemcpyHostToDevice);

        index_t num_binoms = binomial_coeff.get_num_n() * binomial_coeff.get_max_tuple_length();

        CUDACHECK(cudaMalloc((void **) &h_d_binoms, sizeof(index_t) * num_binoms));
        cudaMemcpy(h_d_binoms, binomial_coeff.binoms, sizeof(index_t) * num_binoms, cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_binomial_coeff->binoms), &h_d_binoms, sizeof(index_t *), cudaMemcpyHostToDevice);

        cudaHostAlloc((void **) &h_num_columns_to_reduce, sizeof(index_t), cudaHostAllocPortable | cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_num_columns_to_reduce, h_num_columns_to_reduce, 0);
        cudaHostAlloc((void **) &h_num_nonapparent, sizeof(index_t), cudaHostAllocPortable | cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_num_nonapparent, h_num_nonapparent, 0);
        cudaHostAlloc((void **) &h_num_simplices, sizeof(index_t), cudaHostAllocPortable | cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_num_simplices, h_num_simplices, 0);

        CUDACHECK(
                cudaMalloc((void **) &d_lowest_one_of_apparent_pair, sizeof(index_t) * max_num_simplices_forall_dims));
        CUDACHECK(cudaMalloc((void **) &d_pivot_array,
                             sizeof(struct index_t_pair_struct) * max_num_simplices_forall_dims));
        h_pivot_array = (struct index_t_pair_struct *) malloc(
                sizeof(struct index_t_pair_struct) * max_num_simplices_forall_dims);
        if (h_pivot_array == NULL) {
            std::cerr << "malloc for h_pivot_array failed" << std::endl;
            exit(1);
        }
        CUDACHECK(cudaMalloc((void **) &d_simplices,
                             sizeof(struct diameter_index_t_struct) * max_num_simplices_forall_dims));
        h_simplices = (struct diameter_index_t_struct *) malloc(
                sizeof(struct diameter_index_t_struct) * max_num_simplices_forall_dims);

        if (h_simplices == NULL) {
            std::cerr << "malloc for h_simplices failed" << std::endl;
            exit(1);
        }
#ifdef PROFILING
        cudaMemGetInfo(&freeMem,&totalMem);
        std::cerr<<"after GPU memory allocation: total mem, free mem: " <<totalMem<<" bytes, "<<freeMem<<" bytes"<<std::endl;
#endif

    }
    sw.stop();
#ifdef PROFILING
    std::cerr<<"CUDA PREPROCESSING TIME (e.g. memory allocation): "<<sw.ms()/1000.0<<"s"<<std::endl;
#endif
    sw.start();

    columns_to_reduce.clear();
    if (dim_max >= 1) {

        gpu_compute_dim_0_pairs(columns_to_reduce);
        sw.stop();
#ifdef PROFILING
        std::cerr<<"0-dimensional persistence total computation time with GPU: "<<sw.ms()/1000.0<<"s"<<std::endl;
#endif
    } else {
        std::vector<diameter_index_t_struct> simplices;
        compute_dim_0_pairs(simplices, columns_to_reduce);
        sw.stop();
#ifdef PROFILING
        std::cerr<<"0-dimensional persistence total computation time with CPU alone: "<<sw.ms()/1000.0<<"s"<<std::endl;
#endif
    }

    //index_t dim_forgpuscan= MAX_INT64;//never do gpuscan
    index_t dim_forgpuscan = 1;
    for (index_t dim = 1; dim <= dim_max; ++dim) {
        Stopwatch sw;
        sw.start();
#ifdef USE_PHASHMAP
        phmap_clear();
#endif
#ifdef USE_GOOGLE_HASHMAP
        pivot_column_index.clear();
            pivot_column_index.resize(*h_num_columns_to_reduce);
#endif
        *h_num_nonapparent = 0;

        gpuscan(dim);
        //dim_forgpuscan= dim;
        sw.stop();
#ifdef PROFILING
        std::cerr << "-SUM OF GPU MATRIX SCAN and post processing time for dim " << dim << ": " << sw.ms() / 1000.0
                  << "s" << std::endl;
#endif

        sw.start();

        compute_pairs_plusplus(
                dim, dim_forgpuscan);
        sw.stop();
#ifdef PROFILING
        std::cerr << "SUBMATRIX REDUCTION TIME for dim " << dim << ": " << sw.ms() / 1000.0 << "s" << "\n" << std::endl;
#endif
        if (dim < dim_max) {
            sw.start();
            gpu_assemble_columns_to_reduce_plusplus(dim + 1);

            sw.stop();

#ifdef PROFILING
            std::cerr << "ASSEMBLE COLS TIME for dim " << dim + 1 << ": " << sw.ms() / 1000.0
                      << "s" << std::endl;
#endif
        }
    }
    gpu_accel_timer.stop();
#ifdef PROFILING
    std::cerr<<"GPU ACCELERATED COMPUTATION: "<<gpu_accel_timer.ms()/1000.0<<"s"<<std::endl;
#endif

    if (maxgpu_dim >= 1 && n>=10) {
        free_init_cpumem();
        free_remaining_cpumem();
        free_gpumem_sparse_computation();

        cudaFreeHost(h_num_columns_to_reduce);
        cudaFreeHost(h_num_nonapparent);
        //free(h_pivot_array)
        //free(h_columns_to_reduce)
        //free(h_pivot_column_index_array_OR_nonapparent_cols)
#if defined(ASSEMBLE_REDUCTION_SUBMATRIX)
        free(h_flagarray_OR_index_to_subindex);
#endif
        cudaFreeHost(h_num_simplices);
        free(h_simplices);
    }
}
///I/O code

enum file_format { LOWER_DISTANCE_MATRIX, DISTANCE_MATRIX, POINT_CLOUD, DIPHA, SPARSE, BINARY };

template <typename T> T read(std::istream& s) {
    T result;
    s.read(reinterpret_cast<char*>(&result), sizeof(T));
    return result; // on little endian: boost::endian::little_to_native(result);
}
compressed_lower_distance_matrix read_point_cloud_python(value_t* matrix, int num_rows, int num_columns){
    std::vector<std::vector<value_t>> points;
    for(int i= 0; i < num_rows; i++) {
        std::vector <value_t> point;
        for (int j= 0; j < num_columns; j++) {
            point.push_back(matrix[i * num_columns + j]);
        }
        if (!point.empty()) {
            points.push_back(point);
        }
        assert(point.size() == points.front().size());
    }
    //only l2 distance implemented so far
    euclidean_distance_matrix eucl_dist(std::move(points));

    index_t n= eucl_dist.size();
#ifdef COUNTING
    std::cout << "point cloud with " << n << " points in dimension "
                  << eucl_dist.points.front().size() << std::endl;
#endif
    std::vector<value_t> distances;

    for (int i= 0; i < n; ++i)
        for (int j= 0; j < i; ++j) distances.push_back(eucl_dist(i, j));

    return compressed_lower_distance_matrix(std::move(distances));
}

compressed_lower_distance_matrix read_point_cloud(std::istream& input_stream) {
    std::vector<std::vector<value_t>> points;

    std::string line;
    value_t value;
    while (std::getline(input_stream, line)) {
        std::vector<value_t> point;
        std::istringstream s(line);
        while (s >> value) {
            point.push_back(value);
            s.ignore();
        }
        if (!point.empty()) points.push_back(point);
        assert(point.size() == points.front().size());
    }

    euclidean_distance_matrix eucl_dist(std::move(points));

    index_t n= eucl_dist.size();
#ifdef COUNTING
    std::cout << "point cloud with " << n << " points in dimension "
              << eucl_dist.points.front().size() << std::endl;
#endif
    std::vector<value_t> distances;

    for (int i= 0; i < n; ++i)
        for (int j= 0; j < i; ++j) distances.push_back(eucl_dist(i, j));

    return compressed_lower_distance_matrix(std::move(distances));
}

//the coo format input is of a lower triangular matrix
sparse_distance_matrix read_sparse_distance_matrix(std::istream& input_stream) {
    std::vector<std::vector<index_diameter_t_struct>> neighbors;
    index_t num_edges= 0;

    std::string line;
    while (std::getline(input_stream, line)) {
        std::istringstream s(line);
        size_t i, j;
        value_t value;
        s >> i;
        s >> j;
        s >> value;
        if (i != j) {
            neighbors.resize(std::max({neighbors.size(), i + 1, j + 1}));
            neighbors[i].push_back({(index_t) j, value});
            neighbors[j].push_back({(index_t) i, value});
            ++num_edges;
        }
    }

    struct lowerindex_lowerdiameter_index_t_struct_compare cmp_index_diameter;

    for (size_t i= 0; i < neighbors.size(); ++i)
        std::sort(neighbors[i].begin(), neighbors[i].end(), cmp_index_diameter);

    return sparse_distance_matrix(std::move(neighbors), num_edges);
}
sparse_distance_matrix read_sparse_distance_matrix_python(value_t* matrix, int matrix_length){
    std::vector<std::vector<index_diameter_t_struct>> neighbors;
    index_t num_edges= 0;

    for(index_t k = 0; k < matrix_length; k+=3){
        size_t i, j;
        value_t value;
        i = matrix[k];
        j = matrix[k+1];
        value = matrix[k+2];
        if (i != j) {
            neighbors.resize(std::max({neighbors.size(), i + 1, j + 1}));
            neighbors[i].push_back({(index_t) j, value});
            neighbors[j].push_back({(index_t) i, value});
            ++num_edges;
        }
    }

    struct lowerindex_lowerdiameter_index_t_struct_compare cmp_index_diameter;

    for (size_t i= 0; i < neighbors.size(); ++i)
        std::sort(neighbors[i].begin(), neighbors[i].end(), cmp_index_diameter);

    return sparse_distance_matrix(std::move(neighbors), num_edges);
}
compressed_lower_distance_matrix read_lower_distance_matrix_python(value_t* matrix, int matrix_length) {

    std::vector<value_t> distances(matrix, matrix + matrix_length);

    return compressed_lower_distance_matrix(std::move(distances));
}

compressed_lower_distance_matrix read_lower_distance_matrix(std::istream& input_stream) {
    std::vector<value_t> distances;
    value_t value;
    while (input_stream >> value) {
        distances.push_back(value);
        input_stream.ignore();
    }

    return compressed_lower_distance_matrix(std::move(distances));
}

compressed_lower_distance_matrix read_distance_matrix_python(value_t* matrix, int matrix_length) {

    std::vector<value_t> distances(matrix, matrix + matrix_length);

    return compressed_lower_distance_matrix(std::move(distances));
}

compressed_lower_distance_matrix read_distance_matrix(std::istream& input_stream) {
    std::vector<value_t> distances;

    std::string line;
    value_t value;
    for (int i= 0; std::getline(input_stream, line); ++i) {
        std::istringstream s(line);
        for (int j= 0; j < i && s >> value; ++j) {
            distances.push_back(value);
            s.ignore();
        }
    }

    return compressed_lower_distance_matrix(std::move(distances));
}

compressed_lower_distance_matrix read_dipha(std::istream& input_stream) {
    if (read<int64_t>(input_stream) != 8067171840) {
        std::cerr << "input is not a Dipha file (magic number: 8067171840)" << std::endl;
        exit(-1);
    }

    if (read<int64_t>(input_stream) != 7) {
        std::cerr << "input is not a Dipha distance matrix (file type: 7)" << std::endl;
        exit(-1);
    }

    index_t n= read<int64_t>(input_stream);

    std::vector<value_t> distances;

    for (int i= 0; i < n; ++i)
        for (int j= 0; j < n; ++j)
            if (i > j)
                distances.push_back(read<double>(input_stream));
            else
                read<double>(input_stream);

    return compressed_lower_distance_matrix(std::move(distances));
}

compressed_lower_distance_matrix read_binary(std::istream& input_stream) {
    std::vector<value_t> distances;
    while (!input_stream.eof()) distances.push_back(read<value_t>(input_stream));
    return compressed_lower_distance_matrix(std::move(distances));
}

compressed_lower_distance_matrix read_matrix_python(value_t* matrix, int num_entries, int num_rows, int num_columns, file_format format) {
    switch (format) {
        case LOWER_DISTANCE_MATRIX:
            return read_lower_distance_matrix_python(matrix, num_entries);
        case DISTANCE_MATRIX://assume that the distance matrix has been changed into lower_distance matrix format
            return read_distance_matrix_python(matrix, num_entries);
        case POINT_CLOUD:
            return read_point_cloud_python(matrix, num_rows, num_columns);
    }
    std::cerr<<"unsupported input file format for python interface"<<std::endl;
    exit(-1);
}

compressed_lower_distance_matrix read_file(std::istream& input_stream, file_format format) {
    switch (format) {
        case LOWER_DISTANCE_MATRIX:
            return read_lower_distance_matrix(input_stream);
        case DISTANCE_MATRIX:
            return read_distance_matrix(input_stream);
        case POINT_CLOUD:
            return read_point_cloud(input_stream);
        case DIPHA:
            return read_dipha(input_stream);
        default:
            return read_binary(input_stream);
    }
    std::cerr<<"unsupported input file format"<<std::endl;
}

void print_usage_and_exit(int exit_code) {
    std::cerr
            << "Usage: "
            << "ripser++ "
            << "[options] [filename]" << std::endl
            << std::endl
            << "Options:" << std::endl
            << std::endl
            << "  --help           print this screen" << std::endl
            << "  --format         use the specified file format for the input. Options are:"
            << std::endl
            << "                     lower-distance (lower triangular distance matrix; default)"
            << std::endl
            << "                     distance       (full distance matrix)" << std::endl
            << "                     point-cloud    (point cloud in Euclidean space)" << std::endl
            << "                     dipha          (distance matrix in DIPHA file format)" << std::endl
            << "                     sparse         (sparse distance matrix in sparse triplet (COO) format)"
            << std::endl
            << "                     binary         (distance matrix in Ripser binary file format)"
            << std::endl
            << "  --dim <k>        compute persistent homology up to dimension <k>" << std::endl
            << "  --threshold <t>  compute Rips complexes up to diameter <t>" << std::endl
            << "  --sparse         force sparse computation "<<std::endl
            << "  --ratio <r>      only show persistence pairs with death/birth ratio > r" << std::endl
            << std::endl;

    exit(exit_code);
}

extern "C" ripser_plusplus_result run_main_filename(int argc,  char** argv, const char* filename) {

    Stopwatch sw;


#ifdef PROFILING
    cudaDeviceProp deviceProp;
    size_t freeMem_start, freeMem_end, totalMemory;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaMemGetInfo(&freeMem_start,&totalMemory);
#endif
    sw.start();

    file_format format= DISTANCE_MATRIX;

    index_t dim_max= 1;
    value_t threshold= std::numeric_limits<value_t>::max();

    float ratio= 1;

    bool use_sparse= false;


    for (index_t i= 0; i < argc; i++) {
        const std::string arg(argv[i]);
        if (arg == "--help") {
            print_usage_and_exit(0);
        } else if (arg == "--dim") {
            std::string parameter= std::string(argv[++i]);
            size_t next_pos;
            dim_max= std::stol(parameter, &next_pos);
            if (next_pos != parameter.size()) print_usage_and_exit(-1);
        } else if (arg == "--threshold") {
            std::string parameter= std::string(argv[++i]);
            size_t next_pos;
            threshold= std::stof(parameter, &next_pos);
            if (next_pos != parameter.size()) print_usage_and_exit(-1);
        } else if (arg == "--ratio") {
            std::string parameter= std::string(argv[++i]);
            size_t next_pos;
            ratio= std::stof(parameter, &next_pos);
            if (next_pos != parameter.size()) print_usage_and_exit(-1);
        } else if (arg == "--format") {
            std::string parameter= std::string(argv[++i]);
            if (parameter == "lower-distance")
                format= LOWER_DISTANCE_MATRIX;
            else if (parameter == "distance")
                format= DISTANCE_MATRIX;
            else if (parameter == "point-cloud")
                format= POINT_CLOUD;
            else if (parameter == "dipha")
                format= DIPHA;
            else if (parameter == "sparse")
                format= SPARSE;
            else if (parameter == "binary")
                format= BINARY;
            else
                print_usage_and_exit(-1);
        } else if(arg=="--sparse") {
            use_sparse= true;
        }
    }

    list_of_barcodes = std::vector<std::vector<birth_death_coordinate>>();
    for(index_t i = 0; i <= dim_max; i++){
        list_of_barcodes.push_back(std::vector<birth_death_coordinate>());
    }

    std::ifstream file_stream(filename);
    if (filename && file_stream.fail()) {
        std::cerr << "couldn't open file " << filename << std::endl;
        exit(-1);
    }
    if (format == SPARSE) {
        Stopwatch IOsw;
        IOsw.start();
        sparse_distance_matrix dist =
                read_sparse_distance_matrix(filename ? file_stream : std::cin);
        IOsw.stop();
#ifdef PROFILING
        std::cerr<<IOsw.ms()/1000.0<<"s time to load sparse distance matrix (I/O)"<<std::endl;
#endif
        assert(dist.num_entries%2==0);
#ifdef COUNTING
        std::cout << "sparse distance matrix with " << dist.size() << " points and "
                  << dist.num_entries/2 << "/" << (dist.size() * (dist.size() - 1)) / 2 << " entries"
                  << std::endl;
#endif
        ripser<sparse_distance_matrix>(std::move(dist), dim_max, threshold, ratio)
                .compute_barcodes();
    }else {

        Stopwatch IOsw;
        IOsw.start();
        compressed_lower_distance_matrix dist= read_file(filename ? file_stream : std::cin, format);
        IOsw.stop();
#ifdef PROFILING
        std::cerr<<IOsw.ms()/1000.0<<"s time to load distance matrix (I/O)"<<std::endl;
#endif
        value_t min= std::numeric_limits<value_t>::infinity(),
                max= -std::numeric_limits<value_t>::infinity(), max_finite= max;
        int num_edges= 0;

        value_t enclosing_radius= std::numeric_limits<value_t>::infinity();
        for (index_t i= 0; i < dist.size(); ++i) {
            value_t r_i= -std::numeric_limits<value_t>::infinity();
            for (index_t j= 0; j < dist.size(); ++j) r_i= std::max(r_i, dist(i, j));
            enclosing_radius= std::min(enclosing_radius, r_i);
        }

        if (threshold == std::numeric_limits<value_t>::max()) threshold= enclosing_radius;

        for (auto d : dist.distances) {
            min= std::min(min, d);
            max= std::max(max, d);
            max_finite= d != std::numeric_limits<value_t>::infinity() ? std::max(max, d) : max_finite;
            if (d <= threshold) ++num_edges;
        }
#ifdef COUNTING
        std::cout << "value range: [" << min << "," << max_finite << "]" << std::endl;
#endif


        if (use_sparse) {
#ifdef COUNTING
            std::cout << "sparse distance matrix with " << dist.size() << " points and "
                      << num_edges << "/" << (dist.size() * (dist.size() - 1)) / 2 << " entries"
                      << std::endl;
#endif
            ripser<sparse_distance_matrix>(sparse_distance_matrix(std::move(dist), threshold),
                                           dim_max, threshold, ratio)
                    .compute_barcodes();
        } else {
#ifdef COUNTING
            std::cout << "distance matrix with " << dist.size() << " points" << std::endl;
#endif
            ripser<compressed_lower_distance_matrix>(std::move(dist), dim_max, threshold, ratio).compute_barcodes();
        }
    }
    sw.stop();
#ifdef INDICATE_PROGRESS
    std::cerr<<clear_line<<std::flush;
#endif
#ifdef PROFILING
    std::cerr<<"total time: "<<sw.ms()/1000.0<<"s"<<std::endl;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaMemGetInfo(&freeMem_end,&totalMemory);

    std::cerr<<"total GPU memory used: "<<(freeMem_start-freeMem_end)/1000.0/1000.0/1000.0<<"GB"<<std::endl;
#endif

    set_of_barcodes* collected_barcodes = (set_of_barcodes*)malloc(sizeof(set_of_barcodes) * list_of_barcodes.size());
    for(index_t i = 0; i < list_of_barcodes.size();i++){
        birth_death_coordinate* barcode_array = (birth_death_coordinate*)malloc(sizeof(birth_death_coordinate) * list_of_barcodes[i].size());

        index_t j;
        for(j = 0; j < list_of_barcodes[i].size(); j++){
            barcode_array[j] = list_of_barcodes[i][j];
        }
        collected_barcodes[i] = {j,barcode_array};
    }

    res = {(int)(dim_max + 1),collected_barcodes};

    return res;
}

extern "C" ripser_plusplus_result run_main(int argc, char** argv, value_t* matrix, int num_entries, int num_rows, int num_columns) {

    Stopwatch sw;
#ifdef PROFILING
    cudaDeviceProp deviceProp;
    size_t freeMem_start, freeMem_end, totalMemory;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaMemGetInfo(&freeMem_start,&totalMemory);
#endif
    sw.start();
    const char* filename= nullptr;

    file_format format= DISTANCE_MATRIX;

    index_t dim_max= 1;
    value_t threshold= std::numeric_limits<value_t>::max();

    float ratio= 1;

    bool use_sparse= false;

    for (index_t i= 0; i < argc; i++) {
        const std::string arg(argv[i]);
        if (arg == "--help") {
            print_usage_and_exit(0);
        } else if (arg == "--dim") {
            std::string parameter= std::string(argv[++i]);
            size_t next_pos;
            dim_max= std::stol(parameter, &next_pos);
            if (next_pos != parameter.size()) print_usage_and_exit(-1);
        } else if (arg == "--threshold") {
            std::string parameter= std::string(argv[++i]);
            size_t next_pos;
            threshold= std::stof(parameter, &next_pos);
            if (next_pos != parameter.size()) print_usage_and_exit(-1);
        } else if (arg == "--ratio") {
            std::string parameter= std::string(argv[++i]);
            size_t next_pos;
            ratio= std::stof(parameter, &next_pos);
            if (next_pos != parameter.size()) print_usage_and_exit(-1);
        } else if (arg == "--format") {
            std::string parameter= std::string(argv[++i]);
            if (parameter == "lower-distance")
                format= LOWER_DISTANCE_MATRIX;
            else if (parameter == "distance")
                format= DISTANCE_MATRIX;
            else if (parameter == "point-cloud")
                format= POINT_CLOUD;
            else if (parameter == "dipha")
                format= DIPHA;
            else if (parameter == "sparse") {
                format= SPARSE;
                use_sparse = true;
            }
            else if (parameter == "binary")
                format= BINARY;
            else
                print_usage_and_exit(-1);
        } else if(arg=="--sparse") {
            use_sparse= true;
        }
    }

    list_of_barcodes = std::vector<std::vector<birth_death_coordinate>>();
    for(index_t i = 0; i <= dim_max; i++){
        list_of_barcodes.push_back(std::vector<birth_death_coordinate>());
    }

    if (format == SPARSE) {//this branch is currently unsupported in run_main, see run_main_filename() instead
        Stopwatch IOsw;
        IOsw.start();


        sparse_distance_matrix dist= read_sparse_distance_matrix_python(matrix,num_entries);

        IOsw.stop();
#ifdef PROFILING
        std::cerr<<IOsw.ms()/1000.0<<"s time to load sparse distance matrix (I/O)"<<std::endl;
#endif
        assert(dist.num_entries%2==0);
#ifdef COUNTING
        std::cout << "sparse distance matrix with " << dist.size() << " points and "
                  << dist.num_entries/2 << "/" << (dist.size() * (dist.size() - 1)) / 2 << " entries"
                  << std::endl;
#endif
        ripser<sparse_distance_matrix>(std::move(dist), dim_max, threshold, ratio)
                .compute_barcodes();
    }else{
        //Stopwatch IOsw;
        //IOsw.start();
        compressed_lower_distance_matrix dist= read_matrix_python(matrix, num_entries, num_rows, num_columns, format);
        //IOsw.stop();

#ifdef PROFILING
        //std::cerr<<IOsw.ms()/1000.0<<"s time to load python matrix"<<std::endl;
        std::cerr<<"loaded python dense user matrix"<<std::endl;
#endif
        value_t min= std::numeric_limits<value_t>::infinity(),
                max= -std::numeric_limits<value_t>::infinity(), max_finite= max;
        int num_edges= 0;

        value_t enclosing_radius= std::numeric_limits<value_t>::infinity();
        for (index_t i= 0; i < dist.size(); ++i) {
            value_t r_i= -std::numeric_limits<value_t>::infinity();
            for (index_t j= 0; j < dist.size(); ++j) r_i= std::max(r_i, dist(i, j));
            enclosing_radius= std::min(enclosing_radius, r_i);
        }

        if (threshold == std::numeric_limits<value_t>::max()) threshold= enclosing_radius;

        for (auto d : dist.distances) {
            min= std::min(min, d);
            max= std::max(max, d);
            max_finite= d != std::numeric_limits<value_t>::infinity() ? std::max(max, d) : max_finite;
            if (d <= threshold) ++num_edges;
        }
#ifdef COUNTING
        std::cout << "value range: [" << min << "," << max_finite << "]" << std::endl;
#endif
        if (use_sparse) {
#ifdef COUNTING
            std::cout << "sparse distance matrix with " << dist.size() << " points and "
                      << num_edges << "/" << (dist.size() * (dist.size() - 1)) / 2 << " entries"
                      << std::endl;
#endif
            ripser<sparse_distance_matrix>(sparse_distance_matrix(std::move(dist), threshold),
                                           dim_max, threshold, ratio)
                    .compute_barcodes();
        } else {
#ifdef COUNTING
            std::cout << "distance matrix with " << dist.size() << " points" << std::endl;
#endif
            ripser<compressed_lower_distance_matrix>(std::move(dist), dim_max, threshold, ratio).compute_barcodes();
        }
    }
    sw.stop();
#ifdef INDICATE_PROGRESS
    std::cerr<<clear_line<<std::flush;
#endif
#ifdef PROFILING
    std::cerr<<"total time: "<<sw.ms()/1000.0<<"s"<<std::endl;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaMemGetInfo(&freeMem_end,&totalMemory);

    std::cerr<<"total GPU memory used: "<<(freeMem_start-freeMem_end)/1000.0/1000.0/1000.0<<"GB"<<std::endl;
#endif

    set_of_barcodes* collected_barcodes = (set_of_barcodes*)malloc(sizeof(set_of_barcodes) * list_of_barcodes.size());
    for(index_t i = 0; i < list_of_barcodes.size();i++){
        birth_death_coordinate* barcode_array = (birth_death_coordinate*)malloc(sizeof(birth_death_coordinate) * list_of_barcodes[i].size());

        index_t j;
        for(j = 0; j < list_of_barcodes[i].size(); j++){
            barcode_array[j] = list_of_barcodes[i][j];
        }
        collected_barcodes[i] = {j,barcode_array};
    }

    res = {(int)(dim_max + 1),collected_barcodes};

    return res;
}

int main(int argc, char** argv) {

    Stopwatch sw;
#ifdef PROFILING
    cudaDeviceProp deviceProp;
    size_t freeMem_start, freeMem_end, totalMemory;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaMemGetInfo(&freeMem_start,&totalMemory);
#endif
    sw.start();
    const char* filename= nullptr;

    file_format format= DISTANCE_MATRIX;

    index_t dim_max= 1;
    value_t threshold= std::numeric_limits<value_t>::max();

    float ratio= 1;

    bool use_sparse= false;

    for (index_t i= 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help") {
            print_usage_and_exit(0);
        } else if (arg == "--dim") {
            std::string parameter= std::string(argv[++i]);
            size_t next_pos;
            dim_max= std::stol(parameter, &next_pos);
            if (next_pos != parameter.size()) print_usage_and_exit(-1);
        } else if (arg == "--threshold") {
            std::string parameter= std::string(argv[++i]);
            size_t next_pos;
            threshold= std::stof(parameter, &next_pos);
            if (next_pos != parameter.size()) print_usage_and_exit(-1);
        } else if (arg == "--ratio") {
            std::string parameter= std::string(argv[++i]);
            size_t next_pos;
            ratio= std::stof(parameter, &next_pos);
            if (next_pos != parameter.size()) print_usage_and_exit(-1);
        } else if (arg == "--format") {
            std::string parameter= std::string(argv[++i]);
            if (parameter == "lower-distance")
                format= LOWER_DISTANCE_MATRIX;
            else if (parameter == "distance")
                format= DISTANCE_MATRIX;
            else if (parameter == "point-cloud")
                format= POINT_CLOUD;
            else if (parameter == "dipha")
                format= DIPHA;
            else if (parameter == "sparse")
                format= SPARSE;
            else if (parameter == "binary")
                format= BINARY;
            else
                print_usage_and_exit(-1);
        } else if(arg=="--sparse") {
            use_sparse= true;
        }else {
            if (filename) { print_usage_and_exit(-1); }
            filename= argv[i];
        }
    }

    list_of_barcodes = std::vector<std::vector<birth_death_coordinate>>();
    for(index_t i = 0; i <= dim_max; i++){
        list_of_barcodes.push_back(std::vector<birth_death_coordinate>());
    }

    std::ifstream file_stream(filename);
    if (filename && file_stream.fail()) {
        std::cerr << "couldn't open file " << filename << std::endl;
        exit(-1);
    }
    if (format == SPARSE) {
        Stopwatch IOsw;
        IOsw.start();
        sparse_distance_matrix dist =
                read_sparse_distance_matrix(filename ? file_stream : std::cin);
        IOsw.stop();
#ifdef PROFILING
        std::cerr<<IOsw.ms()/1000.0<<"s time to load sparse distance matrix (I/O)"<<std::endl;
#endif
        assert(dist.num_entries%2==0);
#ifdef COUNTING
        std::cout << "sparse distance matrix with " << dist.size() << " points and "
                  << dist.num_entries/2 << "/" << (dist.size() * (dist.size() - 1)) / 2 << " entries"
                  << std::endl;
#endif
        ripser<sparse_distance_matrix>(std::move(dist), dim_max, threshold, ratio)
                .compute_barcodes();
    }else {

        Stopwatch IOsw;
        IOsw.start();
        compressed_lower_distance_matrix dist= read_file(filename ? file_stream : std::cin, format);
        IOsw.stop();
#ifdef PROFILING
        std::cerr<<IOsw.ms()/1000.0<<"s time to load distance matrix (I/O)"<<std::endl;
#endif
        value_t min= std::numeric_limits<value_t>::infinity(),
                max= -std::numeric_limits<value_t>::infinity(), max_finite= max;
        int num_edges= 0;

        value_t enclosing_radius= std::numeric_limits<value_t>::infinity();
        for (index_t i= 0; i < dist.size(); ++i) {
            value_t r_i= -std::numeric_limits<value_t>::infinity();
            for (index_t j= 0; j < dist.size(); ++j) r_i= std::max(r_i, dist(i, j));
            enclosing_radius= std::min(enclosing_radius, r_i);
        }

        if (threshold == std::numeric_limits<value_t>::max()) threshold= enclosing_radius;

        for (auto d : dist.distances) {
            min= std::min(min, d);
            max= std::max(max, d);
            max_finite= d != std::numeric_limits<value_t>::infinity() ? std::max(max, d) : max_finite;
            if (d <= threshold) ++num_edges;
        }
#ifdef COUNTING
        std::cout << "value range: [" << min << "," << max_finite << "]" << std::endl;
#endif
        if (use_sparse) {
#ifdef COUNTING
            std::cout << "sparse distance matrix with " << dist.size() << " points and "
                      << num_edges << "/" << (dist.size() * (dist.size() - 1)) / 2 << " entries"
                      << std::endl;
#endif
            ripser<sparse_distance_matrix>(sparse_distance_matrix(std::move(dist), threshold),
                                           dim_max, threshold, ratio)
                    .compute_barcodes();
        } else {
#ifdef COUNTING
            std::cout << "distance matrix with " << dist.size() << " points" << std::endl;
#endif
            ripser<compressed_lower_distance_matrix>(std::move(dist), dim_max, threshold, ratio).compute_barcodes();
        }
    }
    sw.stop();
#ifdef INDICATE_PROGRESS
    std::cerr<<clear_line<<std::flush;
#endif

#ifdef PROFILING
    std::cerr<<"total time: "<<sw.ms()/1000.0<<"s"<<std::endl;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaMemGetInfo(&freeMem_end,&totalMemory);

    std::cerr<<"total GPU memory used: "<<(freeMem_start-freeMem_end)/1000.0/1000.0/1000.0<<"GB"<<std::endl;
#endif
}
