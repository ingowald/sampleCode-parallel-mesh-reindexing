#include <owl/common/math/vec.h>
#include <owl/common/parallel/parallel_for.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#define CUDA_CHECK( call )                                              \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      OWL_RAISE("fatal cuda error");                                    \
    }                                                                   \
  }

#define CUDA_CALL(call) CUDA_CHECK(cuda##call)

#define CUDA_CHECK2( where, call )                                      \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: CUDA call (%s) "                        \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      OWL_RAISE("fatal cuda error");                                    \
    }                                                                   \
  }

#define CUDA_SYNC_CHECK()                                       \
  {                                                             \
    cudaDeviceSynchronize();                                    \
    cudaError_t rc = cudaGetLastError();                        \
    if (rc != cudaSuccess) {                                    \
      fprintf(stderr, "error (%s: line %d): %s\n",              \
              __FILE__, __LINE__, cudaGetErrorString(rc));      \
      OWL_RAISE("fatal cuda error");                            \
    }                                                           \
  }



using namespace owl::common;



__global__
void markUsed(int *isUsed, int *idx, int num)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid < num)
    isUsed[idx[tid]] = true;
}

__global__
void setNoDup(int *noDup, vec2f *vtx, int num)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid < num)
    noDup[tid]
      = (tid == 0)
      ? 1
      : (vtx[tid] != vtx[tid-1]);
}

__global__
void replaceUnused(vec2f *out_vtx,
                   int   *isUsed,
                   vec2f *in_vtx,
                   int   *in_idx,
                   int    numVertices)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numVertices) return;

  out_vtx[tid] = in_vtx[isUsed[tid]
                        ? tid
                        : (in_idx[0])];
}

__global__
void translateVertices(int *idx,
                       int *perm,
                       int *newIdx,
                       int numIndices)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numIndices) return;

  idx[tid] = newIdx[perm[idx[tid]]];
}

__global__
void setPerm(int *perm,
             int *orgID,
             int numVertices)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numVertices) return;

  perm[orgID[tid]] = tid;
}

template<typename T>
  struct subtract_one {
    inline __host__ __device__ T operator()(T &i) const { return i-(T)1; }
  };


template<typename T>
void print(const char *tag,
           const thrust::device_vector<T> &vec)
{
  std::cout << tag << "[" << vec.size() << "]\t: ";
  for (int i=0;i<vec.size();i++)
    std::cout << vec[i] << " ";
  std::cout << std::endl;
}

// remesh using cuda; modifying the arrays in place (but not shrinking
// the d_vtx array; vtxCount is input vertex count on the way in, and
// num actively used vertices on the way out; d_idx[] gets modified in
// place. Most d_vtx and d_idx must be device arrays
void remesh_cuda(thrust::device_vector<vec2f> &vtx,
                 thrust::device_vector<int>   &idx)
{
  thrust::device_vector<int> isUsed(vtx.size());
  thrust::fill(isUsed.begin(),isUsed.end(),0);

  // print("vtx in",vtx);
  // print("idx in",idx);
  
  markUsed<<<divRoundUp((int)idx.size(),1024),1024>>>
    (thrust::raw_pointer_cast(isUsed.data()),
     thrust::raw_pointer_cast(idx.data()),
     idx.size());
  //cudaDeviceSynchronize();
  // print("used\t",isUsed);
  
  thrust::device_vector<vec2f> tmp_vtx(vtx.size());
  // print("pre-remunused",tmp_vtx);
  replaceUnused<<<divRoundUp((int)tmp_vtx.size(),1024),1024>>>
    (thrust::raw_pointer_cast(tmp_vtx.data()),
     thrust::raw_pointer_cast(isUsed.data()),
     thrust::raw_pointer_cast(vtx.data()),
     thrust::raw_pointer_cast(idx.data()),
     vtx.size());
  //cudaDeviceSynchronize();
  // print("post-remunused",tmp_vtx);
  
  // ==================================================================
  // now, sort, and keep track of permutation done in sort
  // ==================================================================
  
  thrust::device_vector<int> orgID(vtx.size());
  thrust::sequence(orgID.begin(),orgID.end());

  // print("unsortedvtx",tmp_vtx);
  thrust::stable_sort_by_key(tmp_vtx.begin(),tmp_vtx.end(),orgID.data());
  // print("sorted vtx",tmp_vtx);

  // compute no dup array
  thrust::device_vector<int> noDup(vtx.size());
  setNoDup<<<divRoundUp((int)tmp_vtx.size(),1024),1024>>>
    (thrust::raw_pointer_cast(noDup.data()),
     thrust::raw_pointer_cast(tmp_vtx.data()),
     tmp_vtx.size());
  //cudaDeviceSynchronize();
  // print("noDup",noDup);
  
  // postfix sum, and subtract one from each element
  thrust::device_vector<int> newIdx(vtx.size());
  thrust::inclusive_scan(noDup.begin(),noDup.end(),newIdx.data());
  thrust::transform(newIdx.begin(),newIdx.end(),newIdx.begin(),subtract_one<int>());
  cudaDeviceSynchronize();

  // print("newIdx",newIdx);

  // get new num vertices
  int newN = newIdx.back()+1;
  // PRINT(newN);
  
  vtx.resize(newN);
  // ... and write new vertex array (we're writing back into vtx,
  // that's what we return to the app
  thrust::scatter(tmp_vtx.begin(),tmp_vtx.end(),newIdx.begin(),vtx.begin());
  // print("new vtx",vtx);
  // ==================================================================
  // new, clean vertex array created; this one contains neither
  // duplicates nor unused vertices. now about those indices...
  // ==================================================================
  
  thrust::device_vector<int> perm(orgID.size());
  setPerm<<<divRoundUp(int(orgID.size()),1024),1024>>>
    (thrust::raw_pointer_cast(perm.data()),
     thrust::raw_pointer_cast(orgID.data()),
     orgID.size());
  // print("orgID",orgID);
  // print("perm",perm);
  // print("newIdx",newIdx);
  
  // first, compute table to reverse the permutation
  translateVertices<<<divRoundUp(int(idx.size()),1024),1024>>>
    (thrust::raw_pointer_cast(idx.data()),
     thrust::raw_pointer_cast(perm.data()),
     thrust::raw_pointer_cast(newIdx.data()),
     idx.size());
                    
  // print("idx",idx);
}
