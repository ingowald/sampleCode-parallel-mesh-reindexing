// ======================================================================== //
// Copyright 2021-2021 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <owl/common/math/vec.h>
#include <owl/common/parallel/parallel_for.h>
#include <thrust/device_vector.h>
#include <tbb/parallel_sort.h>
#include <tbb/iterators.h>
#include <map>
#include <vector>
#include <random>
#include <fstream>

using namespace owl::common;

/*! for this simple sample code we use meshes with 2D float coords and
    quadrilateral elements; this is mainly because our test cases were
    easiest to define for this type of mesh, but adaption to ther mesh
    types (e.g., 3D triangle meshes) should be straightforward */
struct Mesh {
  std::vector<vec2f> vtx;
  std::vector<vec4i> idx;
};

// remesh using cuda; modifying the arrays in place (but not shrinking
// the d_vtx array; vtxCount is input vertex count on the way in, and
// num actively used vertices on the way out; d_idx[] gets modified in
// place. Most d_vtx and d_idx must be device arrays
extern void remesh_cuda(thrust::device_vector<vec2f> &vtx,
                        thrust::device_vector<int>   &idx);

/*! produce a re-indexed mesh using the CUDA sample code */
Mesh remesh_gpu_cuda(const Mesh &in)
{
  thrust::device_vector<vec2f> d_vtx(in.vtx);
  thrust::device_vector<int> d_idx((int*)thrust::raw_pointer_cast(in.idx.data()),
                                   (int*)thrust::raw_pointer_cast(in.idx.data())
                                   +4*in.idx.size());

  double t0 = getCurrentTime();
  remesh_cuda(d_vtx,d_idx);
  double t1 = getCurrentTime();
  std::cout << "# time for remesh_cuda: " << prettyDouble(t1-t0) << "s" << std::endl;

  Mesh out;
  out.vtx.resize(d_vtx.size());
  thrust::copy(d_vtx.begin(),d_vtx.end(),out.vtx.begin());
  out.idx.resize(in.idx.size());
  thrust::copy(d_idx.begin(),d_idx.end(),(int*)out.idx.data());
  return out;
}

/*! serial CPU reference code using a std::map to track which vertices
    have already been added to an initially empty new mesh */
Mesh remesh_cpu_serial(const Mesh &in)
{
  double t0 = getCurrentTime();
  Mesh out;
  std::map<vec2f,int> remeshedIDof;
  for (int j=0;j<in.idx.size();j++) {
    vec4i in_idx = in.idx[j];
    vec4i out_idx;
    for (int i=0;i<4;i++) {
      vec2f v = in.vtx[in_idx[i]];
      auto it = remeshedIDof.find(v);
      if (it == remeshedIDof.end()) {
        int newID = (int)out.vtx.size();
        out_idx[i] = newID;
        remeshedIDof[v] = newID;
        out.vtx.push_back(v);
      }
      else {
        out_idx[i] = it->second;
      }
    }
    out.idx.push_back(out_idx);
  }
  double t1 = getCurrentTime();
  std::cout << "# time for remesh_cpu_serial: " << prettyDouble(t1-t0) << "s" << std::endl;
  return out;
}

/*! our parallel re-indexing algrithm as described in the paper, but
    implemneted on the CPU using TBB */
Mesh remesh_cpu_parallel(const Mesh &in)
{
  double t0 = getCurrentTime();
  // ------------------------------------------------------------------
  // parallel: replace all unused vertices w/ first used one
  // ------------------------------------------------------------------
  std::vector<bool> isUsed(in.vtx.size());
  // mark all vertices as potentially unused (could parallelize that,
  // but not worth it on CPU)
  std::fill(isUsed.begin(),isUsed.end(),false);
  // mark all used vertices as actually used
  parallel_for_blocked
    (0,in.idx.size(),1024,
     [&](int begin, int end){
       for (int i=begin;i<end;i++) {
         vec4i idx = in.idx[i];
         isUsed[idx.x] = true;
         isUsed[idx.y] = true;
         isUsed[idx.z] = true;
         isUsed[idx.w] = true;
       }
     });
  // create new (and re-orderable) vector of vertices that contains
  // only used vertex values
  std::vector<vec2f> vtx(in.vtx.size());
  parallel_for_blocked
    (0,vtx.size(),1024,
     [&](int begin, int end){
       for (int i=begin;i<end;i++) {
         vtx[i]
           = isUsed[i]
           ? /* if used: he actual vertex */in.vtx[i]
           : /* else: any other used vtx  */in.vtx[in.idx[0].x];
       }
     });
  
  // ==================================================================
  // now, sort, and keep track of permutation done in sort
  // ==================================================================
  std::vector<int>  orgID(in.vtx.size());
  // set orgID = {0,1,2...} (could parallelize that, but not worth it
  // on CPU)
  for (int i=0;i<orgID.size();i++) orgID[i] = i;
  auto sort_begin = tbb::make_zip_iterator(vtx.begin(), orgID.begin());
  auto sort_end   = tbb::make_zip_iterator(vtx.end(), orgID.end());
  tbb::parallel_sort(sort_begin,sort_end);

  // compute no dup array
  std::vector<bool> nodup(vtx.size());
  parallel_for_blocked
    (0,vtx.size(),1024,
     [&](int begin, int end){
       for (int i=begin;i<end;i++) {
         nodup[i]
           = i==0
           ? true
           : (vtx[i] != vtx[i-1]);
       }
     });

  // postfix sum, and subtract one from each element
  std::vector<int> newIdx(vtx.size());
  { int sum = 0; for (int i=0;i<vtx.size();i++) { sum += nodup[i]; newIdx[i] = sum-1; } }

  // new num vertices:
  int newN = newIdx.back()+1;

  // allocate out vertex of correct size...
  Mesh out;
  out.vtx.resize(newN);
  // ... and write all the vertices to their new position (we'll just
  // skip writing duplicates, they'd just overwrite to locs that
  // contain that same value already)
  parallel_for_blocked
    (0,vtx.size(),1024,
     [&](int begin, int end){
       for (int i=begin;i<end;i++) 
         if (nodup[i])
           out.vtx[newIdx[i]] = vtx[i];
     });

  // ==================================================================
  // new, clean vertex array created; this one contains neither
  // duplicates nor unused vertices. now about those indices...
  // ==================================================================

  // first, compute table to reverse the permutation
  std::vector<int> perm(orgID.size());
  for (int i=0;i<orgID.size();i++)
    perm[orgID[i]] = i;

  // now allocate the out indices (we'll just treat thse N*vec4i's as
  // if they were (4N)*int's, for convenience)
  out.idx.resize(in.idx.size());
  const int *in_idx  = (const int *)in.idx.data();
  int       *out_idx = (int       *)out.idx.data();
  parallel_for_blocked
    (0,4*in.idx.size(),1024,
     [&](int begin, int end){
       for (int i=begin;i<end;i++) 
         out_idx[i] = newIdx[perm[in_idx[i]]];
     });
  
  double t1 = getCurrentTime();
  std::cout << "# time for remesh_cpu_serial: " << prettyDouble(t1-t0) << "s" << std::endl;
  return out;
}

/*! create a test data set of NxN unit squares, with each square
    containing its own (replicated) vertices even though for inner
    squares these are the same vertices as the neighboring squares'
    corners. To also include some unused vertices each square also
    adds one un-used vertex into the center of each square.*/
Mesh generateTestMesh(int N)
{
  Mesh mesh;
  for (int i=0;i<N;i++)
    for (int j=0;j<N;j++) {
      int i0 = mesh.vtx.size();
      mesh.vtx.push_back(vec2f(i+0,j+0));
      mesh.vtx.push_back(vec2f(i+1,j+0));
      mesh.vtx.push_back(vec2f(i+0,j+1));
      mesh.vtx.push_back(vec2f(i+1,j+1));
      mesh.vtx.push_back(vec2f(i+.5f,j+.5f));
      mesh.idx.push_back(vec4i(0,1,3,2)+i0);
    }
  return mesh;
}

/* save mesh to OBJ, for debugging and verification */
void saveMesh(const std::string &fileName,
              const Mesh &mesh)
{
  std::ofstream out(fileName);
  for(auto v : mesh.vtx)
    out << "v " << v.x << " " << v.y << " 0" << std::endl;
  for(auto idx : mesh.idx)
    out << "f " << (idx.x+1) << " " << (idx.y+1) << " " << (idx.z+1) << " " << (idx.w+1) << std::endl;
}

int main(int ac, char **av)
{
  // ==================================================================
  Mesh testMesh = generateTestMesh((ac == 1) ? 1000 : std::stoi(av[1]));
  std::cout << "generated test mesh:" << std::endl;
  std::cout << "#vertices (flt2) = " << prettyNumber(testMesh.vtx.size()) << std::endl;
  std::cout << "#indices  (int4) = " << prettyNumber(testMesh.idx.size()) << std::endl;
  saveMesh("testMesh.obj",testMesh);

  // ==================================================================
  Mesh result_cpu_serial = remesh_cpu_serial(testMesh);
  std::cout << "reference serial CPU result:" << std::endl;
  std::cout << "#vertices (flt2) = " << prettyNumber(result_cpu_serial.vtx.size()) << std::endl;
  std::cout << "#indices  (int4) = " << prettyNumber(result_cpu_serial.idx.size()) << std::endl;
  saveMesh("cpu_serial.obj",result_cpu_serial);

  // ==================================================================
  Mesh result_cpu_parallel = remesh_cpu_parallel(testMesh);
  std::cout << "parallel CPU result:" << std::endl;
  std::cout << "#vertices (flt2) = " << prettyNumber(result_cpu_parallel.vtx.size()) << std::endl;
  std::cout << "#indices  (int4) = " << prettyNumber(result_cpu_parallel.idx.size()) << std::endl;
  saveMesh("cpu_parallel.obj",result_cpu_parallel);


  // ==================================================================
  Mesh result_gpu_cuda = remesh_gpu_cuda(testMesh);
  std::cout << "cuda version result:" << std::endl;
  std::cout << "#vertices (flt2) = " << prettyNumber(result_gpu_cuda.vtx.size()) << std::endl;
  std::cout << "#indices  (int4) = " << prettyNumber(result_gpu_cuda.idx.size()) << std::endl;
  saveMesh("gpu_cuda.obj",result_gpu_cuda);
}
