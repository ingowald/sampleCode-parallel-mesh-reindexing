# GPGPU-Parallel Re-indexing of Triangle (or other Polygonal or General-Unstructured) Meshes with Duplicate-Vertex and Unused-Vertex Removal

TL/DR: This repository contains sample code for the accompanying 
whitepaper of the same title.

## Background

Over the years I many times stumbled over the need for "cleaning up" a
triangle (or quad, or unstructured, or whatever) mesh such that it
wouldn't contain either unused or duplicate vertices. Usually these
operations were required after operations like merging or splitting
meshes (e.g., by material ID), etcpp; however, at some point I
realized two things:

a) that I was doing the same operation again and again, and

b) that the most straightforward way I used for doing so didn't scale well 
on a CPU, and not at all on a GPU.

Consequently, every time I was working on larger meshes---and/or when
I wanted/needed this to run on a GPU---I ended up re-inventing
different versions of solving this problem in parallel, to varying
success.

Though I *am* certainly not guiltless when it comes to "not invented
here syndrome" (i.e., I'm often more than happy to re-invent my own
wheels even though there's plenty round things around) I at some point
realized that there seems to be no good and/or easily findable
description of the method I ended up using, and as such decided that I
should at least somehow, somewhere document that method. I am
absolutely convinced that there must be many that have solved that
problem before; albeit I could not find their solutoins.

As such, if you found this paper (and this accmpanying sample code): 
I hope it'll save you some re-inventing of your own.

## Building

Given how simple the accompanying code is I decided to not include a
CMake-file that would make it look like a larger project than it is,
or would imply a certain usage model that was neither intended nor
required. I included a linux `makefile` that assumes that you have
CUDA (`nvcc`) in your path; and any windows/visual studio user should
be able to create his own project/solution for this. 

I did split the actual CUDA implementation into its own file
(`remesh.cu`), but this is only for readability: the CUDA version does
need several smaller kernels that for readability I did not want to
throw into the same file as the one that contains the CPU reference
code, sample generator, etc. There is, however, no particular reason
that would require two separate files, so feel free to merge, or pick
only what you want.

## LICENSE

License to this code is Apache 2.0; which is basically, in plain
english: do as you wish with this code, but you can't hold me
responsible for anything that goes wrong. (for the full original text:
http://www.apache.org/licenses/LICENSE-2.0),

## Usage and Sample Code

I would assume the main usage for this code would be to copy-n-paste
(and adapt to different vertex/index types); however, the code as is
also builds a "main()" that generates and runs a sample test case
where it generates NxN unit squares whose vertices overlap, but where
each square generates its own (thus replicated) copies of these
vertices, with an additional un-used vertex in the center of each
square. The generated mesh will consequently have about 1/5th unused
vertices, and almost each other vertex is replicated 4 times; the
resulting meshes--no matter whether the scalar reference, the
CPU-parallel TBB method, or the CUDA methos is used---should contain
the same quads, but should contain no more duplicate or unused
vertices.

Called without parameters, the generated executable runs with
1000x1000 such squares; otherwise the first cmdline parameter is
supposed to be the N for generating NxN such squares. 

For verification I also write out the generated OBJ files; these can
be large, so for large N you may want to disable this.
