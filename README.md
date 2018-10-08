# culip

![culip logo](doc/logo.png)

## Project description
The **cu**da for **l**inear and **i**nteger **p**rogramming project contains
a collection of GPU primitives for
* linear algebra,
* linear programming and
* integer programming.

## Source directory structure
* apps - Executables using the modules in lib/
* dependencies - Building scripts for external dependencies
* libs - This is where the magic happens - all functional/backend code pieces
  * algorithms - symbolical/discrete CPU algorithms
  * data_structures - specialized, templated CPU data structures (ixheap, kvheap, ...)
  * la - Linear algebra modules (wrapper, kkt adapters, iterative solvers, preconditioners)
  * util - various utilities, such as a memory manager
* tests - Tests for executables and libs

## Dependencies
* [NVIDIA Thrust](https://developer.nvidia.com/thrust)
* [NVIDIA CUB](https://nvlabs.github.io/cub/)
* [MMIO](https://math.nist.gov/MatrixMarket/)
* [CBlas](https://www.netlib.org/blas/)
* [LAPACKE](http://www.netlib.org/lapack/lapacke.html)
* [OpenBLAS](https://www.openblas.net/)

All dependencies are downloaded and built automatically in the provided CMake build process.

## Get started
The build process is based on CMake and has been tested on Ubuntu 18.04 with gcc 7.3.0 and CUDA >= 9.2.
Just execute the following after cloning and cd'ing into the cloned folder:
```
$ mkdir build
$ cd build
$ cmake ..
$ ccmake .
$ make
```

In the GUI appearing after `ccmake`, select the parts you wish to build and enter the `CUDA_SDK_ROOT_DIR`, which
should point to the base dir of NVIDIA's CUDA samples included in their CUDA distribution.

## License and Copyright
Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved. <br />
Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.

This software may be modified and distributed under the terms
of the BSD 3-clause license. See the LICENSE file for details.
