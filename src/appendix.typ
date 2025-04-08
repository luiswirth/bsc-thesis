#import "setup.typ": *

= Formoniq Source Code

All source code of our implementation *Formoniq* is available on GitHub.\
#[
  #set text(size: 20pt)
  #set align(center)
  #weblink("https://github.com/luiswirth/formoniq")[github:luiswirth/formoniq]
]

The main component of the Git repository is the Rust code.\
In order to run the Rust code, the Rust toolchain needs to be installed on the
machine.
This is best done through the offical Rust installer and toolchain manager
Rustup, which is easily installed by running a single command `curl` command
in the shell, which can be found on #weblink("https://rustup.rs/")[rustup.rs].

In the repository root we have a `./Cargo.toml` file that specifies a Cargo
workspace. Therefore *all Cargo commands should be run from this root directory*.
In `./crates/` all the various libraries that make up Formoniq are placed.
Each of them follows the standard structure of a Rust library.
In `./crates/<crate>/src/` the source code of the current library can be found.

Formoniq is first-and-foremost a library, so just running it doesn't make any
sense, since it is not an executable.
However we do provide various example codes that do produce executables and
directly make use of the functionality of our library.
All of these examples can be found under `./crates/formoniq/examples/`.
These examples can be run using the following command.

#[
  #set text(size: 20pt)
  #set align(center)
  ```sh
  cargo run --example <example>
  ```
]
This command compliles all crates and runs the example in the shell.
It is recommended to additionally add the `--release` flag to build in release
mode, instead of debug mode to profit from optimizations.
To figure out the names of the available examples one can look into the example
directory or use tab-completion in the cargo command.

== PETSc/SLEPc Solver

Due to the immature Rust sparse linear algebra ecosystem, we have to rely
on `C/C++` implementations of sparse solvers. \
For this we rely on *PETSc* for direct LSE solvers and on *SLEPc*, which builds
on PETSc, for eigensolvers.
We've written small PETSc and SLEPc programs that can be found under `./petsc-solver`.
These programs load the system matrices from
disk, solve the problem and write the solution back to disk.
Our Rust program interoperates with these programs, by doing this writing and
reading to disk as well, to communicate the problem setup and retrieve the solution.
It then also automatically calls the PETSc/SLEPc program.
So really the user does never directly interact with these solvers, but the
Rust code manages this itself.

All the user has to do is to the small solver programs.
For this both PETSc and SLEPc need to be installed on the system.
We refer to the offical PETSc and SLEPc documentation on #weblink("https://petsc.org/")[petsc.org]
and #weblink("https://slepc.upv.es/")[slepc.upv.es], where installation
of both software suites is explained. The steps that worked for
the author are outlined in @appendix:petsc.

Given that PETSc and SLEPc are successfully installed, we can now
build our small PETSc/SLEPc solver.
For this it is crucial that both `PETSC_DIR` and `SLEPC_DIR` are set.
```sh
export PETSC_DIR=<PATH_TO_PETSC>
export SLEPC_DIR=<PATH_TO_SLEPC>
```

Then we can navigate into the `./petsc-solver` directory and simply run `make`.
This will produce two exectuables `ghiep.o` and `hils.o`. These executables
will then be found and run by the Formoniq.

== Plotting

We create a simple python matplotlib script that is capable of visualizing Whitney
1-forms that correspond to arbitrary 1-cochains on arbitrary 2D meshes embedded
in 2D.
The necessary code can be found in `./plot` and is managed using the `uv`
python package manager, which can be found under #weblink("https://github.com/astral-sh/uv")[github:astral-sh/uv].

The usage should be clear from this `--help` message.
```sh
> uv run src/main.py --help
usage: main.py [-h] [--skip-zero] [--heatmap-res HEATMAP_RES]
               [--quiver-count QUIVER_COUNT] [--highlight]
               path

Whitney Forms Visualizer

positional arguments:
  path                  Path to the input files

options:
  -h, --help            show this help message and exit
  --skip-zero           Skip triangles with all zero DOFs
  --heatmap-res HEATMAP_RES
                        Resolution of the heatmap (default: 30)
  --quiver-count QUIVER_COUNT
                        Number of quiver arrows per triangle dimension
                        (default: 20)
  --highlight           Disable highlighting of non-zero DOF edges
```

In the directory `./plot/in/` various simple input are prepered for visualizing
local shape functions on the reference triangle and global shape functions on
the equilateral "triforce" mesh.
One can for example run the following command for some nice visuals that
are also shows in the thesis.
```sh
uv run src/main.py --quiver-count=5 --heatmap-res=10 in/triforce/
```



= Thesis Typst Source Code

The thesis document itself has been written using the new modern type-setting language
Typst, which can be found under #weblink("https://typst.app/")[typst.app].
It is very similar to LaTeX, but aims to do things better and to eleviate
various points of frustration that are common in LaTeX.

The source code for this Typst document, is available on GitHub.
#[
  #set text(size: 20pt)
  #set align(center)
  #weblink("https://github.com/luiswirth/bsc-thesis")[github:luiswirth/bsc-thesis]
]

To build the document in the form of a PDF, one needs to have the Typst
compiler installed and then only has to run the `./build.sh` script from the root

The document is also available as a project on the Typst web app.
It can be viewed under the following read-only link.
#[
  #set text(size: 20pt)
  #set align(center)
  #weblink("https://typst.app/project/rxBzpCgbFpynXA0BT9V6Ks")[typst.app/project/rxBzpCgbFpynXA0BT9V6Ks]
]





= PETSc/SLEPc Installation
<appendix:petsc>

We quickly explain the installation steps for PETSc and SLEPc that
worked for the author.

== PETSc
For PETSc we simply cloned the repository, configured the installation
and the ran make to compile all files.
```sh
git clone -b release https://gitlab.com/petsc/petsc.git petsc
cd petsc

./configure \
  --with-cc=gcc --with-cxx=g++ --with-fc=gfortran \
  --download-mpich --download-fblaslapack \
  -–download-mumps -–download-scalapack -–download-parmetis \
  -–download-metis -–download-ptscotch

make all check
```
Alternatively one could have build with MPI support, but
for us distributed computations did not work.
```
--with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90
```
Now PETSc should be successfully installed.

== SLEPc

SLEPc dependes on PETSc and therefore needs to be installed after it
and told it's location.
For this we set the `PETSC_DIR` environment variable to point to the PETSc
installation directory.
If you are inside the PETSc directory directory, then the following works.
```sh
export PETSC_DIR=$(pwd)
```
Also `PETSC_ARCH` has to be set to the architecture for which PETSc has been build.
This is the name of one of the folder that was produced in the PETSc directory.
For our linux debug build, we need to set the envvar to the following.
```sh
export PETSC_ARCH=arch-linux-c-debug
```

To do the actual SLEPc installation the steps are very similar to PETSc: We
clone the repository, where we make sure we have the same version of SLEPc as
for PETSc, then run the configure script and finally run make.
```sh
git clone https://gitlab.com/slepc/slepc
git checkout v3.23.0
cd slepc

./configure

make
make test
```

If everything went well without any errors, SLEPc should be succesfully
installed.
