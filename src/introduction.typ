= Introduction 

Partial Differential Equations (PDEs) are the mathematical language we use to
model continuum problems across physics and engineering.
From heat transfer and fluid dynamics to electromagnetism and quantum mechanics,
PDEs describe the fundamental systems that govern our universe.
Because of this, solving PDEs efficiently and accurately is central to modern
computational science.

The Finite Element Method (FEM) is one of the major methods employed to
numerically solve PDEs on unstructured meshes inspired by ideas from functional
analysis.

While scalar-valued FEM is relatively easily constructed and studied using
Lagrangian FE spaces, vector-valued FEM is far more involved.
#quote(block: true, attribution: [Hiptmair @hiptmair-whitney])[
Without referring to differential geometry, several authors had devised vector
valued finite elements that can be regarded as special cases of discrete
differential forms. Their constructions are formidably intricate and require
much technical effort. A substantial simplification can be achieved: One should
exploit the facilities provided by differential geometry for a completely
coordinate-free treatment of discrete differential forms. Once we have shed
the cumbersome vector calculus, everything can be constructed and studied with
unmatched generality and elegance. In particular, all orders of forms and all
degrees of polynomial approximation can be dealt with in the same framework.
This can be done for simplicial meshes in arbitrary dimension.
]

Finite Element Exterior Calculus (FEEC) is this unified mathematical framework
for scalar- and vector-valued FEM, making use of the far more general
and elegant theory of differential geometry and the exterior calculus of
differential forms instead of vector calculus to solve PDEs.
FEEC provides structure-preserving discretizations that ensure stability,
accuracy, and convergence.

FEEC is the de facto standard for analyzing and constructing conforming
FE spaces for arbitrary rank differential forms in arbitrary dimensions on
arbitrary topologies.
In modern FEM theory is therefore standard to embrace differential geometry
instead of vectors calculus.
This is in stark contrast to FEM software implementations, which are usually
hard-coded to 3 dimensions and rely on vector proxies instead of actual
differential forms and exterior algebra.
Furthermore almost all implementations make use of global coordinates on the
manifolds, therefore relying on embeddings instead of the intrinsic geometry
nature of the manifold.

This thesis takes a different approach to implementation of FEM.
We want to fully embrace differential geometry and in this way provide a
implementation of FEEC that works in arbitrary dimensions for arbitrary
Riemannian manifolds with arbitrary simplicial topology without any global
coordinates but only an intrinsic Riemannian metric.

We will restrict ourselves to 1st order piecewise linear FE and therefore
also just piecewise-flat approximations of the underlying manifold
(an admissible geometric variational crime).

The prototypical 2nd order elliptic differential operator in FEEC is the
Hodge-Laplace operator, a generalization of the ordinary Laplace-Beltrami operator.
We will work to solve the eigenvalue problem and source problem corresponding to
this operator, which will guide our implementation.
For both these problems we rely on a mixed weak formulation of Hodge-Laplacian.

For the treatment of arbitrary topologies, a big theme is homology and cohomology.
Homology theory is topological disciple concerned with the counting of holes of
a topological space, in our case the simplicial complex which approximates our
PDE domain.
The simplicial homology of our mesh is isomorphic to the de Rham cohomology of
the space of differential forms.
It makes statement about the existence of differential forms on our domain and
therefore has influence on the existence and uniqueness of our PDE problem.
The ability of FEEC to treat arbitrary topologies is thanks to homology theory.
In the concrete case of the Hodge-Laplace operator we are dealing with Hodge
theory.
We will not explain homology in detail in this thesis, since in the implementation
it has only a very small part.

*Rust*\
The implementation of our FEM library will be done in the Rust programming language.
Rust was chosen for its strong guarantees in memory safety, performance, and
modern language features, making it ideal for high-performance computing tasks
like finite elements. The Rust ownership model, borrow checker, and type system
act as a proof system to ensure there are no memory bugs, race conditions, or
similar undefined behaviors in any program, while achieving performance levels
comparable to C/C++.

*Goals and Contributions*\
The thesis aims to lower the bar of entry to the theory of FEEC by providing a
beginner friendly exposition of the main concepts.
Furthermore we provide a implementation of a FEEC library that should be useable
for solving real-life PDEs.
It should also guide as a reference for future implementations of FEEC in other
programming languages with different paradigms.
We want to to lay out the necessary steps without relying too much on the vast
and complicated mathematical framework that was created around FEEC.
This thesis is more pragmatic and should appeal to a wider audience than the
original books and papers on FEEC.

*Outline of the thesis structure*\
Our library is the core of the thesis, so the structure of it should parallel
the structure of the library.
The first chapter provide context: Rust choices and software architecture.
- The next sections introduce the mathematical foundations that your crates encapsulate.
- The final sections describe how Formoniq ties everything together and its practical application.

The structure of this thesis is as follows:

We first learn about the building blocks of our mesh,
which are simplicies and look at some algebraic topology.

We first start with some topology and differential geometry
and develope a mesh data structure.
The topology of the mesh is represented as a simplicial complex
and the geometry is given coordinate-free in the form of a
Riemannian metric, in our case the Regge metric, which solely
relies on the edge lengths of the mesh.

Then we take a look at exterior algebra and exterior calculus.
Exterior algebra is to exterior calculus, what vector algebra is to
vector calculus.
In this section we will learn about differential forms, which are
the values our PDEs will take. We will learn about the most important
properties of them and also learn about their discrete counterparts called cochains.

Once we've explored al of these various theories relevant to the implementation,
we will finally talk about the implementation of the heart of our FEEC library,
which is the computation of the Galerkin matrices for various weak differential operators.
We will both be solving the Hodge-Laplace EVP and the Hodge-Laplace source problem.
For this we will look at the Whitney basis functions and the Whitney FE space.

Lastly we will test our library and generate various results by solving
model problems, visualizing them and looking at error convergences and other metrics.
To this end we will use the method of manufactured solutions to verify
the correctness of our FE solutions.


#pagebreak()
