= Introduction 

Partial Differential Equations (PDEs) are the mathematical language we use to
model continuum problems across physics and engineering. From heat transfer
and fluid dynamics to electromagnetism and quantum mechanics, PDEs describe
the fundamental systems that govern our universe. Because of this, solving PDEs
efficiently and accurately is central to modern computational science.

The Finite Element Method (FEM) has revolutionized PDE solving by discretizing
domains into manageable pieces, enabling high-precision simulations of complex
systems. FEM has become indispensable in structural mechanics, aerodynamics, and
electromagnetics, where computational models drive innovation.

While scalar-valued FEM is realtively easily constructed and studied
using Lagrangian FE spaces, vector-valued FEM is far more involved.
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
for scalar- and vector-valued FEM, making use of the far more general and elegant theory of differential
geometry and the exterior calculus of differential forms instead of vector calculus to solve PDEs on curved
Riemannian manifolds in arbitrary dimensions with any topology.
FEEC provides structure-preserving discretizations that ensure stability,
accuracy, and convergence, particularly for problems involving differential
forms.

FEEC has mostly been used to analyze and construct standard vector-valued FEM
that don't embrace differential geometry and exterior calculus in it's implementation.
Meaning the implementations are constrained to at most 3 dimensions
and no differential forms are used but instead only scalarfield and vectorfield (proxies).
The manifolds are given global coordinates and therefore don't respect the
intrinsic nature of differential geometry.

This thesis takes a different approach. We want to fully embrace differential geometry
and in this way provide a implementation of FEEC that works in arbitrary dimensions
for arbitrary Riemannian manifolds without any global coordinates but only
an intrinsic Riemannian metric.

We will restrict ourselves to 1st order piecewise linear FEM and therefore
also just piecewise-flat approximations of the underlying manifold (admissable geometric variational crime).
These approximations are in the forms of a simplicial complex.

For this, we are using the Rust programming language, leveraging its performance,
safety, and concurrency to create a robust tool.
This library brings FEEC into computational practice, bridging the gap between
theory and application.

The thesis aims to lower the bar of entry to the theory of FEEC by providing
a beginner friendly exposition of the main concepts.
Furthermore we provide a implementation of a FEEC library that should be useable
for solving real-life PDEs. It should also guide as a reference for future implementations
of FEEC in other programming languages with different paradigmes.
We want to to lay out the necessary steps without relying too much on the vast and complicated mathematical framework
that was created around FEEC. This thesis is more pragmatic and should appeal to a wider audiance
than the original books and papers on FEEC.

The prototypical PDE in FEEC is the elliptic Hodge-Laplace Source Problem.
Which we will mainly focus on and will be guiding the implementation.

A far more meaningful PDE system that has some really interesting applications in real-life
are Maxwell's Equation describing Electromagnetism.
FEEC is the perfect fit for Maxwell's equations, since the relativistic variant of them
is also formulated in terms of differential geometry as is general relativity.
This means that purely thanks to the generality of the library we are able to solve
Maxwell's equations on the curved 4D spacetime manifold.


Relevant mathematical theories to this thesis:
- Algebraic Topology
- Differential Geometry
- Exterior Algebra and Calculus
- Homology
- Functional Analysis

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

After this we take a dive into homology theory, which is topological disciple
concerned with the couting of holes of a topological space, in our case
the simplicial complex which approximates our PDE domain.
The simplicial homology of our mesh influences the de Rham cohomology
which makes statement about the existance of differential forms
and therefore has influence on the existance and uniqueness of our PDE problem.
A lot of the magic of FEEC lies in homology.

Once we've explored al of these various theories relevant to the implementation,
we will finally talk about the implementation of the heart of our FEEC library,
which is the computation of the galerkin matrices for various weak differential operator.
We will both be solving the Hodge-Laplace EVP and the Hodge-Laplace source problem.
For this we will look at the Whitney basis functions and the Whitney FE space.

Lastly we will test our library and generate various results by solving
model problems, visualizing them and looking at error convergences and other metrics.

#pagebreak()
