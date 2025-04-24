#import "../setup.typ": *
#import "../setup-math.typ": *
#import "../layout.typ": *


= Conclusion and Outlook

== Failed Ideas

Discarded ideas and failed approaches

=== Compile-Time Type-Level Programming

Tried to introduce generic (static and dynamic) dimensionality à la nalgebra/Eigen.

=== Abstracting over `Simplex` and `ExteriorTerm` using `MultiIndex`

Overabstraction is worse than minor code duplication.

== Future Work

=== Varying-Coefficents & Quadrature
=== Higher-Order FEEC: Higher-Order Manifold & Higher-Order Elements


===  Maxwell's Equations

A far more meaningful PDE system that has some really interesting applications in real-life
are Maxwell's Equation describing Electromagnetism.
FEEC is the perfect fit for Maxwell's equations, since the relativistic variant of them
is also formulated in terms of differential geometry as is general relativity.
This means that purely thanks to the generality of the library we are able to solve
Maxwell's equations on the curved 4D spacetime manifold.

== Comparison to Other Implementations

We researched which other implementation related to FEEC exist.
There are other major libraries.
Formoniq didn't draw any inspiration from them.
Only after the fact we looked at them.

We want to quickly do a comparison of formoniq and these other implementations.

=== Formoniq

Focus on FEEC (FEM)
Arbitrary Dimensions
Simplicial Complexes
Differential Forms & Exterior Algebra
1st order whitney form
Intrinsic Geometry

Available on
#weblink(
  "https://github.com/luiswirth/formoniq.jl",
  [github:luiswirth/formoniq]
).

=== PyDEC

Focus on DEC + some FEEC
Arbitrary Dimensions
Simplicial and Cubical Complexes
Intrinsic and Extrinsic Geometry (Embedded and abstract complexes)
1st order whitney forms


Simplicial Complexes of any dimension. Embedded or Intrinsic.
Cubical Cpmples of any dimension.

PyDEC seems to be the most mature implementation of DEC and 1st order FEEC.
It implements 1st order Whitney Forms and a Hodge Mass matrix.
Is support simplicial and cubical complexes (maybe not for FEEC?)
It can compute cohomology and Hodge decompositions.

Available on
#weblink(
  "https://github.com/hirani/pydec",
  [github:hirani/pydec],
).
@pydec

=== FEEC++ / simplefem


Focus on FEEC
Hardcoded 2D and 3D
Intrinsic geometry
Arbitrary order polynomial differential forms

By Martin Licht from EPFL.
Work in progress.

FEEC++ implements finite element spaces of *arbitrary (uniform) polynomial degree*
over simplicial meshes, including Whitney forms.

Hard-coded Simplicial meshes in dimensions 1, 2, and 3.

Available on
#weblink(
  "https://github.com/martinlicht/simplefem",
  [github:martinlicht/simplefem]
).
@feecpp

=== DDF.jl

No focus on PDEs, but mostly DEC and FEEC
Arbitrary dimensions
Simplicial mesh
Intrinsic geometry
Higher-order discretizations

Work in progres..

Simplicial meshes of arbitrary dimension.
Arbitrary order?

In Julia there is quite an ecosystem for Exterior Algebra/Calculus.


It builds on top of this library.
https://github.com/eschnett/DifferentialForms.jl/tree/master

Available on
#weblink(
  "https://github.com/eschnett/DDF.jl",
  [github:eschnett/DDF.jl]
).
@ddfjl

=== dexterior

The dexterior library is a Rust-based toolkit for Discrete Exterior Calculus
(DEC), developed by Mikael Myyrä. It offers foundational components for
discretizing partial differential equations (PDEs) using DEC principles.

dexterior provides building blocks for the discretization of partial
differential equations using the mathematical framework of Discrete Exterior
Calculus (DEC). These building blocks are sparse matrix operators (exterior
derivative, Hodge star) and vector operands (cochains) associated with
a simplicial mesh of any dimension. An effort is made to provide as many
compile-time checks and type inference opportunities as possible to produce
concise and correct code.

Any dimensions.
Simplicial Complex.
Extrinsic Geometry based on Embedding
Focus on DEC, no FEEC.

Visualization using wgpu

Heavily inspired by PyDEC, but in Rust.

Available on
#weblink(
  "https://github.com/m0lentum/dexterior",
  [github:m0lentum/dexterior]
).
@dexterior
