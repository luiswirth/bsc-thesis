= Introduction

Partial Differential Equations (PDEs) are the mathematical language used to
model continuum problems across physics and engineering. From heat transfer
and fluid dynamics to electromagnetism and quantum mechanics, PDEs describe
the fundamental systems that govern our universe. Solving PDEs efficiently
and accurately is therefore central to modern computational science.
@hiptmair:numpde

The Finite Element Method (FEM) is one of the most important methods employed to
numerically solve PDEs, particularly on unstructured meshes, drawing inspiration
from functional analysis @hiptmair:numpde.
While scalar-valued FEM using Lagrangian finite element spaces is
well-established, extending FEM to vector-valued problems traditionally involved
intricate constructions. @hiptmair:whitneyforms

#quote(block: true, attribution: [Hiptmair @hiptmair:whitneyforms])[
Without referring to differential geometry, several authors had devised vector
valued finite elements that can be regarded as special cases of discrete
differential forms. Their constructions are formidably intricate and require
much technical effort. A substantial simplification can be achieved: One should
exploit the facilities provided by differential geometry for a completely
coordinate-free treatment of discrete differential forms. Once we have shed
the cumbersome vector calculus, everything can be constructed and studied with
unmatched generality and elegance. This can be done for simplicial meshes in
arbitrary dimension.
]

Finite Element Exterior Calculus (FEEC) developed by Douglas Arnold and Richard
Falk and Ragnar Winther @douglas:feec-article, @douglas:feec-book is exactly
this unified mathematical framework. By employing the language of differential
geometry @frankel:diffgeo and algebraic topology @hatcher:algtop, FEEC extends
FEM to handle problems involving differential forms of arbitrary rank. This
approach offers robust discretizations that preserve key topological and
geometric structures inherent in the underlying PDEs, ensuring stability,
accuracy, and convergence @douglas:feec-article. FEEC is now the standard
framework for analyzing and constructing conforming finite element spaces for
differential forms in arbitrary dimensions and on domains with complex topology
@douglas:feec-book.

A key strength of FEEC lies in its ability to naturally handle arbitrary domain
topologies. This relies on fundamental connections between the algebraic topology
of the simplicial complex discretizing the domain (simplicial homology @hatcher:algtop)
and the structure of differential forms on the continuous domain (de Rham cohomology
@frankel:diffgeo). The de Rham theorem establishes an isomorphism, ensuring that the
discrete formulation accurately captures topological features, such as holes, which
influence the existence and uniqueness of PDE solutions. @douglas:feec-book

This theoretical elegance, however, is in stark contrast to many existing FEM
software implementations, which are usually hard-coded to 3 dimensions and rely
on vector proxies instead of actual differential forms and exterior algebra.
Furthermore, almost all implementations make use of global coordinates on the
manifolds, therefore relying on embeddings instead of the intrinsic geometry of
the manifold.

This thesis presents a different approach, fully embracing the coordinate-free
perspective inherent in differential geometry for unparalleled generality.
Since FEEC is formulated using differential geometry, PDE domains can be treated
as abstract Riemannian manifolds @frankel:diffgeo. We develop a novel finite
element library in the Rust programming language @RustLang that operates on
such abstract simplicial complexes @hatcher:algtop in arbitrary dimensions,
avoiding any reliance on coordinate embeddings. The geometry is defined purely
intrinsically via a Riemannian metric @frankel:diffgeo derived from edge lengths
using Regge Calculus @regge.

We restrict this work to first-order methods, employing piecewise linear
Whitney forms @whitney:geointegration, @douglas:feec-article as basis functions.
This corresponds to a piecewise-flat approximation of the manifold geometry,
which constitutes an admissible geometric variational crime @holst:gvc.

The prototypical second-order elliptic operator in FEEC is the Hodge-Laplace
operator @frankel:diffgeo, a generalization of the standard Laplace-Beltrami
operator. Central to its analysis is Hodge theory @frankel:diffgeo, which provides
the crucial link between this elliptic operator, the topology of the manifold via
cohomology, and its kernel (the space of harmonic forms). Our implementation is
guided by the goal of solving the Hodge-Laplace eigenvalue and source problems on
the $n$D de Rham complex @douglas:feec-article. For both problems, we utilize a
mixed weak formulation @douglas:feec-article, @douglas:feec-book.

The choice of Rust @RustLang stems from its strong guarantees in memory safety,
performance comparable to C/C++, and modern language features suitable for
complex scientific software. Its ownership model, borrow checker, and type
system act like a proof system, preventing entire classes of bugs like memory
errors and data races, which is crucial for reliable high-performance
computing.
