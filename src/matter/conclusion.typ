#import "../setup.typ": *
#import "../setup-math.typ": *
#import "../layout.typ": *

= Conclusion and Outlook

This thesis presented `formoniq`, a novel implementation of Finite Element
Exterior Calculus (FEEC) developed in the Rust programming language. The core
contribution lies in its foundation on intrinsic, coordinate-free geometry and
its capability to operate on simplicial complexes in arbitrary dimensions. By
leveraging Rust's performance and safety features, we aimed to provide a modern
and robust tool for structure-preserving discretization of partial differential
equations formulated in the language of differential forms.

We successfully developed modules for handling the topology of
arbitrary-dimensional simplicial complexes, representing intrinsic Riemannian
geometry via edge lengths inspired by Regge Calculus, performing exterior
algebra computations, and implementing discrete differential forms using
cochains and first-order Whitney basis functions. Building upon this foundation,
we implemented the necessary Galerkin operators for FEEC, specifically targeting
the mixed weak formulation of the Hodge-Laplace equation.

The library's functionality was validated through numerical experiments,
including solving the Hodge-Laplacian eigenvalue problem on a torus, correctly
capturing its topology via harmonic forms, and performing a Method of
Manufactured Solutions convergence study for the source problem. This study
confirmed the expected $O(h^1)$ convergence rate for the $L^2$ error of the
exterior derivative, consistent with theory. However, the study also revealed
an $O(h^1)$ rate for the $L^2$ error of the solution itself, differing
from the commonly expected $O(h^2)$ rate, an observation requiring further
investigation. Furthermore, the analysis constituted a partial validation,
as the error related to the codifferential was not assessed due to time
constraints.

`formoniq` demonstrates the feasibility and benefits of combining the rigorous
mathematical framework of intrinsic FEEC with the modern software engineering
practices enabled by Rust.

== Future Work

Several avenues exist for extending and enhancing the `formoniq` library:

-  Higher-Order FEEC: Extend the implementation to support higher-order
    polynomial basis functions for differential forms. This would enable higher
    accuracy and faster convergence for problems with smooth solutions but must
    be coupled with corresponding higher-order representations of the manifold
    geometry (curved simplices) to avoid introducing non-admissible geometric
    variational crimes. @douglas:feec-book, @hiptmair:whitneyforms, @holst:gvc
- Maxwell's Equations: Apply the framework to solve Maxwell's equations,
    particularly in contexts where the coordinate-free and arbitrary-dimensional
    nature is advantageous, such as electromagnetism on curved spacetimes.
    FEEC provides a natural and structure-preserving discretization for
    these equations. @hiptmair:electromagnetism, @douglas:feec-article,
    @frankel:diffgeo
- Optimization: Profile and optimize core computational routines in Rust,
    potentially exploring alternative sparse matrix libraries or parallelization
    strategies beyond the assembly loop currently handled by Rayon.

== Comparison to Other Implementations

The field of computational differential geometry and structure-preserving
discretizations has seen several software development efforts. To position
`formoniq` within this landscape, we briefly compare it to some notable existing
libraries based on available documentation and publications. `formoniq` did
not directly draw inspiration from these specific implementations but addresses
similar challenges.

*Formoniq (This Thesis)*
- *Focus:* FEEC
- *Dimension:* Arbitrary
- *Mesh:* Simplicial Complexes
- *Geometry:* Intrinsic Regge metric
- *Discretization:* 1st order Whitney forms
- *Language:* Rust
- *Key Features:* Emphasis on FEEC on coordinate-free intrinsic geometry in arbitrary dimensions.
- *Repository:* #weblink("https://github.com/luiswirth/formoniq", [github:luiswirth/formoniq])
#v(0.5cm)

*PyDEC*
- *Focus:* Primarily DEC, some FEEC elements
- *Dimension:* Arbitrary
- *Mesh:* Simplicial and Cubical Complexes
- *Geometry:* Embedded
- *Discretization:* Cochains
- *Language:* Python
- *Key Features:* Mature library for DEC., includes tools for cohomology and
  Hodge decomposition. @pydec
- *Repository:* #weblink("https://github.com/hirani/pydec", [github:hirani/pydec])
#v(0.5cm)

*FEEC++ / simplefem*
- *Focus:* FEEC
- *Dimension:* Hardcoded 1D, 2D and 3D
- *Mesh:* Simplicial Complexes
- *Geometry:* Embedded
- *Discretization:* Arbitrary order polynomial differential forms
- *Language:* C++
- *Key Features:* Focus on arbitrary polynomial order differential forms,
  including Whitney and Sullivan forms. Comes with all necessary linear algebra
  subroutines. @feecpp
- *Repository:* #weblink("https://github.com/martinlicht/simplefem", [github:martinlicht/simplefem])
#v(0.5cm)

*DDF.jl*
- *Focus:* Foundational tools for DEC and FEEC
- *Dimension:* Arbitrary
- *Mesh:* Simplicial Complexes
- *Geometry:* Embedded
- *Discretization:* Higher-order discretizations
- *Language:* Julia
- *Key Features:* Arbitrary dimensions and higher-order methods. Unfinished. @ddfjl
- *Repository:* #weblink("https://github.com/eschnett/DDF.jl", [github:eschnett/DDF.jl])
#v(0.5cm)

*dexterior*
- *Focus:* DEC
- *Dimension:* Arbitrary
- *Mesh:* Simplicial Complexes
- *Geometry:* Embedded
- *Discretization:* Cochain
- *Language:* Rust
- *Key Features:* DEC in Rust, inspired by PyDEC. wgpu visualizer. @dexterior
- *Repository:* #weblink("https://github.com/m0lentum/dexterior", [github:m0lentum/dexterior])

This comparison highlights `formoniq`'s specific niche: providing a
arbitrary-dimensional FEEC implementation fundamentally based on intrinsic
geometry, currently focused on first-order methods. It complements existing
libraries by offering a different language choice (Rust) and a distinct focus on
the coordinate-free geometric perspective inherent in FEEC.
