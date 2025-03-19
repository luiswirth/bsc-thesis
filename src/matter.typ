#import "setup-math.typ": *
#import "layout.typ": *
#import "setup.typ": *

= Software Design & Implementation Choices

In this chapter we want to briefly mention and explain
some software engineering decisions we made.

== Why Rust?

We have chosen Rust as the main programming language for the implementation of
our Finite Element library.
There are various reasons for this choice, some of which we briefly outline
here.

=== Memory Safety + Performance

Rust is a strongly-typed, modern systems programming language that combines
performance on par with `C/C++` with strong memory safety guarantees.
Unlike traditional memory-safe languages that rely on garbage collection,
Rust achieves memory safety through a unique approach inspired by formal
software verification and static analysis techniques, ensuring safety
at compile-time therefore not compromising performance.

The Rust compiler acts as a proof checker, requiring the programmer to provide
sufficient evidence for the safety of their code.
This is accomplished by extending Rust's strong type system with an
ownership and borrowing model that enforces clear semantics regarding
memory responsibility.
This system completely eliminates an entire class of memory-related bugs, making
software significantly more reliable.

Not only does this model guarantee the absence of bugs such as dangling
pointers, use-after-free, memory-aliasing violations, and null-pointer
dereferences, but it also extends to concurrent and parallel programming,
ensuring that data races can never occur.
This "fearless concurrency" feature allows developers to be fully confident that
any parallel code written in Rust that compiles will behave as expected.

=== Expressiveness and Abstraction

Rust is a highly expressive language that enables powerful abstractions
without sacrificing performance, often referred to as zero-cost abstractions.
This allows for a more direct realization of ideas and concepts, making Rust
particularly well-suited for capturing precise mathematical structures and
expressing complex logic in a natural way. Below, we highlight some of the
features that the author finds particularly valuable.

- *Powerful Type System*: Rust features a strong, static type system that
  enables encoding invariants and constraints directly into the type system. This
  ensures contract violations are caught at compile time, significantly reducing
  runtime errors and proving correctness. Through techniques like type-level
  state, one can represent the state of the program using types instead of
  runtime variables, allowing a compile-time style of programming.

- *Traits and Generics*: Rust's trait system facilitates powerful
  polymorphism, enabling code reuse and extensibility without the drawbacks of
  traditional object-oriented inheritance. Unlike classical inheritance-based
  approaches, traits define shared behavior without enforcing a rigid hierarchy.
  Rust’s generics are monomorphized at compile time, ensuring zero-cost
  abstraction while allowing for highly flexible and reusable code structures.
  This approach eliminates the notorious template-related complexities of `C++`,
  as trait bounds explicitly state required behaviors within function and struct
  signatures.

- *Enums, Option, and Result*: Rust provides algebraic data types,
  particularly the sum type `enum`, which acts as a tagged union. This simple
  yet powerful form of polymorphism allows developers to express complex state
  transitions in a type-safe manner. The standard library includes two widely
  used enums: `Option` and `Result`. The `Option` type eliminates null-pointer
  exceptions entirely by enforcing explicit handling of absence. The `Result` type
  enables structured error handling without introducing exceptional control flow,
  leveraging standard function returns for working with errors in a predictable
  way.

- *Expression-Based Language and Pattern Matching*: Unlike many imperative
  languages, Rust is expression-based, meaning that most constructs return
  values rather than merely executing statements. For example, an `if` expression
  evaluates to the value of its selected branch, eliminating redundant variable
  assignments. Combined with Rust’s powerful pattern matching system, which allows
  destructuring of composite types like enums, this leads to concise and readable
  code while enabling functional-style composition.

- *Functional Programming and Iterators*: Rust embraces functional programming
  principles such as higher-order functions, closures (lambdas), and iterators.
  The iterator pattern allows for efficient, lazy evaluation of collections,
  reducing unnecessary memory allocations and improving performance. Functional
  constructs such as `map`, `filter`, and `fold` are available as standard methods
  on iterators, promoting declarative and expressive coding styles.


Together, these features make Rust an expressive language, providing developers
with the tools to write concise, maintainable, and high-performance software. By
combining modern programming paradigms with low-level control, Rust ensures both
safety and efficiency, making it an excellent choice for scientific computing
and systems programming.



=== Infrastructure and Tooling

Beyond its language features, Rust also stands out due to its exceptional
infrastructure and tooling ecosystem, which greatly enhances the development
workflow.

A key advantage is the official nature of all tooling, which reduces
fragmentation and prevents competing tools from creating confusion over choices,
in contrast to the `C++` ecosystem. This consistency fosters a more streamlined
and productive development experience.

- *Cargo* is Rust's offical package manager and build system, which is one of
  the most impressive pieces of tooling. It eliminates the need for traditional
  build tools like Makefiles and CMake, which are often complex and difficult
  to maintain—not to mention the dozens of other build systems for `C++`.
  Cargo simplifies dependency management through its seamless integration with
  `crates.io`, Rust’s central package repository. Developers can effortlessly
  include third-party libraries by specifying them in the `Cargo.toml` file, with
  Cargo automatically handling downloading, compiling, and dependency resolution
  while enforcing semantic versioning. Publishing a crate is equally simple via
  `cargo publish`, which we have also used to distribute the libraries developed
  for this thesis.
- *Rust Analyzer* is Rust's official Language Server Protocol (LSP)
  implementation, providing advanced IDE support, including real-time feedback,
  type hints, and code completion. This significantly enhances the ergonomics of
  Rust development.
- *Clippy* is Rust's official linter, offering valuable suggestions for
  improving code quality, adhering to best practices, and catching common
  mistakes. Our codebase does not produce a single warning or lint, passing all
  default checks for code quality.
- *Rustdoc* is Rust's official documentation tool, allowing developers to
  write inline documentation using Markdown, seamlessly integrated with code
  comments. This documentation can be compiled into a browsable website with
  `cargo doc` and is automatically published to `docs.rs` when a crate is uploaded
  to `crates.io`. The documentation for our libraries is also available there.
- *Rusttest* is the testing functionality built into Cargo for running unit
  and integration tests. Unit tests can be placed alongside the normal source code
  with a simple `#[test]` attribute, and the `cargo test` command will execute
  all test functions, verifying correctness. This command also ensures that all
  code snippets in the documentation are compiled and checked for runtime errors,
  keeping documentation up-to-date without requiring external test frameworks like
  Google Test.
- *Rustfmt* standardizes code formatting, eliminating debates about code style
  and ensuring consistency across projects. Our codebase fully adheres to Rustfmt's
  formatting guidelines. For conciseness however we will be breaking
  the formatting style when putting code inline into this document.

Together, Rust’s comprehensive tooling ecosystem ensures a smooth, efficient,
and reliable development experience, reinforcing its position as a robust choice
for scientific computing and large-scale software development.


There are many more good reasons to choose Rust, such as it's great ecosystem
of libraries, which are some of the most impressive libraries the author has ever seen.


== External libraries

We want to quickly discuss here the major external libraries,
we will be using in our project.

=== nalgebra (linear algebra)

For implementing numerical algorithms linear algebra libraries are invaluable.
`C++` has set a high standard with `Eigen` as it's major linear algebra library.
Rust offers a very direct euivalent called `nalgebra`, which just as Eigen
relies heavily on generics to represent both statically and dynamically know
matrix dimensions.
All basic matrix and vector operations are available.
We will be using nalgebra all over the place, pretty much always we have to deal
with numerical values.

Sparse matrices will also be relevant in our library.
For this we will be using `nalgebra-sparse`.


=== PETSc & SLEPc (solvers)

Unfortunatly the rust sparse linear algebra ecosystem is rather immature.
Only very few sparse solvers are available in Rust.
For this reason we will be using one of the big `C/C++` sparse matrix libraries
called PETSc. We will be using direct solvers.

For eigensolvers we will be using SLEPc, which builds on top of PETSc.

== General software architecture

We aim to model mathematical concepts as faithfully as possible, ensuring both
mathematical accuracy and code clarity.
This renders the code mathematically expressive and self-documenting for those
with the necessary background.
While we do not shy away from mathematical complexity or oversimplify for
the sake of accessibility, we recognize the importance of good API design and
HPC principles. Our goal is to strike a balance between mathematical rigor,
usability, and performance.

=== Modularity

As is the nature with most FEM libraries, they are rather big pieces
of software. They consists of many parts with different responsibilities.
So of which are useable standalone, for instance the mesh could also be used
for a different application. For this reason we split up our FEM library
into multiple libraries than built on top of each other.

We rely on a Cargo workspace to organize the various parts of our library ecosystem.

We have the following crates:
- multi-index
- manifold
- exterior
- whitney
- formoniq

All of which have been published to `crates.io`.

===  Type safety

The implementation has a big emphasis on providing safety through the introduction
of many types that uphold gurantees regarding the contained data.
Constructed instances of types should always be valid.

=== Performance considerations

All datastructures are written with performance in mind.

We are also always focused on a memory-economic representation of information.





= Arbitrary Dimensions & Multi-Index Combinatorics

== Programming in Arbitrary Dimensions

Supporting arbitrary dimensions requires a special style of programming.
One way in which this manifests is how we work with for-loops.
Usually if one would work in fixed 3D, then one would iterate over
3D arrays using 3 nested for-loops. But in arbitrary dimensions one would need
a number of for loops that is determined by a variable at run-time. This is not possible.
So we cannot rely on nested for-loops to iterate over $n$-dimensional data.
Instead we will rely on a arbitrary dimensional multi-index implementation.

A multi-index is a dimensional generalization of a index.
It looks like this
$ I = (i_0,i_1,dots,i_n) $
It's a tuple of single indices, grouping them together.
We can write a for loop over all variants of valid multi indices and
then use the multi-index to index into our multi-dimensional data structure.


This motivates the creation of a small library that supports multi-indices.
There are two main flavors of multi-indices that concern us.

== Cartesian Multi-Index

A cartesian multi-index is pretty simple. It's an element of a cartesian product
of various single-index sets.

- Cartesian product style for-loops
- Reference simplex style for-loops

== Anti-symmetric multi-indices

Anti-Symmetric Mutli-Indices play a huge role in the combinatorics
of simplicies and exterior algebras.

These are multi-indicies with are ordered sets (no duplicates)
of indices. They are governed by various rules surrounding
permutations and sign.

=== Sign

Swapping any two indicies results in a sign change of the multi-index.
We represent the sign of such a multi-index using a `Sign` enum.

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Sign {
  #[default]
  Pos = 1,
  Neg = -1,
}
```

It comes with some basic convenience functions
```rust
pub fn from_bool(b: bool) -> Self {
  match b {
    true => Self::Pos,
    false => Self::Neg,
  }
}
pub fn from_f64(f: f64) -> Option<Self> {
  if f == 0.0 { return None; }
  Some(Self::from_bool(f > 0.0))
}

/// useful for permutation parity
pub fn from_parity(n: usize) -> Self {
  match n % 2 {
    0 => Self::Pos,
    1 => Self::Neg,
    _ => unreachable!(),
  }
}

pub fn other(self) -> Self {
  match self {
    Sign::Pos => Sign::Neg,
    Sign::Neg => Sign::Pos,
  }
}
pub fn flip(&mut self) {
  *self = self.other()
}
```

We implement some basic arithmetic for this struct, such
as negation and multiplication.
```rust
impl std::ops::Neg for Sign {
  type Output = Self;
  fn neg(self) -> Self::Output {
    match self {
      Self::Pos => Self::Neg,
      Self::Neg => Self::Pos,
    }
  }
}
impl std::ops::Mul for Sign {
  type Output = Self;
  fn mul(self, other: Self) -> Self::Output {
    match self == other {
      true => Self::Pos,
      false => Self::Neg,
    }
  }
}
```

The main use of this `Sign` type is representing the sign of a permutation.
For this we implement a basic bubble sort that counts the number of swaps,
based on which we can compute the sign of the sorted permutation relative
to the original permutation.
```rust
/// Returns the sorted permutation of `a` and the sign of the permutation.
pub fn sort_signed<T: Ord>(a: &mut [T]) -> Sign {
  Sign::from_parity(sort_count_swaps(a))
}

/// Returns the sorted permutation of `a` and the number of swaps.
pub fn sort_count_swaps<T: Ord>(a: &mut [T]) -> usize {
  let mut nswaps = 0;

  let mut n = a.len();
  if n > 0 {
    let mut swapped = true;
    while swapped {
      swapped = false;
      for i in 1..n {
        if a[i - 1] > a[i] {
          a.swap(i - 1, i);
          swapped = true;
          nswaps += 1;
        }
      }
      n -= 1;
    }
  }
  nswaps
}
```

=== IndexSet

This is our antisym multi-index struct.
```rust
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct MultiIndex<O: IndexKind> {
  indices: Vec<usize>,
  _order: PhantomData<O>,
}
```

Through the generic type parameter `O: IndexKind` we introduce
a marker type that represents the kind of index we are dealing with.
There are two kinds: `ArbitraryList`, with no constraints on
the indices and `IncreasingSet`. These are represent using marker types
(zero-sized) and a marker trait (no associated methods).
```rust
pub trait IndexKind: Debug + Default + Clone + Copy + PartialEq + Eq + Hash {}

/// Arbitrary order of elements. May contain duplicates.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArbitraryList;
impl IndexKind for ArbitraryList {}

/// Strictly increasing elements! No duplicates allowed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IncreasingSet;
impl IndexKind for IncreasingSet {}
```

The `IncreasingSet` kind represents the antisymmetric index.

This crate is rather convoluted due to the type state.
The design is not optimal, even though much time was spent
refactoring this module. It has been a difficult design challenge.
We will keep it as is.

Many operations are possible on the two kinds of multi indices.
Such as computing all permutations or all subsets of indices.
Or computing the "boundary" of a IncreasingSet. Which
is the same operation as for a simplex.




= Topology & Geometry of Simplicial Riemannian Manifolds

In this chapter we will develop various data structures and algorithms to
represent and work with our Finite Element mesh.
It will store the topological and geometrical properties of our arbitrary
dimensional discrete PDE domain.
A simplicial complex will be used to represent the topology (incidence and
adjacency) of the mesh and double as the container for all the mesh entities,
which are all simplicies. It will also provide unique identification through a
global numbering, and iteration of all these entities.
For the geometry, all edge lengths of the mesh will be stored to compute the
piecewise-flat (over the cells) Riemannian metric, known as the Regge metric.
We also support the optional storage of global vertex coordinates, if an
embedding were to be known.

== Coordinate Simplicies

Finite Element Methods benefit from their ability to work on unstructured meshes.
So instead of subdividing a domain into a regular grid, FEM works on potentially
highly non-uniform meshes.
The simplest type of mesh that works for such non-uniform meshes are simplicial
meshes.
In 2D these are triangular meshes as known from computer graphics.
A 3D simplicial mesh is made up from tetrahedrons.
These building blocks need to be generalized to our arbitrary dimensional
implementation.

We begin the exposition of the mesh topic with a coordinate-based object
that relies on an embedding in an ambient space.
Later on we will shed the coordinates and only rely on intrinsic geometry.
For didactics it's however useful to start with coordinates.

#v(0.5cm)

The generalization of 2D triangles and 3D tetrahedra to $n$D
is called a $n$-simplex.
There is a type of simplex for every dimension.
These are first 4 kinds:
- A 0-simplex is a point,
- a 1-simplex is a line segment,
- a 2-simplex is a triangle, and
- a 3-simplex is a tetrahedron.
The idea here is that an $n$-simplex is the polytope with the fewest vertices
that spans a $n$-dimensional affine subspace of $RR^N$. It's the simplest $n$-polytope there is.
A $n$-simplex $sigma$ always has $n+1$ vertices $avec(v)_0,dots,avec(v)_n in RR^N$ in a possibly higher dimensional
space $RR^N, N >= n$ and the simplex is the patch of space
bounded by the convex hull of the vertices.
$
  Delta(RR^n) in.rev sigma =
  "convex" {avec(v)_0,dots,avec(v)_n} =
  {
    sum_(i=0)^n lambda^i avec(v)_i
    mid(|)
    quad lambda^i in [0,1],
    quad sum_(i=0)^n lambda^i = 1
  }
$ <def-simplex>

We call such an object a *coordinate simplex*, since it depends on global coordinates of
the vertices and lives in a possible higher-dimensinal ambient space $RR^N$. It therefore
relies on an embedding.
This object is uniquely determined by the coordinates $avec(v)_i$ of each vertex,
inspiring a simple computational representation based on a struct, that
stores the coordinates of each vertex in the columns of a matrix.
```rust
pub struct SimplexCoords {
  pub vertices: na::DMatrix<f64>,
}
impl SimplexCoords {
  pub fn nvertices(&self) -> usize { self.vertices.ncols() }
  pub fn coord(&self, ivertex: usize) -> CoordRef { self.vertices.column(ivertex) }
}
```

We implement two methods to compute both the intrinsic dimension $n$, which
is one less the number of vertices and the ambient dimension $N$ of the global coordinates.
A special and particulary simple case is when intrinsic dimension and ambient dimension
match up $n=N$.
```rust
pub type Dim = usize;
pub fn dim_intrinsic(&self) -> Dim { self.nvertices() - 1 }
pub fn dim_embedded(&self) -> Dim { self.vertices.nrows() }
pub fn is_same_dim(&self) -> bool { self.dim_intrinsic() == self.dim_embedded() }
```

#v(0.5cm)
=== Barycentric Coordinates

The coefficents $avec(lambda) = [lambda^i]_(i=0)^n$ in @def-simplex are called
*barycentric coordinates*.
They appear inside the convex combination / weighted average $sum_(i=0)^n lambda^i avec(v)_i$
as weights $lambda^i in [0,1]$ with condition $sum_(i=0)^n lambda^i = 1$ in front of
each cartesian vertex coordinate $avec(v)_i in RR^N$.
They constitute an intrinsic local coordiante representation with respect to the simplex $sigma in RR^N$,
which is independent from the embedding in $RR^N$ and only relies on the convex combination
of vertices. \
The coordinate transformation $psi: avec(lambda) |-> avec(x)$ from intrinsic
barycentric $avec(lambda) in RR^n$ to ambient cartesian $avec(x) in RR^N$ coordinate is
given by
$
  avec(x) = psi (avec(lambda))
  = sum_(i=0)^n lambda^i avec(v)_i
$

We can easily implement this as
```rust
pub fn bary2global<'a>(&self, bary: impl Into<BaryCoordRef<'a>>) -> EmbeddingCoord {
  let bary = bary.into();
  self
    .vertices
    .coord_iter()
    .zip(bary.iter())
    .map(|(vi, &baryi)| baryi * vi)
    .sum()
}
```
The barycentric coordinate representation can be extended beyond the bounds of the simplex to
the whole affine subspace.
The condition $sum_(i=0)^n lambda^i = 1$ must still hold but only points
$avec(x) in sigma$ inside the simplex have $lambda^i in [0,1]$.
```rust
pub fn is_coord_inside(&self, global: CoordRef) -> bool {
  let bary = self.global2bary(global);
  is_bary_inside(&bary)
}
pub fn is_bary_inside<'a>(bary: impl Into<CoordRef<'a>>) -> bool {
  bary.into().iter().all(|&b| (0.0..=1.0).contains(&b))
}
```
Outside the simplex $avec(x) in.not sigma$, $lambda^i$ may be greater than one
or even negative. \
The barycenter $avec(m) = 1/(n+1) sum_(i=0)^n avec(v)_i$ always has the special
barycentric coordinates $psi(avec(m)) = [1/n]^(n+1)$.
```rust
pub fn barycenter(&self) -> Coord {
  let mut barycenter = na::DVector::zeros(self.dim_embedded());
  self.vertices.coord_iter().for_each(|v| barycenter += v);
  barycenter /= self.nvertices() as f64;
  barycenter
}
```

This coordinate system treats all vertices on equal footing and therefore
there is a weight for each vertex.
But as a consequence of this, there is some redundancy in this coordinate representation,
making it not a proper coordinate system,
since we have $n+1$ vertices for an only $n$-dimensional affine subspace.

We can instead single out a special vertex to remove this redundancy. We choose
for this vertex $avec(v)_0$ and call it the *base vertex*.
```rust
pub fn base_vertex(&self) -> CoordRef { self.coord(0) }
```

We can then leave off the redundant $lambda^0 = 1 - sum_(i=1)^n lambda^i$
corresponding to $avec(v)_0$.
Then the reduced barycentric coordiantes $avec(lambda)^- = [lambda^i]_(i=1)^n$
constitutes a proper coordinate system without any redundant information.
This coordinate system also works for the whole affine subspace, but now
the coordinates are completly free and there is a bijection between
the affine subspace and the whole of $RR^n$. There is a unique representation
for both ways.

#v(0.5cm)
=== Spanning Vectors

If we consider the edges eminating from the base vertex, we get the
*spanning vectors* $[avec(e)_i]_(i=1)^n$ with $avec(e)_i = avec(v)_i - avec(v)_0 in RR^N$.
We can define a matrix $amat(E) in RR^(N times n)$ that has these
spanning vectors as columns.
$
  amat(E) = 
  mat(
      |,  , |;
      avec(e)_1,dots.c,avec(e)_n;
      |,  , |;
    )
$

We implement a function to compute this matrix.
```rust
pub fn spanning_vectors(&self) -> na::DMatrix<f64> {
  let mut mat = na::DMatrix::zeros(self.dim_embedded(), self.dim_intrinsic());
  let v0 = self.base_vertex();
  for (i, vi) in self.vertices.coord_iter().skip(1).enumerate() {
    let v0i = vi - v0;
    mat.set_column(i, &v0i);
  }
  mat
}
```

These spanning vectors are very natural to the reduced barycentric coordinate system,
since we can rewrite the coordinate transformation $psi$ as
$
  avec(x)
  = sum_(i=0)^n lambda^i avec(v)_i
  = avec(v)_0 + sum_(i=1)^n lambda^i (avec(v)_i - avec(v)_0)
  = avec(v)_0 + amat(E) avec(lambda)
$
This makes it very apparant that this transformation is an affine map, between
the reduced barycentric coordinates $avec(lambda)^-$, which we also just call
local coordinates and the cartesian coordinates of the affine subspace
spanned up by the spanning vectors positioned at the base vertex.

We can implement some transformation functions.
```rust
pub fn linear_transform(&self) -> na::DMatrix<f64> { self.spanning_vectors() }
pub fn affine_transform(&self) -> AffineTransform {
  let translation = self.base_vertex().into_owned();
  let linear = self.linear_transform();
  AffineTransform::new(translation, linear)
}
pub fn local2global<'a>(&self, local: impl Into<LocalCoordRef<'a>>) -> EmbeddingCoord {
  let local = local.into();
  self.affine_transform().apply_forward(local)
}
```

Where we represent an affine transform as
```rust
pub struct AffineTransform {
  pub translation: na::DVector<f64>,
  pub linear: na::DMatrix<f64>,
}
pub fn apply_forward(&self, coord: na::DVectorView<f64>) -> na::DVector<f64> {
  &self.linear * coord + &self.translation
}
```

Reversing the transformation, is more subtle, as due to floating point inaccuricies,
we almost never exactly lie in the affine subspace. Therefore we rely on the
Moore-Penrose pseudo-inverse, computed via SVD, to get a least-square solution.
$
  avec(lambda)^- = phi(avec(x))
  = amat(E)^dagger (avec(x) - avec(v)_0)
$

```rust
pub fn global2local<'a>(&self, global: impl Into<EmbeddingCoordRef<'a>>) -> LocalCoord {
  let global = global.into();
  self.affine_transform().apply_backward(global)
}

pub fn apply_backward(&self, coord: na::DVectorView<f64>) -> na::DVector<f64> {
  if self.linear.is_empty() { return na::DVector::default(); }
  self
    .linear
    .clone()
    .svd(true, true)
    .solve(&(coord - &self.translation), 1e-12)
    .unwrap()
}

pub fn pseudo_inverse(&self) -> Self {
  let linear = self.linear.clone().pseudo_inverse(1e-12).unwrap();
  let translation = &linear * &self.translation;
  Self { translation, linear, }
}

```

By computing derivatives of the affine parametrization of the simplex, we
find that the spanning vectors are a very natural frame/basis for the tangent space
$T_p sigma$ of the simplex $sigma$ at each point $p in sigma$.
The jacobian of the affine map is exactly the linear map represented by $amat(E)$.
$
  (diff avec(x))/(diff lambda^i)
  = avec(e)_i
  quad quad
  (diff avec(x))/(diff avec(lambda))
  = amat(E)
$

Furthermore we can also compute the total differential of the barycentric coordinate
functions based on the pseudo-inverse.
This constitutes a natural basis for cotangent space $T^*_p sigma$ of the simplex $sigma$
at each point $p in sigma$. It is in fact the dual basis.
$
  (diff avec(lambda))/(diff avec(x))
  = amat(E)^dagger
  quad quad
  avec(epsilon)^i = dif lambda^i = (diff lambda^i)/(diff avec(x))
  = (amat(E)^dagger)_(i,:)
  quad quad
  dif lambda^0 = -sum_(i=1)^n dif lambda^i
  quad quad
  dif lambda^i (diff/(diff lambda^j)) = delta^i_j
$
```rust
pub fn difbarys(&self) -> na::DMatrix<f64> {
  let difs = self.linear_transform().pseudo_inverse(1e-12).unwrap();
  let mut grads = difs.insert_row(0, 0.0);
  grads.set_row(0, &-grads.row_sum());
  grads
}
```

TODO: consider moving to Riemannian geometry section...
Using the euclidean geometry of the ambient space, we can compute the metric tensor
expressed in this tangent space basis to do intrinsic Riemannian geometry.
$
  amat(G) = amat(E)^transp amat(E)
$

```rust
pub fn metric_tensor(&self) -> RiemannianMetric {
  let metric = self.spanning_vectors().gramian();
  RiemannianMetric::new(metric)
}
```

Furthermore these spanning vectors define a parallelepiped.
This parallelipied can be used to compute the volume of the simplex, as a
fraction $(n!)^(-1)$ of the volume of the parallelepiped, which is computed as
the determinant of the spanning vectors in the case $n=N$ and otherwise using
the square root of the gramian determinant
$sqrt(det(amat(E)^transp amat(E)))$.
```rust
pub fn det(&self) -> f64 {
  let det = if self.is_same_dim() {
    self.spanning_vectors().determinant()
  } else {
    self.spanning_vectors().gram_det_sqrt()
  };
  ref_vol(self.dim_intrinsic()) * det
}
pub fn vol(&self) -> f64 { self.det().abs() }
pub fn ref_vol(dim: Dim) -> f64 { (factorial(dim) as f64).recip() }

```
Based on this we can also get the global orientation of the simplex via
the sign of the determinant.
```rust
pub fn orientation(&self) -> Sign {
  Sign::from_f64(self.det()).unwrap()
}
```
As a consequence swapping to vertices in the simplex, will swap the orientation of the simplex,
by the properties of the determinant.

Every simplex has two orientations positive and negative, just like the determinant
always has only two signs.
#table(
  columns: 4,
  stroke: fgcolor,
  table.header(table.cell(colspan: 4, align: center)[*Simplex Orientation*]),
  $n$, [Simplex], [Positive], [Negative],
  $1$, [Line Segment], [left-to-right], [right-to-left],
  $2$, [Triangle], [counterclockwise], [clockwise],
  $3$, [Tetrahedron], [right-handed], [left-handed],
)

#v(0.5cm)
=== Reference Simplex

There is a special simplex, called the *reference simplex*, which has exactly the
barycentric coordinates also as global coordinates.
$
  sigma_"ref"^n = {(lambda_1,dots,lambda_n) in RR^n mid(|) lambda_i in [0,1], quad sum_(i=1)^n lambda_i <= 1 }
$
For each dimension $n in NN$ there is exactly one reference simplex $sigma_"ref"^n$,
which has coinciding intrinsic and ambient dimension $N=n$.
For this simplex the edge vectors are exactly the euclidean standard basis vectors
$avec(e)_i = nvec(e)_i$ with $(nvec(e)_i)_j = delta_i^j$.
Is has volume $det(sigma) = (n!)^(-1)$.

They give rise to an euclidean orthonormal tangent space basis. $amat(E) = amat(I)_n$
Which manifests as a metric tensor that is equal to the identity matrix
$amat(G) = amat(I)_n^transp amat(I)_n = amat(I)_n$.

The base vertex is the origin $v_0 = avec(0) in RR^n$ and the other vertices
are $avec(v)_i = avec(0) + nvec(e)_i = nvec(e)_i$.
The parametrization of the reference simplex is the identity map.
$phi: avec(lambda)^- |-> avec(0) + amat(I) avec(lambda)^-$

Every (real) simplex is the image of the reference simplex under the affine parametrization
map.
$
  sigma = phi(sigma_"ref"^n)
$
The barycentric coordinate functions are since they are intrinsic affine-invariant.

== Simplicial Topology

After studying coordinate simplicies, the reader has hopefully developed
some intuitive understanding of simplicies. We will now shed the coordinates
and represent simplicies in a more abstract way, by just considering
them as a list of vertex indicies, without any vertex coordinates.
This makes our simplicies just combinatorial objects and these combinatorics
will be heart of our mesh datastructure.
```rust
pub type VertexIdx = usize;
pub struct Simplex {
  vertices: Vec<VertexIdx>,
}
```

The topology of the mesh will then be captured by the fact that some simplicies
share the same vertices, meaning that they are somehow topologically connected,
by being either adjacent or indicent.


=== Orientation

We have seen that with coordinate simplicies due to the properties of the
determiant the swapping of two vertices changes the orientation.
This property is still present in our abstract simplicies, where
the orientation is dependent on the ordering of the vertices.
For any simplex there are exactly two orientations, which
are opposites of each other.
All permutations can be divided into two equivalence classes.
Given a reference ordering of the vertices of a simplex,
we can call these even and odd permutations, relative to this reference.
We call simplicies with even permutations, positively oriented and
simplicies with odd permutations negatively oriented.

Our convention will be to call simplicies with increasing
indices positively oriented.

=== Skeleton

It's our goal to have a discrete variant of a $n$-manifold.
We can do this by combining multiple $n$-simplicies.
These $n$-simplicies are supposed to approximate the topology of the manifold.
We call such a set of simplicies a skeleton.
We will be using $n$-skeletons to represent the top-level topology of our domain
$n$-manifold.

```rust
pub struct Skeleton {
  simplicies: Vec<Simplex>,
}
```

=== Triangulation of Manifold

The procedure of producing a skeleton from a manifold is called
triangulation.
The name stems from the 2D case, where a surface (2D topology)
is approximated by a collection of triangles.
We aren't too concerned with triangulating manifolds in this thesis
as we mostly assume that we are already given such a triangulation.

Nontheless formoniq features a triangulation algorithm for arbitrary dimensional
tensor-product domains. These domains are $n$-dimensional cartesian products $[0,1]^n$
of the unit interval $[0,1]$. The simplicial skeleton will be computed based on
a cartesian grid that subdivides the domain into $l^n$ many $n$-cubes, which are
generalizations of squares and cubes. Here $l$ is the number of subdivisions per axis.
To obtain a simplicial skeleton, we need to split each $n$-cube into non-overlapping simplicies
that make us it's volume. In 2D it's very natural to split a square into two triangles
of equal volume. This can be generalized to higher dimensions. The trivial
triangulation of a $n$-cube into $n!$ simplicies is based on the $n!$ many permutations
of the $n$ coordinate axes.

The $n$-cube has $2^n$ vertices, which can all be identified using a multi-indicies
$
  V = {(i_1,dots,i_n) mid(|) i_j in {0,1}}
$
All $n!$ simplicies will be based on this vertex base set. To generate the list
of vertices of the simplex, we start at the origin vertex $v_0 = 0 = (0)^n$.
From there we walk along on axis directions from vertex to vertex.
For this we consider all $n!$ permutations of the basis directions $avec(e)_1,dots,avec(e)_n$.
A permutation $sigma$ tells in which axis direction we need to walk next.
This gives us the vertices $v_0,dots,v_n$ that forms a simplex.
$
  v_k = v_0 + sum_(i=1)^k avec(e)_sigma(i)
$

The algorithm in Rust looks like:
```rust
pub fn compute_cell_skeleton(&self) -> Skeleton {
  let nboxes = self.ncells();
  let nboxes_axis = self.ncells_axis();

  let dim = self.dim();
  let nsimplicies = factorial(dim) * nboxes;
  let mut simplicies: Vec<SortedSimplex> = Vec::with_capacity(nsimplicies);

  // iterate through all boxes that make up the mesh
  for ibox in 0..nboxes {
    let cube_icart = linear_index2cartesian_index(ibox, nboxes_axis, self.dim());

    let vertex_icart_origin = cube_icart;
    let ivertex_origin =
      cartesian_index2linear_index(vertex_icart_origin.clone(), self.nvertices_axis());

    let basisdirs = IndexSet::increasing(dim);

    // Construct all $d!$ simplexes that make up the current box.
    // Each permutation of the basis directions (dimensions) gives rise to one simplex.
    let cube_simplicies = basisdirs.permutations().map(|basisdirs| {
      // Construct simplex by adding all shifted vertices.
      let mut simplex = vec![ivertex_origin];

      // Add every shift (according to permutation) to vertex iteratively.
      // Every shift step gives us one vertex.
      let mut vertex_icart = vertex_icart_origin.clone();
      for basisdir in basisdirs.set.iter() {
        vertex_icart[basisdir] += 1;

        let ivertex = cartesian_index2linear_index(vertex_icart.clone(), self.nvertices_axis());
        simplex.push(ivertex);
      }

      Simplex::from(simplex).assume_sorted()
    });

    simplicies.extend(cube_simplicies);
  }

  Skeleton::new(simplicies)
}
```

We can note here, that the computational complexity of this algorithm, grows extremly fast
in the dimension $n$.
We have a factorial scaling $cal(O)(n!)$ (worse than exponential scaling $cal(O)(e^n)$)
for splitting the cube into simplicies. Given $l$ subdivisions per dimensions, we have
$l^n$ cubes. So the overall computational complexity is dominated by $cal(O)(l^n n!)$,
a terrible result, due to the curse of dimensionality.
The memory usage is dictated by the same scaling law.


=== The Simplicial Complex

A simplicial skeleton contains all information necessary to define the topology
of the discrete manifold, but it's lacking information on all lower-dimensional
simplicies contained in it. These are very relevant to the algorithms of FEEC.

Inside of a simplex we can identify subsimplicies.
For example a triangle $[0,1,2]$ contains the edges $[0,1]$, $[1,2]$ and $[3, 2]$.

Every $n$-simplex $sigma$ contains $k$-subsimplicies $Delta_k (sigma)$ for all $k <= n$.
For every subset of vertices there exists a subsimplex.

Some useful terminology is
- The $0$-simplicies are called vertices.
- The $1$-simplicies are called edges.
- The $2$-simplicies are called faces.
- The $3$-simplicies are called tets.
- The $(n-1)$-simplicies are called facets.
- The $n$-simplicies are called cells.

The skeleton only stores the top-level simplicies $Delta_n (mesh)$, but our FEM library
also needs to reference the lower-level simplicies $Delta_k (mesh)$, since these are also
also mesh entities on which the DOFs of our FE space live.

For this reason we need another concept that extends our skeleton.
This concept is called a simplicial complex. It contains all
subsimplicies of all dimensions of a $n$-dimensional skeleton.
It will be the main topological data structure that we will, pass as argument
into all FEEC algorithm routines.

```rust
pub struct TopologyComplex {
  skeletons: Vec<ComplexSkeleton>,
}
pub type ComplexSkeleton = IndexMap<SortedSimplex, SimplexData>;
```

From a mathematical perspective we have an abstract (no geometry) simplicial complex
that is a families of abstract simplicies, which are all just ordered finite sets of
the vertex indices, that are closed under the subset relation.
This makes the simplical complex are purely combinatorial data strucuture.

In general a simplicial complex need not have a manifold topology, since it can
represent many more topological spaces beyond manifolds.
For our PDE framework domains needs to be manifold, so our simplical complex
data structure will also only support this.
For a simplicial complex to manifold, the neighborhood of each vertex (i.e. the
set of simplices that contain that point as a vertex) needs to be homeomorphic
to a $n$-ball.
This manifests for instance in the fact that there are at most 2 facets for each cell.
Another property is that the simplicial complex needs to be pure, meaning
every $k$-subsimplex is contained in some cell.

== Simplicial Homology

Simplicial complexes are objects that are studied in algebraic topology.

One of the major theories relevant to FEEC is homology theory,
which is rooted in algebraic topology.
Informally speaking homology is all about identifying holes
of a topological space, which in our case is our PDE domain manifold
represented as a simplicial complex.

The presence of holes has influence on the existance and uniqueness of
PDE solutions and therefore is relevant and needs to be studied.


== Atlas and Differential Structure

Until now we only studied the manifold as a topological space.
We now start studying additional structure.
We start with coordinate charts and the atlas.


These reference charts define the local coordinate systems we will be working
in on each cell.

The collection of all local coordinate charts for each cell
covers the whole manifold, giving us an atlas.

This is a very useful atlas, but it is not helpful to establish a differential
structure on the manifold, as the overlapping regions are only $(n-1)$-dimensional
facets and therefore not facilitate a proper $n$-dimensional open neighboorhood.
Therefore we cannot investigate the smoothness of the transition maps.

Without constructing an another atlas to establish this, we just state here
that this is a piecewise $C^oo$-smooth manifold. Where the cells are the pieces.

=== Tangent space

Now that we've established the existance of a differentiable structure, we
can now talk about the tangent bundle and the tangent spaces.

The tangent space $T_p M$ is the linear space of vectors $v in T_p M$,
that are tangent to the manifold $M$ in some point $p$. Since a manifold
is curved in general the space of tangent vectors changes from point to point.

Since our manifold is piecewise-flat the tangent space remains the same
over each cell. So instead of a tangent space $T_p M$ at each point $p$,
we have a tangent space $T_sigma M$ at each cell $sigma$.

=== Cotangent space

There is a dual space to the tangent space, called the cotangent space.
The element of which are called covectors. Covectors are basically just
linear forms on the tangent space.
$
  T^*_p M = { alpha mid(|) alpha: T_p M -> RR }
$

You can think of covectors as measuring (in a sense different from measure theory)
tangent vectors.

The cotangent space also demands a basis, for which there is once again a very natural choice.
We use the dual basis defined by
$
  dif x^i (diff/(diff x^j)) = delta^i_j
$

So for our edge basis, we only need to specify how each edge gets measure and by linearity (only scaling)
this tells us how each vector is measured.

Covectors are *covariant* tensors.
In exterior algebra we will extend them to multiforms that are rank $k$ fully
covariant tensors. We will then consider fields of these multiforms that
vary over the manifold, which we will then call differential forms.

== Simplicial Manifold Geometry

Our PDE domain is a curved manifold, it's curvature is not represented in the topology,
but in the geometry. Curvature is a infintesimal notion, but for our discrete mesh
we will need to discretize it.
The approximation we will make here, is that the mesh is *piecewise-flat* over the cells
of the mesh, meaning inside of a cell we don't observe any change of curvature.
So we will not be using any higher-order mesh elements, like quadratic or cubic elements.
This approximation will be fine for our implementation, as we restrict ourselves to
1st order finite elements, where the basis functions are only linear.
For linear finite elements, the piecewise-flat approximation is an *admissable geometric variational crime*,
meaning it doesn't influence the order of convergence and therefore doesn't have a
too bad influence on the solution.

There are two ways of doing geometry.
- *Extrinsic Eucliean Geometry*, and
- *Intrinsic Riemannian Geometry*
Formoniq supports both representation of the mesh. This might come as a suprise, since
initially we stated that we will be solely relying on coordinate-free meshes.
What we mean by this is that the Finite Element algorithms will only rely on the
intrinisic geometry. This is still true. But in order to obtain the intrinsic description
it is helpful to first construct a coordinate representation and then compute
the intrinsic geometry and to forget about the coordinates then.
We will see how this works in formoniq.

=== Extrinsic Euclidean Geometry

Extrinsic geometry, is the typical euclidean geometry
everybody knows, where the manifold is embedded in an ambient space, for example the
unit sphere inside of $RR^3$. The euclidean ambient space allows one to measure
angles, lengths and volumes by using the standard euclidean inner product (dot product).
The embedding gives the manifold global coordinates, which identify the points of the manifold
using a position $xv in RR^N$. For our piecewise-flat mesh, the necessary geometrical information
would only be the coordinates of all vertices.
This is the usual way mesh-based PDE solvers work, by relying on these global coordinates.
These vertex coordinates can be easily stored in a single struct that lays out
the vertex coordinates as the columns of a matrix.
```rust
pub struct MeshVertexCoords(na::DMatrix<f64>);
```
Specifying the $k+1$ vertex coordinates $v_i in RR^N$ of a $k$-simplex defines a $k$-dimensional affine subspace
of the ambient euclidean space $RR^N$. This is because $k+1$ points always define a $k$-dimensional plane
uniquely. This makes the geometry piecewise-flat.


=== Intrinsic Riemannian Geometry

In contrast to this we have intrinsic Riemannian geometry that souly relies
on a structure over the manifold called a *Riemannian metric* $g$.
It is a continuous function over the whole manifold, which at each point $p$
gives us an inner product $g_p: T_p M times T_p M -> RR^+$ on the tangent space $T_p M$ at this point.
It is the analog to the euclidean inner product in euclidean geometry. Since the manifold
is curved the inner product changes from point to point, reflecting the changing geometry.

The Riemannian metric is a fully covariant grade 2 symmetric tensor field.
Given a basis $diff/(diff x^1),dots,diff/(diff x^n)$ of the tangent space $T_p M$ (induced by a chart)
the metric at a point $p$ can be fully represented as a matrix $amat(G) in RR^(n times n)$
by plugging in all combinations of basis vectors into the two arguments of the bilinear form.
$
  amat(G) = [g(diff/(diff x^i),diff/(diff x^j))]_(i,j=1)^(n times n)
$

$
  g_(i j) = g(diff/(diff x^i),diff/(diff x^j))
$


This is called a gramian matrix and can be used to represent any inner product
of a vector space, given a basis. We will use gramians to computationally represent
the metric at a point.

The inverse metric tensor gives an inner product on the cotangent space (covectors).
When using the dual basis the inverse metric tensor represented as a gramian in this basis is
just the matrix inverse of the metric gramian.
$
  amat(G)^(-1) = [g(dif x^i,dif x^j)]_(i,j=1)^(n times n)
$

$
  g^(i j) = g(dif x^i,dif x^j)
$

$
  g^(i k) g_(k j)​= delta_j^i
$

One can easily derive the Riemannian metric from
an embedding (or even an immersion) $f: M -> RR^N$. It's differential is a
function $dif f_p: T_p M -> T_p RR^n$, also called the push-forward and tells
us how our intrinsic tangential vectors are being stretched when viewed
geometrically.
The differential tells us also how to take an inner product of our tangent
vectors, by inducing a metric
$
  g(u, v) = dif f(u) dot dif f(v)
$

Computationally this differential $dif f$ can be represented, since it is a
linear map, by a Jacobi Matrix $amat(J)$.
The metric gramian can then be obtained by a simple matrix product.
$
  amat(G) = amat(J)^transp amat(J)
$


The Riemannian metric (tensor) is an inner product on the tangent space with
basis $diff/(diff x^i)$.\
The inverse metric is an inner product on the cotangent space with basis $dif x^i$.\

The fact that our geometry is piecewise-flat over the cells, means that
the metric is constant over each cell and changes only from cell to cell.

This piecewise-constant metric is known as the *Regge metric* and comes from
Regge calculus, a theory for numerical general relativety that is about
producing simplicial approximations of spacetimes that are solutions to the
Einstein field equation.

A global way to store the Regge metric is based of edge lengths. Instead
of giving all vertices a global coordinate, as one would do in extrinsic
geometry, we just give each edge in the mesh a positive length. Just knowing
the lengths doesn't tell you the positioning of the mesh in an ambient space
but it's enough to give the whole mesh it's piecewise-flat geometry.
Storing only the edge lengths of the whole mesh is a more memory efficent
representation of the geometry than storing all the metric tensors.

Mathematically this is just a function on the edges to the positive real numbers.
$
  f: Delta_1 (mesh) -> RR^+
$
that gives each edge $e in Delta_1 (mesh)$ a positive length $f(e) in RR^+$.

As an interesting side-note: If we would allow for pseudo-Riemannian manifolds
with a pseudo-metric, meaning we would drop the positive-definiteness requirement,
the edge lengths could become zero or even negative.

Computationally we repesent the edge lengths in a single struct
that has all lengths stored continuously in memory in a nalgebra vector.
```rust
pub struct MeshEdgeLengths {
  vector: na::DVector<f64>,
}
```
Our topological simplicial complex struct gives a global numbering to our
edges, which then gives us the indices into this nalgebra vector.

Our topological simplicial manifold together with these edge lengths
gives us a simplicial Riemannian manifold.

One can reconstruct the constant Regge metric on each cell based on the edge lengths
of this very cell. 
It can be derived via the law of cosines:
$
  amat(G)_(i j) = 1/2 (e_(0 i)^2 + e_(0 j)^2 - e_(i j)^2)
$

== Further Functionality

Formoniq implements some further functionality for the mesh,
such as importing and exporting meshes.
It supports loading gmsh meshes, as well as `.obj` meshes.
We are also consider implementing Visualization Toolkit (VTK) export
functionality.


== Manifold Crate

Our mesh implementation comes in the form of a Rust crate (library)
that has been published (TODO!) to https://crates.io/crate/manifold.
It could be used for other libraries or applications that built on top of it,
that are not necessarily FEEC related.

The structure of the crate is as follows:
```
manifold/src
├── lib.rs
├── topology.rs
├── topology
│   ├── simplex.rs
│   ├── skeleton.rs
│   ├── complex.rs
│   └── complex
│       ├── dim.rs
│       ├── attribute.rs
│       └── handle.rs
├── geometry.rs
├── geometry
│   ├── metric.rs
│   ├── coord.rs
│   └── coord
│       ├── local.rs
│       └── quadrature.rs
├── dim3.rs
├── gen.rs
├── gen
│   └── cartesian.rs
├── io.rs
└── io
    ├── blender.rs
    ├── gmsh.rs
    └── vtk.rs
```

= Exterior Algebra & Basis Representation

FEEC makes use of Exterior Calculus and Differential Forms. To develop
these notions a good starting point is exterior algebra.
This is just like how one first needs to learn about vector algebra, before
one can do vector calculus.

An Exterior Algebra is a construction over a vector space.
In this section we will consider this vector space $V$ to be fixed
together with some ordered basis ${e_i}_(i=1)^n$ and it's dual
space $V^*$ with the dual basis ${epsilon^i}_(i=1)^n$.
They are the representatives of the tangent space $T_p M$
and it's basis ${diff/(diff x^i)}_(i=1)^n$
and the cotangent space $T_p^* M$ and it's basis ${dif x^i}_(i=1)^n$ at some specific point $p$.

== Basis Representation

To do computations involving exterior algebras we want to create a datastructure.

If we choose an ordered basis $e_1,dots,e_n$ for our vector space $V$, this directly
induces a lexicographically ordered basis for each exterior power $wedgespace^k V$.
E.g. for $n=3,k=2$ we get an exterior basis $e_1 wedge e_2, e_1 wedge e_3, e_2, wedge e_3$.
We can now just store a list of coefficents for each exterior basis element
and represent in this way an element of an exterior algebra with just real numbers,
which is computationally easily represented.


We have dimensionality given by the binomial coefficent.
$
  dim wedgespace^k (V) = binom(n,k)
$


```rust
pub struct ExteriorElement<V: VarianceMarker> {
  coeffs: na::DVector<f64>,
  dim: Dim,
  grade: ExteriorGrade,
  variance: PhantomData<V>,
}
```

We can implement an iterator yielding a wedge term together with it's coefficent.
```rust
pub fn basis_iter(&self) -> impl Iterator<Item = (f64, ExteriorBase<V>)> + use<'_, V> {
  let dim = self.dim;
  let grade = self.grade;
  self
    .coeffs
    .iter()
    .copied()
    .enumerate()
    .map(move |(i, coeff)| {
      let basis = IndexSet::from_lex_rank(dim, grade, i).ext(dim);
      (coeff, basis)
    })
```

All standard vector space operations, such
as multivector addition and scalar multiplication have been implemented as well.

== Exterior Product

```rust
pub fn wedge(&self, other: &Self) -> Self {
  assert_eq!(self.dim, other.dim);
  let dim = self.dim;

  let new_grade = self.grade + other.grade;
  assert!(new_grade <= dim);

  let new_basis_size = binomial(self.dim, new_grade);
  let mut new_coeffs = na::DVector::zeros(new_basis_size);

  for (self_coeff, self_basis) in self.basis_iter() {
    for (other_coeff, other_basis) in other.basis_iter() {
      if self_basis == other_basis {
        continue;
      }
      if self_coeff == 0.0 || other_coeff == 0.0 {
        continue;
      }

      if let Some(merged_basis) = self_basis
        .indices()
        .clone()
        .union(other_basis.indices().clone())
        .try_into_sorted_signed()
      {
        let sign = merged_basis.sign;
        let merged_basis = merged_basis.set.lex_rank(dim);
        new_coeffs[merged_basis] += sign.as_f64() * self_coeff * other_coeff;
      }
    }
  }

  Self::new(new_coeffs, self.dim, new_grade)
}

pub fn wedge_big(factors: impl IntoIterator<Item = Self>) -> Option<Self> {
  let mut factors = factors.into_iter();
  let first = factors.next()?;
  let prod = factors.fold(first, |acc, factor| acc.wedge(&factor));
  Some(prod)
}
```


== Multivectors vs Multiforms

Space of alternating multilinear forms is an exterior algebra.
It's the dual exterior algebra.

The $k$-th exterior algebra $wedgespace^k (V^*)$ over the dual space $V^*$ of $V$ is
called the space of $k$-forms.\

== Inner product


Given a Riemannian metric $g$, we get an inner product on each fiber
$wedge.big^k T^*_p (Omega)$.

Computationally this is done using the basis and we compute an extended
gramian matrix for the inner product on $k$-forms using the determinant.
This can be further extended to an inner product on #strike[differential] $k$-forms
with basis $dif x_i_1 wedge dots wedge dif x_i_k$.
$
  inner(dif x_I, dif x_J) = det [inner(dif x_I_i, dif x_I_j)]_(i,j)^k
$

```rust
impl RiemannianMetricExt for RiemannianMetric {
  fn multi_form_gramian(&self, k: ExteriorGrade) -> na::DMatrix<f64> {
    let n = self.dim();
    let combinations: Vec<_> = IndexSubsets::canonical(n, k).collect();
    let covector_gramian = self.covector_gramian();

    let mut multi_form_gramian = na::DMatrix::zeros(combinations.len(), combinations.len());
    let mut multi_basis_mat = na::DMatrix::zeros(k, k);

    for icomb in 0..combinations.len() {
      let combi = &combinations[icomb];
      for jcomb in icomb..combinations.len() {
        let combj = &combinations[jcomb];

        for iicomb in 0..k {
          let combii = combi[iicomb];
          for jjcomb in 0..k {
            let combjj = combj[jjcomb];
            multi_basis_mat[(iicomb, jjcomb)] = covector_gramian[(combii, combjj)];
          }
        }
        let det = multi_basis_mat.determinant();
        multi_form_gramian[(icomb, jcomb)] = det;
        multi_form_gramian[(jcomb, icomb)] = det;
      }
    }
    multi_form_gramian
  }
}
```


= Discrete Differential Forms: Simplicial Cochains and Whitney Forms

In this chapter we will introduce discrete differential forms, which we will
represent as simplicial cochains.
We will discuss projection of arbitrary continuum differential forms expressed
in a global coordinate basis onto cochains and the finite element space of
whitney forms and it's basis.

This chapter corresponds exactly to the functionality of the `whitney` crate.

== Cochains

The discretization of differential forms on a mesh is of outmost importance.
For 1st order FEEC, as we are doing, the representation of discrete differential
forms is the same as in *discrete exterior calculus* (DEC) and is a
so called cochain. Cochains are isomorphic to the FE functions in 1st order FEEC.

A discrete differential $k$-form on a mesh is a $k$-cochain defined on this mesh.
This is a real-valued function $omega: Delta_k (mesh) -> RR$ defined on all
$k$-simplicies $Delta_k (mesh)$ of the mesh $mesh$.
One can represent this function on the simplicies, using a list of real values
that are ordered according to the global numbering of the simplicies.

```rust
pub struct Cochain {
  pub coeffs: na::DVector<f64>,
  pub dim: Dim,
}
```


=== Discretiation: Cochain-Projection via de Rham's map

The discretization of the continuous differential forms is then just a projection
onto this cochain space. This projection is the very geometric step of
integration the continuous differential form over each $k$-simplex to obtain
the real value, the function has as output.
This map is called the *de Rham map*.
It ensures that the homology is preserved.

$
  omega_sigma = integral_sigma omega
  quad forall sigma in Delta_k (mesh)
$

```rust
/// Discretize continuous coordinate-based differential k-form into
/// discrete k-cochain on CoordComplex via de Rham map (integration over k-simplex).
pub fn discretize_form_on_mesh(
  form: &impl DifferentialMultiForm,
  topology: &Complex,
  coords: &MeshVertexCoords,
) -> Cochain<Dim> {
  let cochain = topology
    .skeleton(form.grade())
    .handle_iter()
    .map(|simp| SimplexCoords::from_simplex_and_coords(simp.simplex_set(), coords))
    .map(|simp| discretize_form_on_simplex(form, &simp))
    .collect::<Vec<_>>()
    .into();
  Cochain::new(form.grade(), cochain)
}

/// Approximates the integral of a differential k-form over a k-simplex,
/// by means of barycentric quadrature.
pub fn discretize_form_on_simplex(
  differential_form: &impl DifferentialMultiForm,
  simplex: &SimplexCoords,
) -> f64 {
  let multivector = simplex.spanning_multivector();
  let f = |coord: CoordRef| {
    differential_form
      .at_point(simplex.local_to_global_coord(coord).as_view())
      .on_multivector(&multivector)
  };
  let std_simp = SimplexCoords::standard(simplex.dim_intrinsic());
  barycentric_quadrature(&f, &std_simp)
}
```


=== Discrete Exterior Derivative via Stokes' Theorem

The exterior derivative is the derivative in exterior calculus.
We want to define a discrete exterior derivative for our discrete differential forms.
This is done with some really simple cochain calculus.
We make us of the famous *Stokes' Theorem* for chains, that relates the exterior
derivative to the boundary of the chain.
$
  integral_c dif omega = integral_(diff c) omega
$
We can express this rule using a dual pairing
$
  inner(dif omega, c) = inner(omega, diff c)
$

This inspires a defintion of the discrete exterior derivative as
the opposite of the boundary operator, the *coboundary* operator.
$
  dif omega(c) = omega(diff c)
$

From a computational standpoint, the boundary operator is a signed incidence matrix
and this definition makes the coboundary operator be the transpose of this
signed incidence matrix.
$
  dif^k = diff_(k+1)^transp
$
```rust
/// Extension trait
pub trait ManifoldComplexExt { ... }
impl ManifoldComplexExt for Complex {
  /// $dif^k: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$
  fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> SparseMatrix {
    self.boundary_operator(grade + 1).transpose()
  }
}
```

By construction Stokes' Theorem is fullfiled, which is crucial to many applications.

The exterior derivative is closed in the space of Whitney forms, because of the de Rham complex.

The local (on a single cell) exterior derivative is always the same for any cell.
Therefore we can compute it on the reference cell.


== Whitney Forms

The basis we will be working with is called the Whitney basis.
The Whitney space is the space of piecewise-linear (over the cells)
differential forms.


To give an idea of the type of functions we are dealing with, we will
look at visualizations of the basis functions in the 2 dimensional case.

$
  lambda_(i j) = lambda_i dif lambda_j - lambda_j dif lambda_i
$

$
  lambda_01 &= (1-y) dif x + x dif y
  \
  lambda_02 &= y dif x + (1-x) dif y
  \
  lambda_12 &= -y dif x + x dif y
$

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    rows: 1,
    gutter: 3pt,
    image("../res/ref_lambda01.png", width: 100%),
    image("../res/ref_lambda02.png", width: 100%),
    image("../res/ref_lambda12.png", width: 100%),
  ),
  caption: [
    Vector proxies of Reference Local Shape Functions
    $lambda_01, lambda_02, lambda_12 in cal(W) Lambda^1 (Delta_2^"ref")$.
  ],
) <img:ref_whitneys>


#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    rows: 1,
    gutter: 3pt,
    image("../res/eq_phi01.png", width: 100%),
    image("../res/eq_phi02.png", width: 100%),
    image("../res/eq_phi12.png", width: 100%),
  ),
  caption: [
    Vector proxies of Global Shape Functions
    $phi_01, phi_02, phi_12 in cal(W) Lambda^1 (mesh)$ \
    on equilateral triangle mesh $mesh$.
  ],
) <img:global_whitneys>

=== Reconstruction: Whitney Interpolatoin via the Whitney map

Whitney forms are the piecewise-linear (over the cells) differential forms
that can be uniquely reconstructed from cochains. This reconstruction
is achieved by the so called *Whitney map*. It can be seen as a
generalized interpolation (in an integral sense instead of a pointwise sense)
operator.

They are our finite element differential forms. Our finite element function space
is the space $cal(W) Lambda^k (mesh)$ of Whitney forms over our mesh $mesh$.

The Whitney $k$-form basis function live on all $k$-simplicies of the mesh $mesh$.
$
  cal(W) Lambda^k (mesh) = "span" {lambda_sigma : sigma in Delta_k (mesh)}
$

Each Whitney $k$-form is associated with a particular $k$-simplex.
This simplex is the DOF and it's coefficent is the cochain value
on this simplex.

There is a isomorphism between Whitney $k$-forms and cochains.\
Represented through the de Rham map (discretization) and Whitney interpolation:\
- The integration of each Whitney $k$-form over its associated $k$-simplex yields a $k$-cochain.
- The interpolation of a $k$-cochain yields a Whitney $k$-form.\


Whitney forms are essential for us to compute our element matrices.
For this we just need to express Whitney forms locally on a cell
in the barycentric coordinate basis.

Another important use for Whitney forms is the reconstruction
of the global solution once we have computed the basis expansion--
which is the cochain-- into a point-evaluatable differential form.
For this we will express Whitney forms in a global coordinate basis.

For this we have a simple struct that represents a particular Whitney form
inside of a coordinatized cell associated with it's DOF subsimplex.
```rust
pub struct WhitneyForm<O: SetOrder> {
  cell_coords: SimplexCoords,
  associated_subsimp: Simplex<O>,
  difbarys: Vec<MultiForm>,
}
```
We store the the vertex coordiantes of the cell, the local subsimplex
and additionaly store the precomputed constant exterior derivatives of the
barycentric coordinate functions.
```rust
impl<O: SetOrder> WhitneyForm<O> {
  pub fn new(cell_coords: SimplexCoords, associated_subsimp: Simplex<O>) -> Self {
    let difbarys = associated_subsimp
      .vertices
      .iter()
      .map(|vertex| cell_coords.difbary(vertex))
      .collect();

    Self {
      cell_coords,
      associated_subsimp,
      difbarys,
    }
  }
}

```

The local Whitney form $lambda_(i_0 dots i_k)$ associated with the DOF simplex
$sigma = [i_0 dots i_k] subset.eq tau$ on the cell $tau = [j_0 dots j_n]$ is
defined using the barycentric coordinate functions $lambda_i_s$ of the cell.
$
  lambda_(i_0 dots i_k) =
  k! sum_(l=0)^k (-1)^l lambda_i_l
  (dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k)
$

We can observe that the big wedge terms
$dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k$
are constant. We write a method any of these terms.
```rust
pub fn wedge_term(&self, iterm: usize) -> MultiForm {
  let wedge_terms = self
    .difbarys
    .iter()
    .enumerate()
    // leave off i'th difbary
    .filter_map(|(ipos, bary)| (ipos != iterm).then_some(bary.clone()));
  MultiForm::wedge_big(wedge_terms).unwrap_or(MultiForm::one(self.dim()))
}
pub fn wedge_terms(&self) -> MultiFormList {
  (0..self.difbarys.len())
    .map(|i| self.wedge_term(i))
    .collect()
}
```
The Whitney forms as a whole however is not constant but varies over the cell.
We write a function to evaluate the Whitney form at any coordinate.
For this we implement the `ExteriorField` trait from our `exterior` crate
for `WhitneyForm`.
```rust
impl<O: SetOrder> ExteriorField for WhitneyForm<O> {
  type Variance = variance::Co;
  fn dim(&self) -> Dim { self.cell_coords.dim_embedded() }
  fn grade(&self) -> ExteriorGrade { self.associated_subsimp.dim() }

  fn at_point<'a>(
    &self,
    coord_global: impl Into<CoordRef<'a>>
  ) -> ExteriorElement<Self::Variance> {
    let coord_global = coord_global.into();
    assert_eq!(coord_global.len(), self.dim());
    let barys = self.cell_coords.global_to_bary_coord(coord_global);

    let dim = self.dim();
    let grade = self.grade();
    let mut form = MultiForm::zero(dim, grade);
    for (i, vertex) in self.associated_subsimp.vertices.iter().enumerate() {
      let sign = Sign::from_parity(i);
      let wedge = self.wedge_term(i);

      let bary = barys[vertex];
      form += sign.as_f64() * bary * wedge;
    }
    factorial(grade) as f64 * form
  }
}
```

Since the Whitney form is a linear differential form over the cell,
it's exterior derivative must be a constant multiform.
We can easily derive it's value.
$
  dif lambda_(i_0 dots i_k)
  &= k! sum_(l=0)^k (-1)^l dif lambda_i_l wedge
  (dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k)
  \
  &= k! sum_(l=0)^k (-1)^l (-1)^l
  (dif lambda_i_0 wedge dots.c wedge dif lambda_i_l wedge dots.c wedge dif lambda_i_k)
  \
  &= (k+1)! dif lambda_i_0 wedge dots.c wedge dif lambda_i_k
$

We implement another function to compute it as well.
```rust
pub fn dif(&self) -> MultiForm {
  if self.grade() == self.dim() {
    return MultiForm::zero(self.dim(), self.grade() + 1);
  }
  let factorial = factorial(self.grade() + 1) as f64;
  let difbarys = self.difbarys.clone();
  factorial * MultiForm::wedge_big(difbarys).unwrap()
}
```

The defining property of the Whitney basis is a from pointwise to integral
generalized Lagrange basis property:\
For any two $k$-simplicies $sigma, tau in Delta_k (mesh)$, we have
$
  integral_sigma lambda_tau = cases(
    +&1 quad &"if" sigma = +tau,
    -&1 quad &"if" sigma = -tau,
     &0 quad &"if" sigma != plus.minus tau,
  )
$

We can write a test that verifies our implementation by checking this property.


= Finite Element Methods for Differential Forms

We have now arrived at the chapter talking about the
actual finite element library formoniq. \
Here we will derive and implement the formulas for computing the element matrices
of the various weak differential operators in FEEC.
Furthermore we implement the assembly algorithm that will give us the
final galerkin matrices.

== Variational Formulation & Element Matrix Computation

There are only 4 types of variational operators that
are relevant to the mixed weak formulation of the Hodge-Laplace operator.
All of them are based on the inner product on Whitney forms.

Above all is the mass bilinear form, which really is just
the inner product.
$
  m^k (u, v) &= inner(u, v)_(L^2 Lambda^k (Omega))
  quad
  u in L^2 Lambda^k, v in L^2 Lambda^k
$

The next bilinear form is the one for the exterior derivative
$
  d^k (u, v) &= inner(dif u, v)_(L^2 Lambda^k (Omega))
  quad
  u in H Lambda^(k-1), v in L^2 Lambda^k
$

Also relevant is the bilinear form of the codifferential
$
  c(u, v) &= inner(delta u, v)_(L^2 Lambda^k (Omega))
$
Using the adjoint property we can rewrite it using the exterior derivative
applied to the test function.
$
  c^k (u, v) &= inner(u, dif v)_(L^2 Lambda^k (Omega))
  quad
  u in L^2 Lambda^k, v in H Lambda^(k-1)
$

Lastly there is the bilinear form in the style of the scalar Laplacian, with
exterior derivatives on both arguments. It's $delta dif u$, which for a 0-form
is $div grad u = Delta u$.
$
  l^k (u, v) &= inner(dif u, dif v)_(L^2 Lambda^(k+1) (Omega))
  quad
  u in H Lambda^k, v in H Lambda^k
$

After Galerkin discretization, by means of the Whitney finite element space
with the Whitney basis, we arrive at the following Galerkin matrices for our
four weak operators.
$
  amat(M)^k &= [inner(phi^k_i, phi^k_j)]_(i j) \
  amat(D)^k &= [inner(phi^k_i, dif phi^(k-1)_j)]_(i j) \
  amat(C)^k &= [inner(dif phi^(k-1)_i, phi^k_j)]_(i j) \
  amat(L)^k &= [inner(dif phi^k_i, dif phi^k_j)]_(i j) \
$

We can rewrite the 3 operators that involve the exterior derivative
using the mass matrix and the discrete exterior derivative (incidence matrix).
$
  amat(D)^k &= amat(M)^k amat(dif)^(k-1) \
  amat(C)^k &= (amat(dif)^(k-1))^transp amat(M)^k \
  amat(L)^k &= (amat(dif)^k)^transp amat(M)^(k+1) amat(dif)^k \
$

As usual in a FEM library, we define element matrix providers,
that compute the element matrices on each cell of mesh and later on
assemble the full galerkin matrices.

We first define a element matrix provider trait
```rust
pub type ElMat = na::DMatrix<f64>;
pub trait ElMatProvider {
  fn row_grade(&self) -> ExteriorGrade;
  fn col_grade(&self) -> ExteriorGrade;
  fn eval(&self, geometry: &SimplexGeometry) -> ElMat;
}
```
The `eval` method provies us with the element matrix on a
cell, given it's geometry. But we also need to know the exterior grade
of the Whitney forms that correspond to the rows and columns.
This information will be used by the assembly routine.

We will now implement the 3 operators involving exterior derivatives
based on the element matrix provider of the mass bilinear form.

The local exterior derivative only depends on the local topology, which is the same
for any simplex of the same dimension. So we use a global variable that stores
the transposed incidence matrix for any $k$-skeleton of a $n$-complex.

```rust
pub struct DifElmat(pub ExteriorGrade);
impl ElMatProvider for DifElmat {
  fn row_grade(&self) -> ExteriorGrade { self.0 }
  fn col_grade(&self) -> ExteriorGrade { self.0 - 1 }
  fn eval(&self, geometry: &SimplexGeometry) -> na::DMatrix<f64> {
    let dim = geometry.dim();
    let grade = self.0;
    let dif = &LOCAL_DIFFERENTIAL_OPERATORS[dim][grade - 1];
    let mass = HodgeMassElmat(grade).eval(geometry);
    mass * dif
  }
}

pub struct CodifElmat(pub ExteriorGrade);
impl ElMatProvider for CodifElmat {
  fn row_grade(&self) -> ExteriorGrade { self.0 - 1 }
  fn col_grade(&self) -> ExteriorGrade { self.0 }
  fn eval(&self, geometry: &SimplexGeometry) -> na::DMatrix<f64> {
    let dim = geometry.dim();
    let grade = self.0;
    let dif = &LOCAL_DIFFERENTIAL_OPERATORS[dim][grade - 1];
    let codif = dif.transpose();
    let mass = HodgeMassElmat(grade).eval(geometry);
    codif * mass
  }
}

pub struct CodifDifElmat(pub ExteriorGrade);
impl ElMatProvider for CodifDifElmat {
  fn row_grade(&self) -> ExteriorGrade { self.0 }
  fn col_grade(&self) -> ExteriorGrade { self.0 }
  fn eval(&self, geometry: &SimplexGeometry) -> na::DMatrix<f64> {
    let dim = geometry.dim();
    let grade = self.0;
    let dif = &LOCAL_DIFFERENTIAL_OPERATORS[dim][grade];
    let codif = dif.transpose();
    let mass = HodgeMassElmat(grade + 1).eval(geometry);
    codif * mass * dif
  }
}
```

This was really easy.


=== Mass bilinear form

Now we need to implement the element matrix provider to the mass bilinear form.
Here is where the geometry of the domain comes in, through the inner product, which
depends on the Riemannian metric.

One could also understand the mass bilinear form as a weak hodge star operator.
$
  amat(M)_(i j) = integral_Omega phi_j wedge hodge phi_i = inner(phi_j, phi_i)_(L^2 Lambda^k)
$

We will not compute this using the hodge star operator, but instead directly
using the inner product.

We already have an inner product on constant multiforms. We now need to
extend it to an $L^2$ inner product on Whitney forms.
This can be done by inserting the defintion of a Whitney form (in terms of barycentric
coordiante functions) into the inner product.

$
  inner(lambda_(i_0 dots i_k), lambda_(j_0 dots j_k))_(L^2)
  &= k!^2 sum_(l=0)^k sum_(m=0)^k (-)^(l+m) innerlines(
    lambda_i_l (dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k),
    lambda_j_m (dif lambda_j_0 wedge dots.c wedge hat(dif lambda)_j_m wedge dots.c wedge dif lambda_j_k),
  )_(L^2) \
  &= k!^2 sum_(l,m) (-)^(l+m) innerlines(
    dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k,
    dif lambda_j_0 wedge dots.c wedge hat(dif lambda)_j_m wedge dots.c wedge dif lambda_j_k,
  )
  integral_K lambda_i_l lambda_j_m vol \
$

We can now make use of the fact that the exterior derivative of the barycentric
coordinate functions are constant. This makes the wedge big terms also constant.
We acn therefore pull them out of the integral inside the $L^2$-inner product
and now it's just an inner product on constant multiforms.
What remains in the in the integral is the product of two barycentric
coordinate functions. This integral is with respect to the metric and contains
all geometric information.


Using this we can now implement the element matrix provider
to the mass bilinear form in Rust.
```rust
pub struct HodgeMassElmat(pub ExteriorGrade);
impl ElMatProvider for HodgeMassElmat {
  fn row_grade(&self) -> ExteriorGrade { self.0 }
  fn col_grade(&self) -> ExteriorGrade { self.0 }
  fn eval(&self, geometry: &SimplexGeometry) -> na::DMatrix<f64> {
    let dim = geometry.dim();
    let grade = self.0;

    let nvertices = grade + 1;
    let simplicies: Vec<_> = subsimplicies(dim, grade).collect();

    let wedge_terms: Vec<_> = simplicies
      .iter()
      .cloned()
      .map(|simp| WhitneyForm::new(SimplexCoords::standard(dim), simp).wedge_terms())
      .collect();

    let scalar_mass = ScalarMassElmat.eval(geometry);

    let mut elmat = na::DMatrix::zeros(simplicies.len(), simplicies.len());
    for (i, asimp) in simplicies.iter().enumerate() {
      for (j, bsimp) in simplicies.iter().enumerate() {
        let wedge_terms_a = &wedge_terms[i];
        let wedge_terms_b = &wedge_terms[j];
        let wedge_inners = geometry
          .metric()
          .multi_form_inner_product_mat(wedge_terms_a, wedge_terms_b);

        let mut sum = 0.0;
        for avertex in 0..nvertices {
          for bvertex in 0..nvertices {
            let sign = Sign::from_parity(avertex + bvertex);

            let inner = wedge_inners[(avertex, bvertex)];

            sum += sign.as_f64()
              * inner
              * scalar_mass[(asimp.vertices[avertex], bsimp.vertices[bvertex])];
          }
        }

        elmat[(i, j)] = sum;
      }
    }

    (factorial(grade) as f64).powi(2) * elmat
  }
}
```

Now we are just missing an element matrix providre for the scalar mass
bilinear form.
Luckily there exists a closed form solution, for
this integral, which only depends on the volume of the cell.
$
  integral_K lambda_i lambda_j vol
  = abs(K)/((n+2)(n+1)) (1 + delta_(i j))
$
derived from this more general integral formula for powers of barycentric coordinate functions
$
  integral_K lambda_0^(alpha_0) dots.c lambda_n^(alpha_n) vol
  =
  n! abs(K) (alpha_0 ! space dots.c space alpha_n !)/(alpha_0 + dots.c + alpha_n + n)!
$
$K in Delta_n, avec(alpha) in NN^(n+1)$

```rust
pub struct ScalarMassElmat;
impl ElMatProvider for ScalarMassElmat {
  fn row_grade(&self) -> ExteriorGrade { 0 }
  fn col_grade(&self) -> ExteriorGrade { 0 }
  fn eval(&self, geometry: &SimplexGeometry) -> ElMat {
    let ndofs = geometry.nvertices();
    let dim = geometry.dim();
    let v = geometry.vol() / ((dim + 1) * (dim + 2)) as f64;
    let mut elmat = na::DMatrix::from_element(ndofs, ndofs, v);
    elmat.fill_diagonal(2.0 * v);
    elmat
  }
}
```

== Assembly

The element matrix provider tells the assembly routine,
what the exterior grade is of the arguments into the bilinear forms,
based on this the right dimension of simplicies are used to assemble.

```rust
pub fn assemble_galmat(
  topology: &Complex,
  geometry: &MeshEdgeLengths,
  elmat: impl ElMatProvider,
) -> GalMat {
  let row_grade = elmat.row_grade();
  let col_grade = elmat.col_grade();

  let nsimps_row = topology.skeleton(row_grade).len();
  let nsimps_col = topology.skeleton(col_grade).len();

  let mut galmat = SparseMatrix::zeros(nsimps_row, nsimps_col);
  for cell in topology.cells().handle_iter() {
    let geo = geometry.simplex_geometry(cell);
    let elmat = elmat.eval(&geo);

    let row_subs: Vec<_> = cell.subsimps(row_grade).collect();
    let col_subs: Vec<_> = cell.subsimps(col_grade).collect();
    for (ilocal, &iglobal) in row_subs.iter().enumerate() {
      for (jlocal, &jglobal) in col_subs.iter().enumerate() {
        galmat.push(iglobal.kidx(), jglobal.kidx(), elmat[(ilocal, jlocal)]);
      }
    }
  }
  galmat
}
```

= Hodge-Laplacian

In this chapter we now solve some PDEs based on the Hodge-Laplace operator.
We consider the Hodge-Laplace eigenvalue problem and the Hodge-Laplace source
problem (analog of Poisson equation).

The Hodge-Laplace operator generalizes the scalar (0-form) Laplace-Beltrami operator,
to an operator acting on any differential $k$-form. As such the 0-form Hodge-Laplacian
$Delta^0$ is exactly the Laplace-Beltrami operator and we can write it using
the exterior derivative $dif$ and the codifferential $delta$.
$
  Delta^0 f = -div grad f = delta^1 dif^0 f
$

The $k$-form Hodge-Laplacian $Delta^k$ is defined as
$
  Delta^k: Lambda^k (Omega) -> Lambda^k (Omega)
  \
  Delta^k = dif^(k-1) delta^k + delta^(k+1) dif^k
$

== Eigenvalue Problem

We first consider the Eigenvalue problem, because it's a bit simpler
and the source problem, relies on the eigenvalue problem.

The strong primal form of the Hodge-Laplace eigenvalue problem is\
Find $lambda in RR, u in Lambda^k (Omega)$, s.t.
$
  (dif delta + delta dif) u = lambda u
$


In FEM we don't solve the PDE based on the strong form, but instead we rely
on a weak variational form.
The primal weak form is not suited for discretization, so instead we make use of
a mixed variational form that includes an auxiliary variable $sigma$.

The mixed weak form is\
Find $lambda in RR$, $(sigma, u) in (H Lambda^(k-1) times H Lambda^k \\ {0})$, s.t.
$
  inner(sigma, tau) - inner(u, dif tau) &= 0
  quad &&forall tau in H Lambda^(k-1)
  \
  inner(dif sigma, v) + inner(dif u, dif v) &= lambda inner(u,v)
  quad &&forall v in H Lambda^k
$

This formulation involves exactly the bilinear forms, we have implemented
in one of the previous chapters.

We now perform Galerkin discretization of this variational problem by choosing
as finite dimensional subspace of our function space $H Lambda^k$ the space of Whitney forms
$cal(W) lambda^k subset.eq H Lambda^k$ and as basis the Whitney basis ${phi^k_i}$.
We then replace $sigma$ and $u$ by basis expansions $sigma = sum_j sigma_j phi^(k-1)_j$,
$u = sum_i u_i phi^k_i$ and arrive at the linear system of equations.
$
  sum_j sigma_j inner(phi^(k-1)_j, phi^(k-1)_i) - sum_j u_j inner(phi^k_j, dif phi^(k-1)_i) &= 0
  \
  sum_j sigma_j inner(dif phi^(k-1)_j, phi^k_i) + sum_j u_j inner(dif phi^k_j, dif phi^k_i) &= lambda sum_j u_j inner(phi^k_j,phi^k_i)
$


We can now insert our Galerkin matrices.
$
  amat(M)^(k-1) = [inner(phi^(k-1)_i, phi^(k-1)_j)]_(i j)
  quad
  amat(C) = [inner(dif phi^(k-1)_i, phi^k_j)]_(i j)
  \
  amat(D) = [inner(phi^k_i, dif phi^(k-1)_j)]_(i j)
  quad
  amat(L) = [inner(dif phi^k_i, dif phi^k_j)]_(i j)
  quad
  amat(M)^k = [inner(phi^k_i, phi^k_j)]_(i j)
$

And arrive at the LSE.
$
  amat(M)^(k-1) avec(sigma) - amat(C) avec(u) = amat(0)
  \
  amat(D) avec(sigma) + amat(L) avec(u) = lambda amat(M)^k avec(u)
$

This LSE has block structure and can be written as
$
  mat(
    amat(M)^(k-1), -amat(C);
    amat(D), amat(L);
  )
  vec(avec(sigma), avec(u))
  =
  lambda
  mat(
    amat(0)_(sigma times sigma),amat(0)_(sigma times u);
    amat(0)_(u times sigma),amat(M)^k
  )
  vec(avec(sigma), avec(u))
$

This is a symmetric indefinite sparse generalized matrix eigenvalue problem,
that can be solved by an iterative eigensolver such as Krylov-Schur.
In SLEPc terminology this is called a GHIEP problem.

```rust
pub struct MixedGalmats {
  mass_sigma: GalMat,
  dif_sigma: GalMat,
  codif_u: GalMat,
  difdif_u: GalMat,
  mass_u: GalMat,
}
impl MixedGalmats {
  pub fn compute(topology: &Complex, geometry: &MeshEdgeLengths, grade: ExteriorGrade) -> Self {
    let (mass_sigma, dif_sigma, codif_u) = if grade > 0 {
      (
        assemble_galmat(topology, geometry, HodgeMassElmat(grade - 1)),
        assemble_galmat(topology, geometry, DifElmat(grade)),
        assemble_galmat(topology, geometry, CodifElmat(grade)),
      )
    } else {
      (GalMat::default(), GalMat::default(), GalMat::default())
    };
    let difdif_u = assemble_galmat(topology, geometry, CodifDifElmat(grade));
    let mass_u = assemble_galmat(topology, geometry, HodgeMassElmat(grade));

    Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      difdif_u,
      mass_u,
    }
  }

  pub fn sigma_len(&self) -> usize {
    self.mass_sigma.nrows()
  }
  pub fn u_len(&self) -> usize {
    self.mass_u.nrows()
  }

  pub fn mixed_hodge_laplacian(&self) -> SparseMatrix {
    let Self {
      mass_sigma,
      dif_sigma,
      codif_u,
      difdif_u,
      ..
    } = self;
    let codif_u = codif_u.clone();
    SparseMatrix::block(&[&[mass_sigma, &(-codif_u)], &[dif_sigma, difdif_u]])
  }
}
```

```rust
pub fn solve_hodge_laplace_evp(
  topology: &Complex,
  geometry: &MeshEdgeLengths,
  grade: ExteriorGrade,
  neigen_values: usize,
) -> (na::DVector<f64>, na::DMatrix<f64>) {
  let galmats = MixedGalmats::compute(topology, geometry, grade);

  let lhs = galmats.mixed_hodge_laplacian();

  let sigma_len = galmats.sigma_len();
  let u_len = galmats.u_len();
  let mut rhs = SparseMatrix::zeros(sigma_len + u_len, sigma_len + u_len);
  for &(mut r, mut c, v) in galmats.mass_u.triplets() {
    r += sigma_len;
    c += sigma_len;
    rhs.push(r, c, v);
  }

  petsc_ghiep(
    &lhs.to_nalgebra_csr(),
    &rhs.to_nalgebra_csr(),
    neigen_values,
  )
}
```

== Source Problem

The Hodge-Laplace Source Problem is the generalization of the Poisson equation
to arbitrary differential $k$-forms. In strong form it is\
Find $u in Lambda^k (Omega)$, given $f in Lambda^k (Omega)$, s.t.
$
  Delta u = f - P_frak(H) f, quad u perp frak(H)
$
This equation is not quite as simple as the normal Poisson equation $Delta u = f$.
Instead it includes two additional parts involving $frak(H)$, which is the space
of harmonic forms $frak(H)^k = ker Delta = { v in Lambda^k mid(|) Delta v = 0}$.
The first change is that we remove the harmonic part $P_frak(H) f$ of $f$. The second
difference is that we require that our solution $u$ is orthogonal to harmonic forms.

The harmonic forms give a concrete realizations of the cohomology.
They are a representative of the cohomology quotient group $cal(H)^k = (ker dif)/(im dif)$
and as such they are isomorphic $frak(H)^k = cal(H)^k$.

We once again tackle a mixed weak formulation based on the auxiliary variable $sigma$
and this time a second one $p$ that represents $f$ without harmonic component.\
Given $f in L^2 Lambda^k$, find $(sigma,u,p) in (H Lambda^(k-1) times H Lambda^k times frak(H)^k)$ s.t.
$
  inner(sigma,tau) - inner(u,dif tau) &= 0
  quad &&forall tau in H Lambda^(k-1)
  \
  inner(dif sigma,v) + inner(dif u,dif v) + inner(p,v) &= inner(f,v)
  quad &&forall v in H Lambda^k
  \
  inner(u,q) &= 0
  quad &&forall q in frak(H)^k
$

We once again perform Galerkin discretization.
$
  sum_j sigma_j inner(phi^(k-1)_j,phi^(k-1)_i) - sum_j u_j inner(phi^k_j,dif phi^(k-1)_i) &= 0
  \
  sum_j sigma_j inner(dif phi^(k-1)_j,phi^k_i) + sum_j u_j inner(dif phi^k_j,dif phi^k_i) + sum_j p_j inner(eta^k_j,phi^k_i) &= sum_j f_j inner(psi_j,phi^k_i)
  \
  sum_j u_j inner(phi^k_j,eta^k_i) &= 0
$

By inserting our known Galerkin matrices, we obtain.


$
  amat(M)^(k-1) avec(sigma) - amat(C) avec(u) = 0
  \
  amat(D) avec(sigma) + amat(L) avec(u) + amat(M) amat(H) avec(p) = amat(M)^k avec(f)
  \
  amat(H)^transp amat(M) avec(u) = 0
$


//$
//  hodge sigma - dif^transp hodge u &= 0
//  \
//  hodge dif sigma + dif^transp hodge dif u + hodge H p &= hodge f
//  \
//  H^transp hodge u &= 0
//$

Or in block-structure
$
  mat(
    amat(M)^(k-1), -amat(C), 0;
    amat(D), amat(L), amat(M)amat(H);
    0, amat(H)^transp amat(M), 0
  )
  vec(avec(sigma), avec(u), avec(p))
  =
  vec(0, amat(M)^k avec(f), 0)
$

//$
//  mat(
//    hodge, -dif^transp hodge, 0;
//    hodge dif, dif^transp hodge dif, hodge H;
//    0, H^transp hodge, 0;
//  )
//  vec(sigma, u, p)
//  =
//  vec(0, hodge f, 0)
//$

Compute harmonics
```rust
pub fn solve_hodge_laplace_harmonics(
  topology: &Complex,
  geometry: &MeshEdgeLengths,
  grade: ExteriorGrade,
) -> na::DMatrix<f64> {
  // TODO!!!
  //let homology_dim = topology.homology_dim(grade);
  let homology_dim = 0;

  if homology_dim == 0 {
    let nwhitneys = topology.nsimplicies(grade);
    return na::DMatrix::zeros(nwhitneys, 0);
  }

  let (eigenvals, harmonics) = solve_hodge_laplace_evp(topology, geometry, grade, homology_dim);
  assert!(eigenvals.iter().all(|&eigenval| eigenval <= 1e-12));
  harmonics
}
```

```rust
pub fn solve_hodge_laplace_source(
  topology: &Complex,
  geometry: &MeshEdgeLengths,
  grade: ExteriorGrade,
  source_data: Cochain,
) -> (Cochain, Cochain, Cochain) {
  let harmonics = solve_hodge_laplace_harmonics(topology, geometry, grade);

  let galmats = MixedGalmats::compute(topology, geometry, grade);

  let mass_u = galmats.mass_u.to_nalgebra_csr();
  let mass_harmonics = &mass_u * &harmonics;

  let sigma_len = galmats.sigma_len();
  let u_len = galmats.u_len();

  let mut galmat = galmats.mixed_hodge_laplacian();

  galmat.grow(mass_harmonics.ncols(), mass_harmonics.ncols());

  for (mut r, mut c) in (0..mass_harmonics.nrows()).cartesian_product(0..mass_harmonics.ncols()) {
    let v = mass_harmonics[(r, c)];
    r += sigma_len;
    c += sigma_len + u_len;
    galmat.push(r, c, v);
  }
  for (mut r, mut c) in (0..mass_harmonics.nrows()).cartesian_product(0..mass_harmonics.ncols()) {
    let v = mass_harmonics[(r, c)];
    // transpose
    mem::swap(&mut r, &mut c);
    r += sigma_len + u_len;
    c += sigma_len;
    galmat.push(r, c, v);
  }

  let galmat = galmat.to_nalgebra_csr();

  let galvec = mass_u * source_data.coeffs;
  #[allow(clippy::toplevel_ref_arg)]
  let galvec = na::stack![
    na::DVector::zeros(sigma_len);
    galvec;
    na::DVector::zeros(harmonics.ncols());
  ];

  let galsol = petsc_saddle_point(&galmat, &galvec);
  let sigma = Cochain::new(grade - 1, galsol.view_range(..sigma_len, 0).into_owned());
  let u = Cochain::new(
    grade,
    galsol
      .view_range(sigma_len..sigma_len + u_len, 0)
      .into_owned(),
  );
  let p = Cochain::new(
    grade,
    galsol.view_range(sigma_len + u_len.., 0).into_owned(),
  );
  (sigma, u, p)
}
```

= Results

To verify the function of the library we solve a EVP and a source problem.

== Source Problem

We verify the source problem by means of the method of manufactured solution.
Our manifactured solution is a 1-form that follows the same pattern for any
dimensions.

$
  Omega = [0,pi]^n
$

$
  u_i = sin^2 (x^i) product_(j != i) cos(x^j)
$

$
  n=2 ==> u = vec(
    sin^2(x) cos(y),
    cos(x) sin^2(y),
  )
  quad quad
  n=3 ==> u = vec(
    sin^2(x) cos(y) cos(z),
    cos(x) sin^2(y) cos(z),
    cos(x) cos(y) sin^2(z),
  )
$

$
  (Delta^1 avec(u))_i = Delta^0 u_i = -(2 cos(2 x^i) - (n-1) sin^2(x^i)) product_(j != i) cos(x^j)
$

Homogeneous boundary conditions.
$
  trace_(diff Omega) u = 0
  quad quad
  trace_(diff Omega) dif u = 0
$

Non-trivial
$
  curl avec(u) != 0
  quad quad
  div avec(u) != 0
$

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
  let dim = 2;
  let form_grade = 1;

  let exact_solution = |p: CoordRef| {
    let comps = (0..p.len()).map(|i| {
      let prod = p.remove_row(i).map(|a| a.cos()).product();
      p[i].sin().powi(2) * prod
    });
    MultiForm::from_grade1(na::DVector::from_iterator(p.len(), comps))
  };
  let laplacian = |p: CoordRef| {
    let comps = (0..p.len()).map(|i| {
      let prod: f64 = p.remove_row(i).map(|a| a.cos()).product();
      -(2.0 * (2.0 * p[i]).cos() - (p.len() - 1) as f64 * p[i].sin().powi(2)) * prod
    });
    MultiForm::from_grade1(na::DVector::from_iterator(p.len(), comps))
  };

  let laplacian = DifferentialFormClosure::new(Box::new(laplacian), dim, form_grade);
  let exact_solution = DifferentialFormClosure::new(Box::new(exact_solution), dim, form_grade);

  let mut errors = Vec::new();
  for refinement in 0..=10 {
    let nboxes_per_dim = 2usize.pow(refinement);
    let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, PI);
    let (topology, coords) = box_mesh.compute_coord_complex();
    let metric = coords.to_edge_lengths(&topology);

    let laplacian = discretize_form_on_mesh(&laplacian, &topology, &coords);
    let exact_solution = discretize_form_on_mesh(&exact_solution, &topology, &coords);

    let (_sigma, u, _p) =
      hodge_laplace::solve_hodge_laplace_source(&topology, &metric, form_grade, laplacian);

    let diff = exact_solution - u;
    let l2_norm = l2_norm(&diff, &topology, &metric);

    let conv_rate = |errors: &[f64], curr: f64| {
      errors
        .last()
        .map(|&prev| algebraic_convergence_rate(curr, prev))
        .unwrap_or(f64::INFINITY)
    };
    let conv_rate = conv_rate(&errors, l2_norm);
    errors.push(l2_norm);

    println!("refinement={refinement} | L2_error={l2_norm:<7.2e} | conv_rate={conv_rate:>5.2}");
  }

  Ok(())
}
```


The output is
```
refinement=0 | L2_error=7.89e-1 | conv_rate=  inf
refinement=1 | L2_error=1.21e0  | conv_rate=-0.62
refinement=2 | L2_error=4.33e-1 | conv_rate= 1.49
refinement=3 | L2_error=1.24e-1 | conv_rate= 1.81
refinement=4 | L2_error=3.20e-2 | conv_rate= 1.95
refinement=5 | L2_error=8.12e-3 | conv_rate= 1.98
refinement=6 | L2_error=2.20e-3 | conv_rate= 1.88
refinement=7 | L2_error=8.21e-4 | conv_rate= 1.42
```

So almost order $alpha=2$ $L^2$ convergence, which is exactly what
theory predicts, confirming the correct implementation.



// Probably move this to post-face
= Conclusion and Outlook

- Summary of key contributions
- Possible improvements and future work (e.g., efficiency, higher-order elements, more general manifolds)
- Broader impact (e.g., Rust in scientific computing, FEEC extensions)
- Discarded ideas and failed apporaches (generic dimensionality à la nalgebra/eigen)

A far more meaningful PDE system that has some really interesting applications in real-life
are Maxwell's Equation describing Electromagnetism.
FEEC is the perfect fit for Maxwell's equations, since the relativistic variant of them
is also formulated in terms of differential geometry as is general relativity.
This means that purely thanks to the generality of the library we are able to solve
Maxwell's equations on the curved 4D spacetime manifold.

