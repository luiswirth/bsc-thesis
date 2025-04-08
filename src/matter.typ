#import "setup-math.typ": *
#import "layout.typ": *
#import "setup.typ": *

= Software Design & Implementation Choices

In this chapter we want to briefly discuss some general software engineering
decisions for our library.

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
  destructing of composite types like enums, this leads to concise and readable
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

- *Cargo* is Rust's official package manager and build system, which is one of
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


=== Challenges

We want to also mentioned some drawbacks of using Rust and challenges we've
encoutered.

- Rust is a relatively young programming language, as it had it's 1.0 release
in 2015. Due to this the library ecosystem is still evolving and solutions
that are available in `C++`, do not yet exist for Rust. A particular instance
that affects us, is the absence of sophisticated sparse linear algebra implementation.
Only basic sparse matrix implementation are available, but for solvers, we
we're forced to rely on `C/C++` libraries.

- Rust has a high learning curve and has a non-standard syntax
  with many concepts, that might make it hard for people unfamiliar with the language
  to read and understand it.

- One can become too obsessed with expressing concepts in the powerful type system,
  leading to over-engineering, which badly influences the project.
- Rust can become very verbose due to it's many abstraction features.


== External libraries

We want to quickly discuss here the major external libraries,
we will be using in our project.

=== nalgebra (linear algebra)

For implementing numerical algorithms linear algebra libraries are invaluable.
`C++` has set a high standard with `Eigen` as it's major linear algebra library.
Rust offers a very direct equivalent called `nalgebra`, which just as Eigen
relies heavily on generics to represent both statically and dynamically know
matrix dimensions.
All basic matrix and vector operations are available.
We will be using nalgebra all over the place, pretty much always we have to deal
with numerical values.

Sparse matrices will also be relevant in our library.
For this we will be using `nalgebra-sparse`.


=== PETSc & SLEPc (solvers)

Unfortunately the rust sparse linear algebra ecosystem is rather immature.
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
- common
- manifold
- exterior
- whitney
- formoniq

All of which have been published to `crates.io`.

===  Type safety

The implementation has a big emphasis on providing safety through the introduction
of many types that uphold guarantees regarding the contained data.
Constructed instances of types should always be valid.

=== Performance considerations

All data structures are written with performance in mind.

We are also always focused on a memory-economic representation of information.



= Topology & Geometry of\ Simplicial Riemannian Manifolds

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
    quad lambda^i >= 0,
    quad sum_(i=0)^n lambda^i = 1
  }
$ <def-simplex>

We call such an object a *coordinate simplex*, since it depends on global coordinates of
the vertices and lives in a possible higher-dimensional ambient space $RR^N$. It therefore
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
A special and particularly simple case is when intrinsic dimension and ambient dimension
match up $n=N$.
```rust
pub type Dim = usize;
pub fn dim_intrinsic(&self) -> Dim { self.nvertices() - 1 }
pub fn dim_embedded(&self) -> Dim { self.vertices.nrows() }
pub fn is_same_dim(&self) -> bool { self.dim_intrinsic() == self.dim_embedded() }
```

#v(0.5cm)
=== Barycentric Coordinates

The coefficients $avec(lambda) = [lambda^i]_(i=0)^n$ in @def-simplex are called
*barycentric coordinates*.
They appear inside the convex combination / weighted average $sum_(i=0)^n lambda^i avec(v)_i$
as weights $lambda^i in [0,1]$ with condition $sum_(i=0)^n lambda^i = 1$ in front of
each Cartesian vertex coordinate $avec(v)_i in RR^N$.
They constitute an intrinsic local coordinate representation with respect to the simplex $sigma in RR^N$,
which is independent from the embedding in $RR^N$ and only relies on the convex combination
of vertices. \
The coordinate transformation $psi: avec(lambda) |-> avec(x)$ from intrinsic
barycentric $avec(lambda) in RR^n$ to ambient Cartesian $avec(x) in RR^N$ coordinate is
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
Outside the simplex $avec(x) in.not sigma$, $lambda^i$ will be greater than one
or negative. \
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
Then the reduced barycentric coordinates $avec(lambda)^- = [lambda^i]_(i=1)^n$
constitutes a proper coordinate system without any redundant information.
We also call this the *local coordinate system*.
This coordinate system also works for the whole affine subspace, but now
the coordinates are completely free and there is a bijection between
the affine subspace and the whole of $RR^n$. There is a unique representation
for both ways.

#v(0.5cm)
=== Spanning Vectors

If we consider the edges emanating from the base vertex, we get the
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
  = avec(v)_0 + amat(E) avec(lambda)^-
$
This makes it very apparent that this transformation is an affine map, with
translation by $avec(v)_0$ and linear map $avec(lambda)^- |-> amat(E) vvec(lambda)^-$, between
the local coordinates $avec(lambda)^-$ and the Cartesian coordinates $avec(x)$ of the
affine subspace spanned up by the spanning vectors positioned at the base
vertex.

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

Reversing the transformation, is more subtle, as due to floating point inaccuracies,
we almost never exactly lie in the affine subspace.
Furthermore, when the ambient dimension $N$ is greater than the intrinsic dimension $n < N$,
then we have a underdetermined linear system.
Therefore we rely on the Moore-Penrose pseudo-inverse, computed via SVD, to get
a least-square solution.
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
The Jacobian of the affine map is exactly the linear map represented by $amat(E)$.
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

Furthermore these spanning vectors define a parallelepiped.
This parallelepiped can be used to compute the volume of the simplex, as a
fraction $(n!)^(-1)$ of the volume of the parallelepiped, which is computed as
the determinant of the spanning vectors in the case $n=N$ and otherwise using
the square root of the Gramian determinant
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

pub enum Sign {
  Pos = +1,
  Neg = -1,
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
local coordinates (reduced barycentric coordinates) also as global coordinates.
$
  sigma_"ref"^n = {(lambda_1,dots,lambda_n) in RR^n mid(|) lambda_i >= 0, quad sum_(i=1)^n lambda_i <= 1 }
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
The reference simplex acts as a chart of each real simplex $sigma$. While $psi:
sigma -> sigma_"ref"^n subset.eq RR^n$ is the chart map.

The barycentric coordinate functions are since they are intrinsic affine-invariant.


== Abstract Simplicies

After studying coordinate simplicies, the reader has hopefully developed
some intuitive understanding of simplicies. We will now shed the coordinates
and represent simplicies in a more abstract way, by just considering
them as a list of vertex indices, without any vertex coordinates.
A $n$-simplex $sigma$ is a $(n+1)$-tuple of natural numbers, which represent vertex
indices.
$
  sigma = [v_0,dots,v_n] in NN^(n+1)
  quad quad
  v_i in NN
$

In Rust we can simply represent this as the following struct.
```rust
pub type VertexIdx = usize;
pub struct Simplex {
  pub vertices: Vec<VertexIdx>,
}
impl Simplex {
  pub fn new(vertices: Vec<VertexIdx>) -> Self { Self { vertices } }
  pub fn standard(dim: Dim) -> Self { Self::new((0..dim + 1).collect()) }
  pub fn single(v: usize) -> Self { Self::new(vec![v]) }

  pub fn nvertices(&self) -> usize { self.vertices.len() }
  pub fn dim(&self) -> Dim { self.nvertices() - 1 }
}
```

The ordering of the vertices _does_ matter, therefore we really have ordered tuples
and not just unordered sets. This makes our simplicies combinatorial objects and
these combinatorics will be heart of our mesh data structure.

=== Sorted Simplicies

Even though order does matter, simplicies that share the same vertices,
are still pretty much the same. For this reason it is helpful,
to introduce a convention for the canonical representation of a
simplex given a set of vertices.

Our canonical representative will be the tuple, which as it's vertex indices
sorted increasingly. We can take any simplex in arbitrary order and
convert it to it's canonical representation.
```rust
pub fn is_sorted(&self) -> bool { self.vertices.is_sorted() }
pub fn sort(&mut self) { self.vertices.sort_unstable() }
pub fn sorted(mut self) -> Self { self.sort(); self }
```

Using this canonical representation, we can easily check
whether we two simplicies have the same vertex set, meaning
they are permutations of each other.
```rust
pub fn set_eq(&self, other: &Self) -> bool {
  self.clone().sorted() == other.clone().sorted()
}
pub fn is_permutation_of(&self, other: &Self) -> bool {
  self.set_eq(other)
}
```


=== Orientation

For our coordinate simplicies, we have seen that there are always
two orientations that a simplex can have.
We computed the orientation based on the determinant of the spanning vectors,
but without any coordinates this is no longer possible.

However we can still have a notion of relative orientation.
We have seen that with coordinate simplicies that swapping of two vertices
flips the orientation, due to the properties of the determinant.
The same behavior is present in abstract simplicies, based on the
ordering of the vertices.
All permutations can be divided into two equivalence classes.
Given a reference ordering of the vertices of a simplex,
we can call these even and odd permutations.
We call simplicies with even permutations, positively oriented and
simplicies with odd permutations negatively oriented.
Therefore every abstract simplex has exactly two orientations, positive
and negative depending on the ordering of vertices.

We use as reference ordering our canonical sorted representation.
We can determine the orientation relative to this sorted permutation,
by counting the number of swaps necessary to sort the simplex.
For this we implement a basic bubble sort, that keeps track of the number of swaps.
```rust
pub fn orientation_rel_sorted(&self) -> Sign { self.clone().sort_signed() }
pub fn sort_signed(&mut self) -> Sign { sort_signed(&mut self.vertices) }

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


Two simplicies that are made up of the same vertices, have equal orientation iff
their two permutations fall into the same (even or odd) permutation equivalence
class. Using the transitivity of this equivalence relation, we can do the check
relative to the sorted permutation.
```rust
pub fn orientation_eq(&self, other: &Self) -> bool {
  self.orientation_rel_sorted() == other.orientation_rel_sorted()
}
```

=== Subsets

Another important notion is the idea of a subsimplex or a face of a simplex.

A subsimplex is just a subset of a simplex.

We can easily check if a simplex is a subset of another simplex,
by using the subset relation definition.
$A subset.eq B <=> (forall a in A => a in B)$
```rust
pub fn is_subset_of(&self, other: &Self) -> bool {
  self.iter().all(|v| other.vertices.contains(v))
}
pub fn is_superset_of(&self, other: &Self) -> bool {
  other.is_subset_of(self)
}
```

We can generate all subsets, for which we rely on a
Itertools implementation.
```rust
pub fn subsets(&self, sub_dim: Dim) -> impl Iterator<Item = Self> {
  itertools::Itertools::permutations(self.clone().into_iter(), sub_dim + 1).map(Self::from)
}
```

The number of subsimplicies is given by the binomial coefficient.
```rust
pub fn nsubsimplicies(dim_cell: Dim, dim_sub: Dim) -> usize {
  binomial(dim_cell + 1, dim_sub + 1)
}
```


```rust
/// Computes local vertex numbers relative to sup.
pub fn relative_to(&self, sup: &Self) -> Simplex {
  let local = self
    .iter()
    .map(|iglobal| {
      sup
        .iter()
        .position(|iother| iglobal == iother)
        .expect("Not a subset.")
    })
    .collect();
  Simplex::new(local)
}
```


=== Substrings

When considering the facets of an $n$-simplex, there are multiple
permutations with the same set of vertices. It would be nice
to instead have only one permutation per subset of vertices.
For this we can instead consider the substrings of the original simplex.
This then also preservers the vertex order.

We have some methods to check whether a simplex is a substring of another, based
on a naive substring check algorithm.
```rust
pub fn is_substring_of(&self, other: &Self) -> bool {
  let sub = self.clone().sorted();
  let sup = other.clone().sorted();
  sup
    .vertices
    .windows(self.nvertices())
    .any(|w| w == sub.vertices)
}
pub fn is_superstring_of(&self, other: &Self) -> bool {
  other.is_substring_of(other)
}
```

We also have a method for generating all $k$-subsimplicies that
are substrings of a $n$-simplex. For this we generate all $k+1$-substrings
of the original $n+1$ vertices.
We use here the implementation of provided by the itertools crate.
```rust
pub fn substrings(&self, sub_dim: Dim) -> impl Iterator<Item = Self> {
  itertools::Itertools::combinations(self.clone().into_iter(), sub_dim + 1).map(Self::from)
}
```
This implementation is nice, since it provides the substrings in a lexicographical
order w.r.t. the local indices.
If the original simplex was sorted, then the substrings are truly lexicographically ordered even
w.r.t. the global indices.

A very standard operation is to generate all substrings simplicies of the standard simplex.
We call these the standard subsimplicies.
```rust
pub fn standard_subsimps(dim_cell: Dim, dim_sub: Dim) -> impl Iterator<Item = Simplex> {
  Simplex::standard(dim_cell).substrings(dim_sub)
}
```

We can also generate all the various standard subsimplicies for each standard simplex
dimension in a graded fashion. Something we will be using for generating the standard
simplicial complex.
```rust
pub fn graded_subsimps(dim_cell: Dim) -> impl Iterator<Item = impl Iterator<Item = Simplex>> {
  (0..=dim_cell).map(move |d| standard_subsimps(dim_cell, d))
}
```

We can also go the other directions and generate the superstrings of a given simplex,
if we are given a root simplex that has both the original simplex and it's substrings as
substrings.
```rust
pub fn superstrings(&self, super_dim: Dim, root: &Self) -> impl Iterator<Item = Self> + use<'_> {
  root
    .substrings(super_dim)
    .filter(|sup| self.is_substring_of(sup))
}
```

=== Boundary

There is a special operation related to substring simplicies, called the
boundary operator.
The boundary operator can be applied to any $n$-simplex $sigma in NN^(n+1)$ and
is the defined as.
$
  diff sigma = sum_i (-1)^i [v_0,dots,hat(v)_i,dots,v_n]
$
On the left we have a formal sum of simplicies.
This formal sum consists of all the $(n-1)$-substrings of a $n$-simplex,
together with a sign, giving the boundary simplicies a meaningful orientation.

When understanding this formal linear combinations as an element of the
free Abelian group generated by the basis of all simplicies,
then this operator is linear.

For instance, the boundary of the triangle $sigma = [0,1,2]$ is
$
  diff sigma = [1,2] - [0,2] + [0,1] = [0,1] + [1,2] + [2,0]
$
which is exactly what you get if you walk along the edges of the triangle.

We introduce an additional convention here regarding the ordering of the boundary simplicies.
We rely on the `subsimps` implementation that gives us a lexicographically ordered
subsimplicies. This is exactly the opposite of the ordered suggested of the sum sign.
We need to make sure the sign is still the same for all boundary simplicies.
```rust
pub fn boundary(&self) -> impl Iterator<Item = SignedSimplex> {
  let mut sign = Sign::from_parity(self.nvertices() - 1);
  self.substrings(self.dim() - 1).map(move |simp| {
    let this_sign = sign;
    sign.flip();
    SignedSimplex::new(simp, this_sign)
  })
}
```


We can add an extra formal sign symbol to the simplex to obtain
a signed simplex.
```rust
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct SignedSimplex {
  pub simplex: Simplex,
  pub sign: Sign,
}
```

== Simplicial Skeleton

Simplicies are the building blocks of our mesh.
If we build our mesh from coordinate simplicies, then we have the typical euclidean
extrinsic description of an embedded mesh, that contains all geometric information. 
As an embedding the union of these coordinate simplicies really is a
$n$-dimensional region of $N$-dimensional of euclidean space.

If we, in contrast, build our mesh only from abstract simplicies, then
we are missing this full geometric information.
since, abstract simplicies only specify the vertices (as indices) that they are made of.
This information however fully defines the topology of our discrete
$n$-manifold.
If two simplicies share the same vertices, then these are connected, by either
being adjacent or by being incident.
This makes the topology purely combinatorial.

In the following sections we study this simplicial topology and implement data
structures and algorithms related to it.

#v(1cm)

To define the topology of our simplicial $n$-manifold, we just need
to store the $n$-simplicies that make it up.
This defines the topology of the mesh at the top-level.
We call such a collection of $n$-simplicies that share the same vertices a $n$-skeleton.

```rust
/// A container for sorted simplicies of the same dimension.
#[derive(Default, Debug, Clone)]
pub struct Skeleton {
  /// Every simplex is sorted.
  simplicies: IndexSet<Simplex>,
  nvertices: usize,
}
```

A skeleton takes care of various responsibilities of a mesh data structure.
It is a container for all $n$-simplicies.
It allows for the iteration of all these mesh entities.
```rust
impl Skeleton {
  pub fn iter(&self) -> indexmap::set::Iter<'_, Simplex> {
    self.simplicies.iter()
  }
}
impl IntoIterator for Skeleton {
  type Item = Simplex;
  type IntoIter = indexmap::set::IntoIter<Self::Item>;
  fn into_iter(self) -> Self::IntoIter {
    self.simplicies.into_iter()
  }
}
```
In provides unique identification of the simplicies
through a global numbering.
This is a bijective mapping between the simplex index
and the abstract simplex itself. This represented through this `IndexSet` data structure
from the `index-map` crate. It has the typical `Vec` functionality for retrieving
a `Simplex` from it's `SimplexIdx`, but it also supports the reverse direction (through hashing)
of retrieving the index from the `Simplex` itself.
```rust
pub fn simplex_by_kidx(&self, idx: KSimplexIdx) -> &Simplex {
  self.simplicies.get_index(idx).unwrap()
}
pub fn kidx_by_simplex(&self, simp: &Simplex) -> KSimplexIdx {
  self.simplicies.get_index_of(simp).unwrap()
}
```


The skeleton constructor upholds various guarantees about the simplicies
that are contained in it.
```rust
pub fn new(simplicies: Vec<Simplex>) -> Self {
  assert!(!simplicies.is_empty(), "Skeleton must not be empty");
  let dim = simplicies[0].dim();
  assert!(
    simplicies.iter().map(|simp| simp.dim()).all(|d| d == dim),
    "Skeleton simplicies must have same dimension."
  );
  assert!(
    simplicies.iter().all(|simp| simp.is_sorted()),
    "Skeleton simplicies must be sorted."
  );
  let nvertices = if dim == 0 {
    assert!(simplicies.iter().enumerate().all(|(i, simp)| simp[0] == i));
    simplicies.len()
  } else {
    simplicies
      .iter()
      .map(|simp| simp.iter().max().expect("Simplex is not empty."))
      .max()
      .expect("Simplicies is not empty.")
      + 1
  };

  let simplicies = IndexSet::from_iter(simplicies);
  Self {
    simplicies,
    nvertices,
  }
}
```
First of all, the skeleton cannot be empty and all simplicies must be of the
same dimension.
Furthermore we want to only store canonical representation of simplicies, since
only then the reverse mapping from simplex to index is useful, because independent
of the current vertex ordering we can always convert to the canonical representation
to get the index.
Lastly we have a special requirement on a 0-skeleton, because there the
simplicies are exactly just the vertices and we want them sorted.


== Simplicial Complex

A $n$-skeleton alone doesn't suffice as data structure for our FEEC implementation,
since it is missing the topology of the lower-dimensional subsimplicies of our cells.
But our FE basis functions are associated with these subsimplicies, so we need to represent them.

The skeleton only stores the top-level simplicies $Delta_n (mesh)$, but our FEM library
also needs to reference the lower-level simplicies $Delta_k (mesh)$, since these are also
also mesh entities on which the DOFs of our FE space live.


Enter the simplicial complex. It stores not only the top-level cells, but also all
$k$-subsimplicies with $0 <= k <= n$.
So a simplicial $n$-complex is made up of $n+1$ skeletons of dimensions $0,dots,n$.

Some useful terminology is
- The $0$-simplicies are called vertices.
- The $1$-simplicies are called edges.
- The $2$-simplicies are called faces.
- The $3$-simplicies are called tets.
- The $(n-1)$-simplicies are called facets.
- The $n$-simplicies are called cells.

It will be the main topological data structure that we will, pass as argument
into all FEEC algorithm routines.

In general a simplicial complex need not have manifold topology, since it can
represent more general topological spaces beyond manifolds.
Our PDE framework however needs domains to be manifold.
For this reason we will restrict our data structure to this. As a consequence
our simplicial complex will be pure, meaning every $k$-subsimplex is contained in at
least one cell.
This will be ensured by the fact, that we will generate our simplicial complex
from a cell-skeleton and all possible subset simplicies will be present.

However the skeleton itself, might not encode a manifold topology. This would be the case
if in 2D, we would have more than two triangles meeting at a single edge. Then we don't
have a surface, but some non-manifold topological space.
In general the rule is that at each facet has at most 2 cocells. A property we
will be checking when building the complex.

For a simplicial complex to be manifold, the neighborhood of each vertex (i.e. the
set of simplices that contain that point as a vertex) needs to be homeomorphic
to a $n$-ball.


```rust
/// A simplicial manifold complex.
#[derive(Default, Debug, Clone)]
pub struct Complex {
  skeletons: Vec<(Skeleton, SkeletonData)>,
}
impl Complex {
  pub fn dim(&self) -> Dim { self.skeletons.len() - 1 }
}
```

One of the main algorithms is to construct a simplicial complex from a top-level
cell-skeleton. For this we generate the substrings of all lengths of the cells.
While constructing we also precompute and store certain topological properties,
such as in which cells the subsimplex is contained.
Afterwards we do some topology checks, such as verifying that the topology of
the given skeleton was actually manifold.
```rust
pub fn from_cells(cells: Skeleton) -> Self {
  let dim = cells.dim();

  let mut skeletons = vec![(Skeleton::default(), SkeletonData::default()); dim + 1];
  skeletons[0].0 = Skeleton::new((0..cells.nvertices()).map(Simplex::single).collect());
  skeletons[0].1 = SkeletonData(
    (0..cells.nvertices())
      .map(|_| SimplexData::default())
      .collect(),
  );

  for (icell, cell) in cells.iter().enumerate() {
    for (dim_sub, (sub_skeleton, sub_skeleton_data)) in skeletons.iter_mut().enumerate() {
      for sub in cell.substrings(dim_sub) {
        let (sub_idx, is_new) = sub_skeleton.insert(sub);
        let sub_data = if is_new {
          sub_skeleton_data.0.push(SimplexData::default());
          sub_skeleton_data.0.last_mut().unwrap()
        } else {
          &mut sub_skeleton_data.0[sub_idx]
        };
        sub_data.cocells.push(SimplexIdx::new(dim, icell));
      }
    }
  }

  // Topology checks.
  if dim >= 1 {
    let facet_data = &skeletons[dim - 1].1;
    for SimplexData { cocells } in &facet_data.0 {
      let nparents = cocells.len();
      let is_manifold = nparents == 2 || nparents == 1;
      assert!(is_manifold, "Topology must be manifold.");
    }
  }

  Self { skeletons }
}
```

=== Boundary Operator

We have already seen the boundary operator for single simplicies.
We can extend this operator to the whole skeleton.
Here the boundary operator itself is returned as a linear operator
from the $k$-skeleton to the $k-1$-skeleton.
It is the signed incidence matrix of the simplicies in the upper skeleton
in the lower skeleton.

$
  amat(D) in {-1,0,+1}^(N_(k-1) times N_k)
$


```rust
/// $diff^k: Delta_k -> Delta_(k-1)$
pub fn boundary_operator(&self, dim: Dim) -> SparseMatrix {
  let sups = &self.skeleton(dim);

  if dim == 0 {
    return SparseMatrix::zeros(0, sups.len());
  }

  let subs = &self.skeleton(dim - 1);
  let mut mat = SparseMatrix::zeros(subs.len(), sups.len());
  for (isup, sup) in sups.handle_iter().enumerate() {
    let sup_boundary = sup.simplex_set().boundary();
    for sub in sup_boundary {
      let sign = sub.sign.as_f64();
      let isub = subs.get_by_simplex(&sub.simplex).kidx();
      mat.push(isub, isup, sign);
    }
  }
  mat
}
```

=== Simplicies in the Mesh: Simplex Indices and Handles

To identify a simplex inside the mesh, we use an indexing system.
If the context of a concrete dimension is given,
than we only need to know the index inside the skeleton, which
is just an integer
```rust
pub type KSimplexIdx = usize;
```
If the skeleton context is not given, then we also need to specify
the dimension, for this we have a fat index, that contains both parts.
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SimplexIdx {
  pub dim: Dim,
  pub kidx: KSimplexIdx,
}
```
Using this we reference any simplex in the mesh. Since we are using a `indexmap::IndexSet`
data structure for storing our simplicies inside a skeleton,
we are able to go both ways. We can not only retrieve the simplex, given the index,
but we can also get the index corresponding to a given simplex.

The `Simplex` struct doesn't reference the mesh and therefore doesn't have access
to any other simplicies. But for doing any kind of topological computations,
it is helpful to be able to reference other simplicies in the mesh.
For this reason we introduce a new concept, that represents a simplex
inside of a mesh. We create a simplex handle, that is like a more sophisticated
pointer to a simplex, that has a reference to the mesh.
This allows us to interact with these simplicies inside the mesh very naturally.

```rust
#[derive(Copy, Clone)]
pub struct SimplexHandle<'c> {
  complex: &'c Complex,
  idx: SimplexIdx,
}
```
Pretty much all functionality that was available on the raw `Simplex` struct,
is also available on the handle, but here we directly provide any relevant
mesh information.

For instance the `SimplexHandle::supersimps` gives us all supersimplicies that
are also contained in the mesh, which are exactly all the superstring simplicies.
The `Simplex::superstring` method however expects a `root: &Simplex`, which
gives the context in which we are searching for superstrings. This context
is directly provided to the method on the handle, since the mesh knows
the cocells of each simplex, which is here chosen as the root, so we
don't need to provide this argument ourselves.
```rust
  pub fn supersimps(&self, dim_super: Dim) -> Vec<SimplexHandle> {
    self
      .cocells()
      .flat_map(|parent| {
        self
          .raw()
          .superstrings(dim_super, parent.raw())
          .map(move |sup| self.complex.skeleton(dim_super).get_by_simplex(&sup))
      })
      .collect()
  }
```
Furthermore these functions always directly access the `IndexSet` and retrieve
the corresponding index of the simplex and construct a new `SimplexHandle` out of
it, such that we can easily apply subsequent method calls on the returned objects.


== Simplicial Geometry

We have now successfully developed the topological structure of our mesh,
by combining many abstract simplicies into skeletons and collecting
all of these skeletons together.

What we are still missing in our mesh data structure now, is any geometry.
The geometry is missing, since we only store abstract simplicies and not
something like coordinate simplicies.

This was purposefully done, because we want to separate the topology from the geometry.
This allows us to switch between a coordinate-based embedded geometry and a coordinate-free
intrinsic geometry based on a Riemannian metric.


== Coordinate-Based Ambient Euclidean Geometry

Let us first quickly look at the familiar coordinate-based euclidean geometry,
that relies on an embedding. It's an extrinsic description of the geometry
of the manifold from the perspective of the ambient space.
All we need is to know the coordinates of all vertices in the mesh.
```rust
#[derive(Debug, Clone)]
pub struct MeshVertexCoords {
  coord_matrix: na::DMatrix<f64>,
}
impl MeshVertexCoords {
  pub fn dim(&self) -> Dim { self.coord_matrix.nrows() }
  pub fn nvertices(&self) -> usize { self.coord_matrix.ncols() }
  pub fn coord(&self, ivertex: VertexIdx) -> CoordRef { self.coord_matrix.column(ivertex) }
}
```
We once again store the coordinates of the vertices in the column of a matrix,
just as we did for the `SimplexCoords` struct, but here we store the coordinates
of all the vertices in the mesh, so this is usually a really wide matrix.
The `dim` function here references the dimension of the ambient space. It
is different from the topology dimension in general.

Here we witness another benefit of separating topology and geometry, which should be done
even when there are not multiple geometry representations supported:
We avoid any redundancy in storing the vertex coordinates. For every vertex we
store it's coordinate exactly once. This is contrast to use a list of `SimplexCoords`, for
which there would have been many duplicate coordinates, since the vertices are
shared by many simplicies. So separating topology and geometry is always very natural
even in the case of the typical coordinate-based geometry.

=== Coordinate Function Functors & Barycentric Quadrature

Before differential geometry, calculus was done on euclidean space $RR^n$
instead of on abstract manifolds. Euclidean space always has global coordinates.
A point $avec(p) in RR^n$ is exactly it's own global coordinate $avec(x) = avec(p)$.
This means that functions $f: p in Omega |-> f(p) in RR$ that take a point $avec(p)$
of the space $Omega$ to a real number $f(avec(p))$, are usually specified by an evaluation rule
$f: avec(x) in Omega |-> f(avec(x)) in RR$ based on coordinates, such as for example
$f(avec(x)) = sin(x_1)$.
This is a very useful and general representation of a function that
relies solely on point evaluation and is very common in numerical codes.
In programming languages this type of object is referred to as a functor.
Functors are types that provide an evaluation operator.

On manifolds the story is a little different. They admit no global coordinates
in general. But instead we can rely on ambient coordinates $x in RR^N$, if an
embedding is available, and work with functions defined on them.

One common use-case that is also relevant to us for a such a point-evaluable
functor is numerical integration of a real valued function via numerical
quadrature.
Since we are doing only 1st order FEEC, we restrict ourselves to
quadrature rules of order 1, that integrate affine-linear functions exactly.
The simplest of these that work on arbitrary-dimensional simplicies is
the barycentric quadrature rule, that just does a single evaluation
of the function at the barycenter of the simplex and multiples this
value by the volume of the simplex, giving us a approximation of the
integral.
$
  integral_sigma f vol approx |sigma| f(avec(m)_sigma)
$

We implement a simple routine in Rust that does exactly.
```rust
pub fn barycentric_quadrature<F>(f: &F, simplex: &SimplexCoords) -> f64
where
  F: Fn(CoordRef) -> f64,
{
  simplex.vol() * f(simplex.barycenter().as_view())
}
```

Here `F` is a generic type, that implements the `Fn` trait, that
supplies an evaluation operator, turning `F` into a functor.
This trait is most prominently implemented by closures (analogs of lambdas in `C++`).

This scalar quadrature can then be used for integrating
coordinate differential form closures, since after evaluating
the differential form on the tangent vectors, we obtain a simple
scalar function.
$
  avec(x) in RR^N |-> omega_avec(x) (diff_1,dots,diff_n) in RR
$


== Coordinate-Free Intrinsic Riemannian Geometry

The coordinate-based Euclidean geometry we've seen so far, is what
is commonly used in almost all FEM implementations.
In our implementation we go one step further and abstract away
the coordinates of the manifold and instead make use of 
coordinate-free Riemannian geometry for all of our FE algorithms.
All FE algorithms only depend on this geometry representation and cannot operate
directly on the coordinate-based geometry. Instead one should always derive a
coordinate-free representation from the coordinate-based one.
Most of the time one starts with a coordinate-based representation
that has been constructed by some mesh generator like gmsh and
then one computes the intrinsic geometry and forgets about the coordinates.
Our library supports this exactly this functionality.

=== Riemannian Metric

Riemannian geometry is an intrinsic description of the manifold,
that doesn't need an ambient space at all. It relies purely on a structure
over the manifold called a *Riemannian metric* $g$.

It is a continuous function over the whole manifold, which at each point $p$
gives us an inner product $g_p: T_p M times T_p M -> RR^+$ on the tangent space
$T_p M$ at this point $p$.
It is the analog to the standard euclidean inner product (dot product) in
euclidean geometry. The inner product on tangent vectors allows one to measure
lengths $norm(v)_g = sqrt(g(v, v))$ angles $phi(v, w) = arccos((g_p (v, w))/(norm(v)_g norm(w)_g))$.
While euclidean space is flat and the inner product is the same everywhere, a
manifold is curved in general and therefore the inner product changes from point
to point, reflecting the changing geometry.


Given a basis $diff/(diff x^1),dots,diff/(diff x^n)$ of the tangent space
$T_p M$ at a point $p$, induced by a chart map
$phi: p in U subset.eq M |-> (x_1,dots,x_n)$, the inner product $g_p$ can be
represented in this basis using components $(g_p)_(i j) in RR$.
This is done by plugging in all combinations of basis vectors into the two
arguments of the bilinear form.
$
  (g_p)_(i j) = g_p (restr(diff/(diff x^i))_p,restr(diff/(diff x^j))_p)
$

We can collect all these components into a matrix $amat(G) in RR^(n times n)$
$
  amat(G) = [g_(i j)]_(i,j=1)^(n times n)
$

This is called a Gram matrix or Gramian and is the discretization of a
inner product of a linear space, given a basis.
This matrix doesn't represent a linear map, which would be a $(1,1)$-tensor, but
instead a bilinear form, which is a $(0,2)$-tensor.
In the context of Riemannian geometry this is called a *metric tensor*.

The inverse metric $g^(-1)_p$ at a point $p$ provides an inner product
$g^(-1)_p: T^*_p M times T^*_p M -> RR^+$ on the
cotangent space $T^*_p M$. It can be obtained by computing the inverse
Gramian matrix $amat(G)^(-1)$, which is then a new Gramian matrix representing
the inner product on the dual basis of covectors.
$
  amat(G)^(-1) = [g(dif x^i,dif x^j)]_(i,j=1)^(n times n)
$
The inverse metric is very important for us, since differential forms are
covariant tensors, therefore they are measured by the inverse metric tensor.

Computing the inverse is numerically unstable and instead it would
be better to rely on matrix factorization to do computation
invovling the inverse metric. However this quickly becomes
intractable. For this reason we chose here to rely on the directly
computed inverse matrix nontheless.

We introduce a struct to represent the Riemannian metric at a particular point
as the Gramian matrix and inverse Gramian matrix.
```rust
#[derive(Debug, Clone)]
pub struct RiemannianMetric {
  metric_tensor: na::DMatrix<f64>,
  inverse_metric_tensor: na::DMatrix<f64>,
}
impl RiemannianMetric {
  pub fn dim(&self) -> Dim { self.metric_tensor.nrows() }
  pub fn inner(&self, i: usize, j: usize) -> f64 { self.metric_tensor[(i, j)] }
  pub fn length_sq(&self, i: usize) -> f64 { self.inner(i, i) }
  pub fn length(&self, i: usize) -> f64 { self.length_sq(i).sqrt() }
  pub fn angle_cos(&self, i: usize, j: usize) -> f64 {
    self.inner(i, j) / self.length(i) / self.length(j)
  }
  pub fn angle(&self, i: usize, j: usize) -> f64 { self.angle_cos(i, j).acos() }
```

The dot product is the standard inner product on flat Euclidean space.
The standard basis vectors are orthonormal w.r.t. this inner product.
Therefore the gram matrix (and it's inverse) are just identity matrices.
```rust
/// Orthonormal flat euclidean metric.
pub fn standard(dim: Dim) -> Self {
  let identity = na::DMatrix::identity(dim, dim);
  let metric_tensor = identity.clone();
  let inverse_metric_tensor = identity;
  Self { metric_tensor, inverse_metric_tensor, }
}
```

=== Deriving the Metric from an Immersion

One can easily derive the Riemannian metric from an immersion $f: M -> RR^N$
into an ambient space $RR^N$.
It's differential is a function $dif f_p: T_p M -> T_p RR^N$, also called
the push-forward and tells us how our intrinsic tangential vectors are being
placed into the ambient space, giving them an extrinsic geometry.

This immersion then induces a metric, that describes the same geometry
intrinsically. For this we just take the standard euclidean inner product
of our immersed tangent vectors. This then inherits the extrinsic ambient geometry
and represents it intrinsically.
$
  g_p (u, v) = dif f_p (u) dot dif f_p (v)
$

Computationally this differential $dif f_p$ can be represented, given a basis,
since it is a linear map, by a Jacobi Matrix $amat(J)$.
The metric is the then the Gramian matrix of the Jacobian.
$
  amat(G) = amat(J)^transp amat(J)
$

The Jacobian has as columns our immersed basis tangent vectors, therefore
really we just need these to compute a metric.
```rust
pub fn from_tangent_basis(basis: na::DMatrix<f64>) -> Self {
  let metric_tensor = basis.gramian();
  Self::new(metric_tensor)
}
impl DMatrixExt for na::DMatrix<f64> {
  fn gramian(&self) -> Self {
    self.transpose() * self
  }
  // ...omitted
}
```

=== Intrinsic Simplicial Geometry & Regge Metric

So far our discussion of Riemannian geometry hasn't referenced
our mesh. But we are of course doing geometry on a simplicial manifold.

We have seen with our coordinate simplicies that our geometry is piecewise-flat
over the cells. This means that our metric is constant over each cell
and changes only from one cell to another.

This piecewise-constant metric over the simplicial mesh is known as the *Regge
metric* and comes from Regge calculus, a theory for numerical general relativity
that is about producing simplicial approximations of spacetimes that are
solutions to the Einstein field equation.

Our coordinate simplicies are an immersion of an abstract simplex
and as such, we can compute the corresponding constant metric tensor on it.
The spanning vectors constitute a basis of the tangent vectors.
```rust
impl SimplexCoords {
  pub fn metric_tensor(&self) -> RiemannianMetric {
    RiemannianMetric::from_tangent_basis(self.spanning_vectors())
  }
}
```

But just like storing coordinate simplicies is a memory-inefficient representation
of the extrinsic geometry, storing the metric tensor on each cell
is also inefficient.
A global way to store the Regge metric is based on edge lengths. Instead
of giving all vertices a global coordinate, as one would do in extrinsic
geometry, we just give each edge in the mesh a positive length. Just knowing
the lengths doesn't tell you the positioning of the mesh in an ambient space
but it's enough to give the whole mesh it's piecewise-flat geometry.
Storing only the edge lengths of the whole mesh is a more memory efficient
representation of the geometry than storing all the metric tensors.

Mathematically this is just a function on the edges to the positive real numbers.
$
  l: Delta_1 (mesh) -> RR^+
$
that gives each edge $e in Delta_1 (mesh)$ a positive length $l_e in RR^+$.

Computationally we represent the edge lengths in a single struct
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

Our FE algorithms than usually take two arguments.
```rust
fn fe_algorithm(topology: &Complex, geometry: &MeshEdgeLengths)
```

We can derive a `MeshEdgeLengths` struct from a `MeshVertexCoords` struct.
```rust
impl MeshVertexCoords {
  pub fn to_edge_lengths(&self, topology: &Complex) -> MeshEdgeLengths {
    let edges = topology.edges();
    let mut edge_lengths = na::DVector::zeros(edges.len());
    for (iedge, edge) in edges.set_iter().enumerate() {
      let [vi, vj] = edge.clone().try_into().unwrap();
      let length = (self.coord(vj) - self.coord(vi)).norm();
      edge_lengths[iedge] = length;
    }
    MeshEdgeLengths::new(edge_lengths)
  }
}
```

We can then restrict these edge lengths to just a single simplex and obtain
a `SimplexEdgeLengths` struct. From this we can directly compute the metric
Gramian, from the law of cosines.
$
  amat(G)_(i j) = 1/2 (l_(0 i)^2 + l_(0 j)^2 - l_(i j)^2)
$

```rust
/// Builds regge metric tensor from edge lenghts of simplex.
pub fn compute_regge_metric(&self) -> RiemannianMetric {
  let dim = self.dim();
  let nvertices = dim + 1;
  let mut metric_tensor = na::DMatrix::zeros(dim, dim);
  for i in 0..dim {
    metric_tensor[(i, i)] = self[i].powi(2);
  }
  for i in 0..dim {
    for j in (i + 1)..dim {
      let l0i = self[i];
      let l0j = self[j];

      let vi = i + 1;
      let vj = j + 1;
      let eij = lex_rank(&[vi, vj], nvertices);
      let lij = self[eij];

      let val = 0.5 * (l0i.powi(2) + l0j.powi(2) - lij.powi(2));

      metric_tensor[(i, j)] = val;
      metric_tensor[(j, i)] = val;
    }
  }
  RiemannianMetric::new(metric_tensor)
}
```

== Higher Order Geometry

The original PDE problem, before discretization, is posed on a smooth manifold, which
we then discretized in the form of a mesh.
This smooth manifold has a non-zero curvature everywhere in general.
This is in contrast to the simplex geometry we've chosen here,
that approximates the geometry, by a *piecewise-flat geometry* over the cells.
Each cell is a flat simplex that has no curvature change inside of it.

This manifests as the fact, that for a coordinate-based representation of the
geometry, the cell is just the convex hull of the vertex coordinates.
And for the metric-based representation, we have a *piecewise-constant metric*
over each cell.

This is a piecewise-linear 1st order approximation of the geometry.
But this is not only possible approximation. Higher-order mesh elements,
such as quadratic or cubic elements allow for higher accuracy approximations
of the mesh. In general any order polynomial elements can be used.
We will restrict ourselves completely to first-order elements in this thesis.
This approximation is sufficient for us, since it represents an
*admissible geometric variational crime*: The order of our FE method coincides with
the order of our mesh geometry; both are linear 1st order.
This approximation doesn't affect the order of convergence in a negative way,
and therefore is admissible.


== Mesh Generation and Loading

=== Tensor-Product Domain Meshing

Formoniq features a meshing algorithm for arbitrary dimensional
tensor-product domains. These domains are $n$-dimensional Cartesian products $[0,1]^n$
of the unit interval $[0,1]$. The simplicial skeleton will be computed based on
a Cartesian grid that subdivides the domain into $l^n$ many $n$-cubes, which are
generalizations of squares and cubes. Here $l$ is the number of subdivisions per axis.
To obtain a simplicial skeleton, we need to split each $n$-cube into non-overlapping $n$-simplicies
that make up it's volume. In 2D it's very natural to split a square into two triangles
of equal volume. This can be generalized to higher dimensions. The trivial
triangulation of a $n$-cube into $n!$ simplicies is based on the $n!$ many permutations
of the $n$ coordinate axes.

The $n$-cube has $2^n$ vertices, which can all be identified using multiindices
$
  V = {0,1}^n = {(i_1,dots,i_n) mid(|) i_j in {0,1}}
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

We can note here, that the computational complexity of this algorithm, grows extremely fast
in the dimension $n$.
We have a factorial scaling $cal(O)(n!)$ (worse than exponential scaling $cal(O)(e^n)$)
for splitting the cube into simplicies. Given $l$ subdivisions per dimensions, we have
$l^n$ cubes. So the overall computational complexity is dominated by $cal(O)(l^n n!)$,
a terrible result, due to the curse of dimensionality.
The memory usage is dictated by the same scaling law.

=== Gmsh Import

The formoniq manifold crate can read gmsh `.msh` files and turn them
into a simplicial complex that we can work on.
This is thanks to the `gmshio` crate.

```rust
pub fn gmsh2coord_cells(bytes: &[u8]) -> (Skeleton, MeshVertexCoords)
```

=== Blender IO: OBJ and MDD

The formoniq manifold crate supports reading and writing of file formats related to blender,
for the easy visualization of the 2-manifold embedded in $RR^3$.

These two formats are OBJ and MDD. Thanks to the simplicity of these formats
we don't rely on any external libraries, but just have very basic
custom readers and writers for these.

=== Custom Format for Arbitrary Dimensions

We also have a maximally simple custom file format that works
great for arbitrary dimensional manifolds.

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

= Exterior Algebra

Exterior algebra is to exterior calculus, what vector algebra is to vector
calculus.\
In vector calculus we have vector fields $v$, which are functions
$v: p in Omega |-> v_p$ over the manifold $Omega$ that at each point
$p$ have a constant vector $v_p in T_p M$ as value.\
In exterior calculus we have differential forms $omega$, which are functions
$omega: p in Omega |-> omega_p$ over the manifold $Omega$ that at each point $p$
have a constant *multiform* $omega_p in wedgespace (T^*_p M)$ as value.

If one were to implement something related to vector calculus
it is of course crucial to be able to represent vectors in the program.
This is usually the job of a basic linear algebra library such as Eigen in `C++`
and nalgebra in Rust.\
Since we want to implement FEEC, which uses exterior calculus,
it is crucial, that we are able to represent multiforms in our program.
For this there aren't any established libraries. So we do this ourselves
and develop a small module.

== Exterior Algebra of Multiforms

In general an exterior algebra $wedgespace (V)$ is a construction over any linear
space $V$. In this section we want to quickly look at the specific linear
space we are dealing with when modelling multiforms as element of an
exterior algebra. But our implementation would work for any finite-dimensional
real linear space $V$ with a given basis.

In our particular case we have the exterior algebra of alternating multilinear
forms $wedgespace (T^*_p M)$. Here the linear space $V$ is the cotangent space
$T^*_p M$ of the manifold $M$ at a point $p in M$. It's the dual space $(T_p
M)^*$ of the tangent space $T_p M$.
The elements of the cotangent space are covectors $a in T^*_p M$, which are
linear functionals $a: T_p M -> RR$ on the tangent space.
The tangent space $T_p M$ has the standard basis ${diff/(diff x^1)}_(i=1)^n$
induced by some chart map $phi: p in U subset.eq M |-> (x_1,dots,x_n)$. This
gives rise to a dual basis ${dif x^i}_(i=1)^n$ of the cotangent space, defined
by $dif x^i (diff/(diff x^j)) = delta^i_j$.

There is a related space, called the space of multivectors $wedgespace (T_p M)$,
which is the exterior algebra over the tangent space, instead of the cotangent space.
The space of multivectors and multiforms are dual to each other.
$
  wedgespace (T^*_p M) =^~ (wedgespace (T_p M))^*
$
The space of multivectors only plays a minor role in exterior calculus, since it
is not metric independent. We just wanted to quickly mentioned it here.

It is common practice to call the elements of any exterior algebra
multivectors, irregardless what the underlying linear space $V$ is.
This is confusing when working with multiforms, which are distinct from multivectors.
To avoid confusion, we therefore just call the elements of the exterior algebra
exterior elements or multielements, just like we say linear space instead of
vector space.

== The Numerical Exterior Algebra $wedgespace (RR^n)$

When working with vectors from a finite-dimensional real linear space $V$, then
we can always represent them computationally, by choosing a basis
${e_i}_(i=1)^n subset.eq V$.
This constructs an isomorphism $V =^~ RR^n$, where $n = dim V$.
This allows us to work with elements $avec(v) in RR^n$, which have real
values $v_i in RR$ as components, which are the basis coefficients.
These real numbers are what we can work with on computers and allow
us to do numerical linear algebra.
This means that when working with any finite-dimensional real linear space $V$
on a computer we always just use the linear space $RR^n$.

The same idea can be used to computationally work with exterior algebras.
By choosing a basis of $V$, we also get an isomorphism on the exterior algebra
$wedgespace (V) =^~ wedgespace (RR^n)$.
Therefore our implementation will be we directly on $wedgespace (RR^n)$.

For our space of multiforms, we will be using the standard cotangent basis
${dif x^i}_(i=1)^n$.

== Representing Exterior Elements

An exterior algebra is a graded algebra.
$
  wedgespace (RR^n) = wedgespace^0 (RR^n) plus.circle.big dots.c plus.circle.big wedgespace^n (RR^n)
$
Each element $v in wedgespace (RR^n)$
has some particular exterior grade $k in {1,dots,n}$ and therefore lives in
a particular exterior power $v in wedgespace^k (RR^n)$.
We make use of this fact in our implementation, by splitting the representation
between these various grades.
```rust
pub type ExteriorGrade = usize;
```

For representing an element in a particular exterior power
$wedgespace^k (RR^n)$, we use the fact that, it itself is a linear space in it's
own right.
Due to the combinatorics of the anti-symmetric exterior algebra, we have $dim
wedgespace^k (RR^n) = binom(n,k)$.
This means that by choosing a basis ${e_I}$ of this exterior power, we can just
use a list of $binom(n,k)$ coefficients to represent an exterior element, by
using the isomorphism $wedgespace^k (RR^n) =^~ RR^binom(n,k)$.
```rust
/// An element of an exterior algebra.
#[derive(Debug, Clone)]
pub struct ExteriorElement {
  coeffs: na::DVector<f64>,
  dim: Dim,
  grade: ExteriorGrade,
}
```
This struct represents an element `self` $in wedgespace^k (RR^n)$ with
`self.dim` $= n$, `self.grade` $= k$ and `self.coeffs.len()` $= binom(n,k)$.

This exterior basis ${e_I}_(I in cal(I)^n_k)$ is different from the basis
${e_i}_(i=1)^n$ of the original linear space $V$, but is best subsequently
constructed from it.
We do this by creating elementary multielements from the
exterior product of basis elements.
$
  e_I = wedge.big_(j=1)^k e_I_j = e_i_1 wedge dots.c wedge e_i_k
$
Here $I = (i_1,dots,i_k)$ is a multiindex, in particular a $k$-index, telling
us which basis elements to wedge.

Because of the anti-symmetry of the exterior product, there are certain conditions
on the multiindices $I$ for ${e_I}$ to constitute a meaningful basis.
First $I$ must not contain any duplicate indices, because otherwise $e_I = 0$
and second there must not be any permutations of the same index in the
basis set, otherwise we have linear dependence of the two elements.
We therefore only consider strictly increasing multiindices $I in cal(I)^n_k$
and denote their set by
$cal(I)^n_k = {(i_1,dots,i_k) in NN^k mid(|) 1 <= i_1 < dots.c < i_k <= n}$.
This is a good convention for supporting arbitrary dimensions.

The basis also needs to be ordered, such that we can know which coefficient
in `self.coeffs` corresponds to which basis. A natural choice here is
a lexicographical ordering.

Taking in all of this together we for example have as exterior basis for
$wedge.big^2 (RR^3)$ the elements $e_1 wedge e_2, e_1 wedge e_3, e_2 wedge e_3$.

== Representing Exterior Terms

It is helpful to represent these exterior basis wedges in our program.
```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExteriorTerm {
  indices: Vec<usize>,
  dim: Dim,
}
impl ExteriorTerm {
  pub fn dim(&self) -> Dim { self.dim }
  pub fn grade(&self) -> ExteriorGrade { self.indices.len() }
}
```
This struct allows for any multiindex, even if they are not strictly increasing.
But we are of course able to check whether this is the case or not and
then to convert it into a increasing representation plus the permutation sign.
We call this representation, canonical.
```rust
pub fn is_basis(&self) -> bool {
  self.is_canonical()
}
pub fn is_canonical(&self) -> bool {
  let Some((sign, canonical)) = self.clone().canonicalized() else {
    return false;
  };
  sign == Sign::Pos && canonical == *self
}
pub fn canonicalized(mut self) -> Option<(Sign, Self)> {
  let sign = sort_signed(&mut self.indices);
  let len = self.indices.len();
  self.indices.dedup();
  if self.indices.len() != len {
    return None;
  }
  Some((sign, self))
}
```

In the case of a strictly increasing term, we can also determine the lexicographical
rank of it in the set of all increasing terms. And the other way constructing
them from lexicographical rank.
```rust
pub fn lex_rank(&self) -> usize {
  assert!(self.is_canonical(), "Must be canonical.");
  let n = self.dim();
  let k = self.indices.len();

  let mut rank = 0;
  for (i, &index) in self.indices.iter().enumerate() {
    let start = if i == 0 { 0 } else { self.indices[i - 1] + 1 };
    for s in start..index {
      rank += binomial(n - s - 1, k - i - 1);
    }
  }
  rank
}

pub fn from_lex_rank(dim: Dim, grade: ExteriorGrade, mut rank: usize) -> Self {
  let mut indices = Vec::with_capacity(grade);
  let mut start = 0;
  for i in 0..grade {
    let remaining = grade - i;
    for x in start..=(dim - remaining) {
      let c = binomial(dim - x - 1, remaining - 1);
      if rank < c {
        indices.push(x);
        start = x + 1;
        break;
      } else {
        rank -= c;
      }
    }
  }
  Self::new(indices, dim)
}
```

Now that we have this we can implement a useful iterator on our `ExteriorElement`
struct that allows us to iterate through the basis expansion consisting
of both the coefficient and the exterior basis element.
```rust
pub fn basis_iter(&self) -> impl Iterator<Item = (f64, ExteriorTerm)> + '_ {
  let dim = self.dim;
  let grade = self.grade;
  self
    .coeffs
    .iter()
    .copied()
    .enumerate()
    .map(move |(i, coeff)| {
      let basis = ExteriorTerm::from_lex_rank(dim, grade, i);
      (coeff, basis)
    })
}
```
We then implemented the addition and scalar multiplication of exterior elements
by just applying the operation to the coefficients.

== Exterior Product

The most obvious operation on a `ExteriorElement` is of course the exterior product.
For this we rely on the exterior product of two `ExteriorTerm`s,
which is just a concatenation of the two multiindices.
```rust
impl ExteriorTerm {
  pub fn wedge(mut self, mut other: Self) -> Self {
    self.indices.append(&mut other.indices);
    self
  }
}
```

For the `ExteriorElement` we just iterate over the all combinations of
basis expansion and canonicalize the wedges of the individual terms.
```rust
impl ExteriorElement {
  pub fn wedge(&self, other: &Self) -> Self {
    assert_eq!(self.dim, other.dim);
    let dim = self.dim;

    let new_grade = self.grade + other.grade;
    if new_grade > dim {
      return Self::zero(dim, 0);
    }

    let new_basis_size = binomial(dim, new_grade);
    let mut new_coeffs = na::DVector::zeros(new_basis_size);

    for (self_coeff, self_basis) in self.basis_iter() {
      for (other_coeff, other_basis) in other.basis_iter() {
        let self_basis = self_basis.clone();

        let coeff_prod = self_coeff * other_coeff;
        if self_basis == other_basis || coeff_prod == 0.0 {
          continue;
        }
        if let Some((sign, merged_basis)) = self_basis.wedge(other_basis).canonicalized() {
          let merged_basis = merged_basis.lex_rank();
          new_coeffs[merged_basis] += sign.as_f64() * coeff_prod;
        }
      }
    }

    Self::new(new_coeffs, dim, new_grade)
  }
}
```

And we also implement a big wedge operator, that takes an iterator of factors.
```rust
pub fn wedge_big(factors: impl IntoIterator<Item = Self>) -> Option<Self> {
  let mut factors = factors.into_iter();
  let first = factors.next()?;
  let prod = factors.fold(first, |acc, factor| acc.wedge(&factor));
  Some(prod)
}
```

== Inner product on Exterior Elements

For the weak formulations of our PDEs we rely on Hilbert spaces that require
an $L^2$-inner product on differential forms.
This is derived directly from the point-wise inner product on multiforms.
Which itself is derived from the inner product on the tangent space,
which comes from the Riemannian metric at the point.

This derivation from the inner product on the tangent space $g_p$
to the inner product on the exterior fiber $wedge.big^k T^*_p M$, shall
be computed.

In general given an inner product on the vector space $V$, we can
derive an inner product on the exterior power $wedgespace^k (V)$.
The rule is the following:
$
  inner(e_I, e_J) = det [inner(dif x_I_i, dif x_I_j)]_(i,j)^k
$

Computationally we represent inner products as Gramian matrices on some basis.
This means that we compute an extended Gramian matrix as the inner product on
multielements from the Gramian matrix of single elements using the determinant.
```rust
impl RiemannianMetricExt for RiemannianMetric {
  fn multi_form_gramian(&self, k: ExteriorGrade) -> na::DMatrix<f64> {
    let n = self.dim();
    let bases: Vec<_> = exterior_bases(n, k).collect();
    let covector_gramian = self.covector_gramian();

    let mut multi_form_gramian = na::DMatrix::zeros(bases.len(), bases.len());
    let mut multi_basis_mat = na::DMatrix::zeros(k, k);

    for icomb in 0..bases.len() {
      let combi = &bases[icomb];
      for jcomb in icomb..bases.len() {
        let combj = &bases[jcomb];

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

We are already at the end of the implementation of the exterior algebra.
There exist many operations that could be implemented as well, such as the
Hodge star operator, based on an inner product, but it's just not necessary
for the rest of the library to have such functionality, therefore we omit it
here.

= Discrete Differential Forms:\ Simplicial Cochains and Whitney Forms

Smooth Manifold discretizes to Simplicial Complex.
Continuous Differential Forms on Manifold discretizes to Simplicial cochain on
Simplicial Complex.
Discrete Differential $k$-form is Simplicial $k$-cochain, which
are real values on the $k$-skeleton.

Simplicial cochains are a structure preserving discretization
and therefore retain the key topological and geometrical properties
from differential geometry.
This will become apart in our discussion of cochain calculus,
where we will see coboundary operators.

We will discuss the discretization procedure of arbitrary coordinate-based continuum
differential forms by means of a cochain-projection via de Rham's map.
And the reconstruction of a continuum differential form over the cells by means
of cochain-interpolation via Whitney's map onto Whitney forms.


Simplicial cochains arise naturally from the combinatorial structure of a
simplicial complex and can be interpreted as discrete analogues of differential
forms via integration. They form the algebraic backbone of discrete exterior
calculus and finite element exterior calculus (FEEC), representing cohomology
classes and supporting discrete versions of exterior operators.

Whitney forms, on the other hand, are low-order, piecewise polynomial
differential forms defined on each simplex. They interpolate cochain values into
the continuous setting and serve as basis functions in finite element spaces
of differential forms.
Whitney map: Canonical map from cochains to differential forms.

This chapter lays the foundation for the discrete variational formulations used
n FEEC.

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


=== Discretization: Cochain-Projection via de Rham's map

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

This inspires a definition of the discrete exterior derivative as
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

By construction Stokes' Theorem is fulfilled, which is crucial to many applications.

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


Via linear combination of the global basis functions we can obtain
any piecewise-linear differential-form from the Whitney space.

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    rows: 1,
    gutter: 3pt,
    image("../res/triforce_constant.cochain.png", width: 100%),
    image("../res/triforce_div.cochain.png", width: 100%),
    image("../res/triforce_rot.cochain.png", width: 100%),
  ),
  caption: [
    Vector proxies of some example FE functions on equilateral triangle mesh
    $mesh$.
  ],
) <img:fe_whitneys>

=== Reconstruction: Whitney Interpolation via the Whitney map

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
This simplex is the DOF and it's coefficient is the cochain value
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
which is the cochain--into a point-evaluable differential form.
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
We store the the vertex coordinates of the cell, the local subsimplex
and additionally store the precomputed constant exterior derivatives of the
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

== Higher-Order Discrete Differential Forms

The theoretical construction of finite element differential forms
exist for all polynomial degrees.
We don't support them in this implementation, but this
a very obvious possible future extension to this implementation.
One just needs to keep in mind that then higher-order manifold
approximations are also needed to not incur any non-admissible geometric
variational crimes.


= Finite Element Methods\ for Differential Forms

We have now arrived at the chapter talking about the
actual finite element library formoniq. \
Here we will derive and implement the formulas for computing the element matrices
of the various weak differential operators in FEEC.
Furthermore we implement the assembly algorithm that will give us the
final Galerkin matrices.

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
assemble the full Galerkin matrices.

We first define a element matrix provider trait
```rust
pub type ElMat = na::DMatrix<f64>;
pub trait ElMatProvider {
  fn row_grade(&self) -> ExteriorGrade;
  fn col_grade(&self) -> ExteriorGrade;
  fn eval(&self, geometry: &SimplexGeometry) -> ElMat;
}
```
The `eval` method provides us with the element matrix on a
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

One could also understand the mass bilinear form as a weak Hodge star operator.
$
  amat(M)_(i j) = integral_Omega phi_j wedge hodge phi_i
  = inner(phi_j, phi_i)_(L^2 Lambda^k (Omega))
$

We will not compute this using the Hodge star operator, but instead directly
using the inner product.

We already have an inner product on constant multiforms. We now need to
extend it to an $L^2$ inner product on Whitney forms.
This can be done by inserting the definition of a Whitney form (in terms of barycentric
coordinate functions) into the inner product.

$
  inner(lambda_(i_0 dots i_k), lambda_(j_0 dots j_k))_(L^2 Lambda^k (Omega))
  &= k!^2 sum_(l=0)^k sum_(m=0)^k (-)^(l+m) innerlines(
    lambda_i_l (dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k),
    lambda_j_m (dif lambda_j_0 wedge dots.c wedge hat(dif lambda)_j_m wedge dots.c wedge dif lambda_j_k),
  )_(L^2 Lambda^k (Omega)) \
  &= k!^2 sum_(l,m) (-)^(l+m) innerlines(
    dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k,
    dif lambda_j_0 wedge dots.c wedge hat(dif lambda)_j_m wedge dots.c wedge dif lambda_j_k,
  )_(Lambda^k)
  integral_K lambda_i_l lambda_j_m vol \
$


We can now make use of the fact that the exterior derivative of the barycentric
coordinate functions are constant. This makes the wedge big terms also constant.
We can therefore pull them out of the integral inside the $L^2$-inner product
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

Now we are just missing an element matrix provider for the scalar mass
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

== 1-Form EVP on Annulus

We meshed a 2D annulus $BB_1 (0) \\ BB_(1\/4) (0)$ using gmsh.

The eigenvalues computed on the annulus correspond to the actual eigenvalues.

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    rows: 1,
    gutter: 3pt,
    image("../res/evp0.png", width: 100%),
    image("../res/evp5.png", width: 100%),
    image("../res/evp6.png", width: 100%),
  ),
  caption: [
    Three interesting computed eigenfunctions.
  ],
) <img:evp>

== 1-Form Source Problem on $RR^n, n >= 1$

We verify the source problem by means of the method of manufactured solution.
Our manufactured solution is a 1-form that follows the same pattern for any
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


#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    rows: 1,
    gutter: 3pt,
    [],
    image("../res/source_problem.png", width: 100%),
    [],
  ),
  caption: [
    Computed solution to source problem.
  ],
) <img:evp>



// Probably move this to post-face
= Conclusion and Outlook

- Summary of key contributions
- Possible improvements and future work
  - efficiency
  - parametric FE
  - higher-order FEEC and higher-order elements
- Broader impact (e.g., Rust in scientific computing, FEEC extensions)
- Discarded ideas and failed approaches (generic dimensionality à la nalgebra/Eigen)

A far more meaningful PDE system that has some really interesting applications in real-life
are Maxwell's Equation describing Electromagnetism.
FEEC is the perfect fit for Maxwell's equations, since the relativistic variant of them
is also formulated in terms of differential geometry as is general relativity.
This means that purely thanks to the generality of the library we are able to solve
Maxwell's equations on the curved 4D spacetime manifold.

