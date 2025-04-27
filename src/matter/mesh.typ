#import "../setup.typ": *
#import "../setup-math.typ": *

= Topology & Geometry of Simplicial Riemannian Manifolds

In this chapter, we develop various data structures and algorithms to represent
and work with our Finite Element mesh. This mesh stores the topological and
geometrical properties of our arbitrary-dimensional discrete PDE domain. A
simplicial complex @hatcher:algtop will represent the topology (incidence and
adjacency) of the mesh and serve as the container for all mesh entities, which
are simplices. It also provides unique identification through global numbering
and iteration over these entities. For the geometry, edge lengths are stored to
compute the piecewise-flat (over the cells) Riemannian metric @frankel:diffgeo,
using methods from Regge Calculus @regge. We also support the optional storage of global
vertex coordinates if an embedding is known.

== Coordinate Simplices

Finite Element Methods benefit from their ability to work on unstructured meshes
@hiptmair:numpde. Instead of subdividing a domain into a regular grid, FEM
operates on potentially highly non-uniform meshes. The simplest mesh type
suitable for such non-uniform domains are simplicial meshes. In 2D, these are
the familiar triangular meshes from computer graphics, while 3D simplicial
meshes are composed of tetrahedra. These building blocks must be generalized for
our arbitrary-dimensional implementation.

We begin the discussion of mesh representation with a coordinate-based object
that relies on an embedding in an ambient space. Later, we will discard the
coordinates and rely solely on intrinsic geometry. However, for didactic
purposes, starting with coordinates is helpful.

#v(0.5cm)

The generalization of 2D triangles and 3D tetrahedra to $n$ dimensions is called
an $n$-simplex @hatcher:algtop. There is a type of simplex for every dimension;
the first four kinds are:
- A 0-simplex is a point.
- A 1-simplex is a line segment.
- A 2-simplex is a triangle.
- A 3-simplex is a tetrahedron.

The underlying idea is that an $n$-simplex is the convex hull of $n+1$ affinely
independent points, forming the simplest possible $n$-dimensional polytope. An
$n$-simplex $sigma$ is defined by $n+1$ vertices $avec(v)_0, ..., avec(v)_n in
RR^N$ in a possibly higher-dimensional space $RR^N$ (where $N >= n$). The
simplex itself is the region bounded by the convex hull of these vertices:
$
  sigma =
  "convex" {avec(v)_0,...,avec(v)_n} =
  {
    sum_(i=0)^n lambda^i avec(v)_i
    mid(|)
    quad lambda^i >= 0,
    quad sum_(i=0)^n lambda^i = 1
  }
$ <def-simplex>

We call such an object a *coordinate simplex* because it depends on the global
coordinates of its vertices and resides within a potentially higher-dimensional
ambient space $RR^N$, thus relying on an embedding. This object is uniquely
determined by the vertex coordinates $avec(v)_i$. This inspires a straightforward
computational representation using a struct that stores the coordinates of each
vertex as columns in a matrix:
```rust
pub struct SimplexCoords {
  pub vertices: na::DMatrix<f64>,
}
impl SimplexCoords {
  pub fn nvertices(&self) -> usize { self.vertices.ncols() }
  pub fn coord(&self, ivertex: usize) -> CoordRef { self.vertices.column(ivertex) }
}
```

We implement methods to retrieve both the intrinsic dimension $n$ (one less than
the number of vertices) and the ambient dimension $N$ (the dimension of the
coordinate vectors). A special and particularly simple case occurs when the
intrinsic and ambient dimensions coincide, $n=N$.
```rust
pub type Dim = usize;
pub fn dim_intrinsic(&self) -> Dim { self.nvertices() - 1 }
pub fn dim_ambient(&self) -> Dim { self.vertices.nrows() }
pub fn is_same_dim(&self) -> bool { self.dim_intrinsic() == self.dim_ambient() }
```

#v(0.5cm)
=== Barycentric Coordinates

The coefficients $avec(lambda) = [lambda^i]_(i=0)^n$ in @def-simplex are called
*barycentric coordinates* @frankel:diffgeo, @hiptmair:numpde. They appear in the
convex combination $sum_(i=0)^n lambda^i avec(v)_i$ as weights
$lambda^i in [0,1]$ (for points within the simplex) that sum to one
$sum_(i=0)^n lambda^i = 1$, in front of the Cartesian vertex coordinates
$avec(v)_i in RR^N$. They constitute an intrinsic local coordinate representation with
respect to the simplex $sigma$, independent of the embedding in $RR^N$, relying
only on the convex combination of vertices.

The coordinate transformation $phi: avec(lambda) |-> avec(x)$ from intrinsic
barycentric coordinates $avec(lambda)$ to ambient Cartesian coordinates $avec(x)$
is given by:
$
  phi: avec(lambda) |-> avec(x) = sum_(i=0)^n lambda^i avec(v)_i
$

This transformation can be implemented as:
```rust
pub fn bary2global<'a>(&self, bary: impl Into<CoordRef<'a>>) -> Coord {
  self
    .vertices
    .coord_iter()
    .zip(bary.into().iter())
    .map(|(vi, &baryi)| baryi * vi)
    .sum()
}
```

The barycentric coordinate representation extends beyond the simplex
boundaries to the entire affine subspace spanned by the vertices. The condition
$sum_(i=0)^n lambda^i = 1$ must still hold, but only points $avec(x) in sigma$
strictly inside the simplex have all $lambda^i in [0,1]$. Outside the simplex,
some $lambda^i$ will be greater than one or negative.
```rust
pub fn is_bary_inside<'a>(bary: impl Into<CoordRef<'a>>) -> bool {
  let bary = bary.into();
  assert_relative_eq!(bary.sum(), 1.0);
  bary.iter().all(|&b| (0.0..=1.0).contains(&b))
}
```

The barycenter $avec(m) = 1/(n+1) sum_(i=0)^n avec(v)_i$ always has the special
barycentric coordinate $avec(lambda) = [1/(n+1)]^(n+1)$ and
is namesake for the barycentric coordinates.
```rust
pub fn barycenter(&self) -> Coord {
  let mut barycenter = na::DVector::zeros(self.dim_ambient());
  self.vertices.column_iter().for_each(|v| barycenter += v);
  barycenter /= self.nvertices() as f64;
  barycenter
}
```

This coordinate system treats all vertices symmetrically, assigning a weight to
each. Consequently, with $n+1$ coordinates $lambda^0, ..., lambda^n$ for an
$n$-dimensional affine subspace subject to the constraint $sum lambda^i = 1$,
there is redundancy. This representation is not a minimal coordinate system.

To obtain a proper coordinate system, we can single out one vertex, say
$avec(v)_0$, as the *base vertex*.
```rust
pub fn base_vertex(&self) -> CoordRef { self.coord(0) }
```
We can then omit the redundant coordinate $lambda^0 = 1 - sum_(i=1)^n lambda^i$
associated with $avec(v)_0$. The remaining *reduced barycentric coordinates*
$avec(lambda)^- = [lambda^i]_(i=1)^n$ form a proper coordinate system for the
$n$-dimensional affine subspace. This is also referred to as the *local
coordinate system*. In this system, the local coordinates $lambda^1, ..., lambda^n$
are unconstrained, providing a unique representation for every point in the
affine subspace via a bijection with $RR^n$.


```rust
pub fn bary2local<'a>(bary: impl Into<CoordRef<'a>>) -> Coord {
  let bary = bary.into();
  bary.view_range(1.., ..).into()
}
pub fn local2bary<'a>(local: impl Into<CoordRef<'a>>) -> Coord {
  let local = local.into();
  let bary0 = 1.0 - local.sum();
  local.insert_row(0, bary0)
}
```


#v(0.5cm)
=== Spanning Vectors

Consider the edge vectors emanating from the base vertex: $avec(e)_i = avec(v)_i
- avec(v)_0 in RR^N$ for $i=1, ..., n$. These are the *spanning vectors*. We can
collect them as columns into a matrix $amat(E) in RR^(N times n)$:
$
  amat(E) = 
  mat(
    |,  , |;
    avec(e)_1,dots.c,avec(e)_n;
    |,  , |;
  )
$

This matrix can be computed as follows:
```rust
pub fn spanning_vectors(&self) -> na::DMatrix<f64> {
  let mut mat = na::DMatrix::zeros(self.dim_ambient(), self.dim_intrinsic());
  let v0 = self.base_vertex();
  // Skip base vertex (index 0)
  for (i, vi) in self.vertices.column_iter().skip(1).enumerate() { 
    let v0i = vi - v0;
    mat.set_column(i, &v0i);
  }
  mat
}
```

This gives us an explicit basis of the affine space located at $avec(v)_0$ and
spanned by $avec(e)_1,...,avec(e)_n$.

We can also rewrite the coordinate transformation $phi: avec(lambda) |->
avec(x)$ using $lambda^0 = 1 - sum_(i=1)^n lambda^i$ in terms of
the spanning vectors instead of the vertices:
$
  avec(x)
  = sum_(i=0)^n lambda^i avec(v)_i 
  = (1 - sum_(i=1)^n lambda^i) avec(v)_0 + sum_(i=1)^n lambda^i avec(v)_i
  = avec(v)_0 + sum_(i=1)^n lambda^i (avec(v)_i - avec(v)_0)
  = avec(v)_0 + amat(E) avec(lambda)^-
$

This shows that the transformation $phi$ is actually an affine map $phi:
avec(lambda)^- |-> avec(x)$ consisting of the linear map represented by $amat(E)$
followed by a translation by $avec(v)_0$.
$
  phi: avec(lambda)^- |-> avec(x) = avec(v)_0 + amat(E) avec(lambda)^-
$

We implement functions for this affine transformation:
```rust
pub fn linear_transform(&self) -> na::DMatrix<f64> { self.spanning_vectors() }
pub fn affine_transform(&self) -> AffineTransform {
  let translation = self.base_vertex().into_owned();
  let linear = self.linear_transform();
  AffineTransform::new(translation, linear)
}
pub fn local2global<'a>(&self, local: impl Into<CoordRef<'a>>) -> Coord {
  let local = local.into();
  self.affine_transform().apply_forward(local)
}
```

This makes use of the `AffineTransform` struct and its forward application:
```rust
pub struct AffineTransform {
  pub translation: na::DVector<f64>,
  pub linear: na::DMatrix<f64>,
}
impl AffineTransform {
  pub fn new(translation: Vector, linear: Matrix) -> Self { Self { translation, linear } }
  pub fn dim_domain(&self) -> usize { self.linear.ncols() }
  pub fn dim_image(&self) -> usize { self.linear.nrows() }
  pub fn apply_forward(&self, coord: na::DVectorView<f64>) -> na::DVector<f64> {
    &self.linear * coord + &self.translation
  }
}
```

The reverse transformation $psi: avec(x) |-> avec(lambda)^-$ from local
$avec(x)$ to global $avec(lambda)^-$ coordinates is
more complex due to the potentially higher-dimensional ambient space $RR^N$ with $N >= n$.
The global coordinate point might not lie exactly in the affine subspace due
to floating-point inaccuracies. This makes the linear system for the reverse
transformation $avec(x) - avec(v)_0 = amat(E) avec(lambda)^-$ potentially
underdetermined. We use the Moore-Penrose pseudo-inverse $amat(E)^dagger$
@hiptmair:numcse, typically computed via Singular Value Decomposition (SVD), to
find the unique least-squares solution of smallest norm:
$
  psi: avec(x) |-> avec(lambda)^- = amat(E)^dagger (avec(x) - avec(v)_0)
$

```rust
impl SimplexCoords {
  pub fn global2local<'a>(&self, global: impl Into<CoordRef<'a>>) -> Coord {
    let global = global.into();
    self.affine_transform().apply_backward(global)
  }
}

impl AffineTransform {
  pub fn apply_backward(&self, coord: VectorView) -> Vector {
    if self.dim_domain() == 0 { return Vector::zeros(0); }
    self
      .linear
      .clone()
      .svd(true, true)
      .solve(&(coord - &self.translation), 1e-12)
      .expect("SVD solve failed")
  }
  pub fn pseudo_inverse(&self) -> Self {
    if self.dim_domain() == 0 {
      return Self::new(Vector::zeros(0), Matrix::zeros(0, self.dim_image()));
    }
    let linear = self.linear.clone().pseudo_inverse(1e-12).unwrap();
    let translation = &linear * &self.translation;
    Self { translation, linear }
  }
}
```

The derivatives of the affine transformation $phi: avec(lambda)^- |->
avec(x)$ reveal that the spanning vectors $avec(e)_i$ form a natural basis
for the tangent space $T_p sigma$ at any point $p$ within the simplex $sigma$
@frankel:diffgeo. The Jacobian of the affine map is precisely $amat(E)$.
$
  diff/(diff lambda^i) = (diff avec(x))/(diff lambda^i) = avec(e)_i
  quad quad
  (diff avec(x))/(diff avec(lambda)^-) = amat(E)
$

In differential geometry terms, the linear map represented by $amat(E)$ acts as
*the pushforward*.
$
  phi_*: avec(u) |-> avec(w) = amat(E) avec(u)
$
It transforms intrinsic tangent vectors $avec(u) = u^i diff/(diff lambda^i)$ to
ambient tangent vectors $avec(w) = w^i avec(e)_i$.
```rust
/// Local2Global Tangentvector
pub fn pushforward_vector<'a>(&self, local: impl Into<TangentVectorRef<'a>>) -> TangentVector {
  self.linear_transform() * local.into()
}
```

Conversely, the *pullback* $phi^*$ operation takes covectors defined in the
ambient space coordinates and expresses them in the local coordinate system.
If $avec(omega) in RR^(1 times N)$ is a covector in the ambient space, its
pullback $avec(eta) = phi^* avec(omega)$ acts on local tangent vectors $avec(u)$ such
that $avec(eta)(avec(u)) = avec(omega)(phi_* avec(u)) = avec(omega)(amat(E) avec(u))$.
The transformation rule is right-multiplication if covectors are row vectors.
$
  phi^*: avec(omega) |-> avec(eta) = avec(omega) amat(E)
$
```rust

/// Global2Local Cotangentvector
pub fn pullback_covector<'a>(
  &self,
  global: impl Into<CoTangentVectorRef<'a>>,
) -> CoTangentVector {
  global.into() * self.linear_transform()
}
```

Separately, we can consider the differentials of the barycentric coordinate
functions $lambda^i$ as functions of the global coordinates $avec(x)$. These
differentials, $dif lambda^i$, form a basis for the cotangent space $T^*_p sigma$.
Their components relative to the ambient basis $dif x^i$ are found using the
differential of the inverse map $psi: avec(x) -> avec(lambda)^-$, which involves the
pseudo-inverse $amat(E)^dagger$. Specifically, the rows of $amat(E)^dagger$ give
the components of $dif lambda^1, ..., dif lambda^n$.
The
differential $dif lambda^0$ is determined by the constraint $sum_i dif lambda^i = 0$.
$


  (diff avec(lambda)^-)/(diff avec(x)) = amat(E)^dagger
  quad quad
  dif lambda^i = (diff lambda^i)/(diff avec(x)) = epsilon^i = (amat(E)^dagger)_(i,:)
  quad quad
  dif lambda^0 = -sum_(i=1)^n dif lambda^i
$

These $epsilon^i = dif lambda^i$ form the basis dual to the tangent basis $avec(e)_1, ...,
avec(e)_n$ @frankel:diffgeo.
$
  dif lambda^i (diff/(diff lambda^j)) = delta^i_j
$


```rust
/// Total differential of barycentric coordinate functions in the rows(!) of
/// a matrix.
pub fn difbarys(&self) -> Matrix {
  let difs = self.inv_linear_transform();
  let mut difs = difs.insert_row(0, 0.0); // Add row for lambda^0
  difs.set_row(0, &-difs.row_sum()); // lambda^0 = -sum(dif lambda^i)
  difs
}
```

The spanning vectors as a basis of the tangent space, can be used to derive
the *Riemannian metric tensor*. This metric on the simplex is induced by the ambient
Euclidean metric via the affine embedding.
$
  amat(G) = amat(E)^transp amat(E)
$

Our implementation has a struct for working with Gramians, which will discuss
in more detail later on.
```rust
pub fn metric_tensor(&self) -> Gramian {
  Gramian::from_euclidean_vectors(self.spanning_vectors())
}
```
Since $amat(E)$ is constant across the simplex, this induced metric $amat(G)$
is also *constant* everywhere within the simplex. This constancy implies that
the simplex is *intrinsically flat* (zero Riemannian curvature). Consequently,
geodesics (shortest paths between points) within the simplex are simply straight
line segments in the embedding.

The spanning vectors also define a parallelepiped. The volume of the $n$-simplex $sigma$
is $1/n!$ times the $n$-dimensional volume of this parallelepiped. The signed volume
is computed using the determinant of the spanning vectors if $n=N$.
$
  vol(sigma) = 1/n! det(amat(E))
$
For higher-dimensional ambient spaces we can use the Gram determinant of the metric tensor.
@frankel:diffgeo.
$
  vol(sigma) = 1/n! sqrt(amat(G)) = 1/n! sqrt(det(amat(E)^transp amat(E)))
$

```rust
impl SimplexCoords {
  pub fn det(&self) -> f64 {
    let det = if self.is_same_dim() {
      self.spanning_vectors().determinant()
    } else {
      self.metric_tensor().det_sqrt()
    };
    refsimp_vol(self.dim_intrinsic()) * det
  }

  pub fn vol(&self) -> f64 { self.det().abs() }
  pub fn is_degenerate(&self) -> bool { self.vol() <= 1e-12 }
}
pub fn refsimp_vol(dim: Dim) -> f64 {
  factorialf(dim).recip()
}
```

The sign of the signed volume gives the global orientation of the coordinate
simplex relative to the ambient space $RR^N$.
```rust
pub fn orientation(&self) -> Sign {
  Sign::from_f64(self.det()).unwrap()
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Sign {
  #[default]
  Pos = 1,
  Neg = -1,
}
impl Sign {
  pub fn from_f64(f: f64) -> Option<Self> {
    if f == 0.0 {
      return None;
    }
    Some(Self::from_bool(f > 0.0))
  }
}
```

Swapping two vertices in the definition of the simplex negates the determinant
and thus flips the orientation. Every non-degenerate simplex has exactly two
orientations, positive and negative @hatcher:algtop.
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

A particularly important simplex is the *reference simplex* @hiptmair:numpde,
which serves as a canonical domain for defining basis functions in FEM. For a
given dimension $n$, the $n$-dimensional reference simplex $hat(sigma)^n$ is
defined in $RR^n$ (so $N=n$) using its local
coordinates as its global Cartesian coordinates:
$
  hat(sigma)^n = {(lambda_1,...,lambda_n) in RR^n mid(|) lambda_i >= 0, quad sum_(i=1)^n lambda_i <= 1 }
$
Its vertices are the origin $avec(v)_0 = avec(0)$ and the standard basis vectors
$avec(v)_i = nvec(e)_i$ for $i=1...n$. The spanning vectors are simply the
standard basis vectors $avec(e)_i = nvec(e)_i$, so $amat(E) = amat(I)_n$. The
metric tensor is the identity matrix $amat(G) = amat(I)_n$, representing the
standard Euclidean inner product. Its volume is $(n!)^(-1)$.

When looking at an arbitrary "real" $n$-simplex $tau$,
the local to global map $phi_tau: avec(lambda)^- |-> avec(x)$ can be seen as a
parametrization $phi_tau: hat(sigma)^n -> tau subset.eq RR^N$, where the parametrization domain
is the reference $n$-simplex. The real simplex is then the image of the reference simplex
$
  sigma = phi_sigma (hat(sigma)^n)
$
Conversely the global to local map $psi_tau: avec(x) |-> avec(lambda)^-$ can
be seen as a chart map $psi_tau: tau -> hat(sigma)^n$ where the chart itself
is the reference $n$-simplex.

When taking the "real" simplex to be the reference simplex, then
the parametrization and chart maps are both identity maps.


Okay, here is a revised version of the "Abstract Simplices" section, incorporating improvements while adhering to your constraints. I have focused on clarity, flow, and correcting minor grammatical issues, ensuring the text accurately describes the provided code's behavior, including the specific interpretation of `subsets`.

---

== Abstract Simplices

After studying coordinate simplices, the reader has hopefully developed some
intuitive understanding of simplices. We will now shed the coordinates and
represent simplices in a more abstract way, by considering them merely as a list
of vertex indices, without any associated vertex coordinates. An $n$-simplex
$sigma$ is thus represented as an $(n+1)$-tuple of natural numbers, which
correspond to vertex indices @hatcher:algtop.
$
  sigma = [v_0,...,v_n] in NN^(n+1)
  quad quad
  v_i in NN
$

In Rust, we can simply represent this using the following struct:
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

The ordering of the vertices _does_ matter; therefore, we are dealing with
ordered tuples, not just unordered sets. This makes our simplices combinatorial
objects, and these combinatorics will be the heart of our mesh data structure.

=== Sorted Simplices

Even though the order matters for defining a specific simplex instance,
simplices that share the same set of vertices are closely related. For this
reason, it is helpful to introduce a convention for the canonical representation
of a simplex given a particular set of vertices.

Our canonical representative will be the tuple whose vertex indices are sorted
in increasing order. We can take any simplex with arbitrarily ordered vertices
and convert it to its canonical representation.
```rust
pub fn is_sorted(&self) -> bool { self.vertices.is_sorted() }
pub fn sort(&mut self) { self.vertices.sort_unstable() }
pub fn sorted(mut self) -> Self { self.sort(); self }
```

Using this canonical representation, we can easily check whether two simplices
have the same vertex set, meaning they are permutations of each other.
```rust
pub fn set_eq(&self, other: &Self) -> bool {
  self.clone().sorted() == other.clone().sorted()
}
pub fn is_permutation_of(&self, other: &Self) -> bool { self.set_eq(other) }
```

=== Orientation

For coordinate simplices, we observed that a simplex always has two possible
orientations. We computed this orientation based on the determinant of the
spanning vectors, but without coordinates, this is no longer possible.

However, we can still define a notion of relative orientation. Recall that
swapping two vertices in a coordinate simplex flips its orientation due to
the properties of the determinant. The same behavior is present in abstract
simplices, based purely on the ordering of the vertices. All permutations of a
given set of vertices can be divided into two equivalence classes relative to
a reference ordering: even and odd permutations @hatcher:algtop. We associate
simplices with even permutations with positive orientation and those with
odd permutations with negative orientation. Therefore, every abstract simplex
has exactly two orientations, positive and negative, depending on its vertex
ordering relative to a reference.

We use our canonical sorted representation as the reference ordering. We can
determine the orientation of any simplex relative to this sorted permutation by
counting the number of swaps necessary to sort its vertex list. An even number
of swaps corresponds to a positive orientation, and an odd number corresponds
to a negative orientation. For this, we implement a basic bubble sort that keeps
track of the number of swaps. The computational complexity $cal(O) (n^2)$
is not optimal but sufficient for the typically small number of vertices per simplex.
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

Two simplices composed of the same vertex set have equal orientation if and
only if their respective permutations fall into the same equivalence class (both
even or both odd). Using the transitivity of this equivalence relation, we can
check this by comparing their orientations relative to the canonical sorted
permutation.
```rust
pub fn orientation_eq(&self, other: &Self) -> bool {
  self.orientation_rel_sorted() == other.orientation_rel_sorted()
}
```

=== Subsets

// TODO: There is some concern that the naming here is confusing.
// Subsets or sets in general are unordered, so producing multiple
// objects that are permutations of each other, might confuse...

Another important notion is that of a subsimplex or a face of a simplex
@hatcher:algtop. In this context, we first consider the concept of vertex
subsets.

A simplex $sigma$ can be considered a subset of another simplex $tau$ if all vertices
of $sigma$ are also vertices of $tau$. We can check this condition directly: $sigma
subset.eq tau <=> (forall a in sigma => a in tau)$
```rust
pub fn is_subset_of(&self, other: &Self) -> bool {
  self.iter().all(|v| other.vertices.contains(v))
}
pub fn is_superset_of(&self, other: &Self) -> bool {
  other.is_subset_of(self)
}
```

We can also generate all $k$-simplices whose vertex sets of size $k+1$
form subsets of the original $n$-simplex's $n+1$ vertices. The following
function generates all ordered $(k+1)$-tuples by taking *permutations* of
the $n+1$-tuple from the original simplex.
```rust
pub fn subsets(&self, sub_dim: Dim) -> impl Iterator<Item = Self> {
  itertools::Itertools::permutations(self.clone().into_iter(), sub_dim + 1).map(Self::from)
}
```

The number of distinct vertex subsets of size $k+1$ within a set of $n+1$
vertices is given by the binomial coefficient $binom(n+1, k+1)$.
```rust
pub fn nsubsimplices(dim_cell: Dim, dim_sub: Dim) -> usize {
  binomial(dim_cell + 1, dim_sub + 1)
}
```

Given a simplex that is a subset of a larger simplex, we can compute its vertex
indices relative to:
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

=== Subsequences

When considering the faces (like facets) of an $n$-simplex, it is often
desirable to have a unique representative simplex for each subset of vertices,
rather than all possible permutations. Furthermore, preserving the relative
order of vertices from the original simplex is often important, particularly for
defining orientations via the boundary operator.

For this, we consider *subsequences* of the original simplex's vertex list. A
subsequence maintains the relative order of the vertices it contains. We can
check if one simplex is a subsequence of another using a naive algorithm:
```rust
pub fn is_subsequence_of(&self, sup: &Self) -> bool {
  let mut sup_iter = sup.iter();
  self.iter().all(|item| sup_iter.any(|x| x == item))
}
pub fn is_supersequence_of(&self, other: &Self) -> bool {
  other.is_subsequence_of(self)
}
```

We provide a method for generating all $k$-dimensional subsimplices that
are *subsequences* of an $n$-simplex. This is achieved by generating all
$(k+1)$-length combinations (which preserve order) of the original $(n+1)$
vertices, using an implementation provided by the `itertools` crate @crate:itertools.
```rust
pub fn subsequences(&self, sub_dim: Dim) -> impl Iterator<Item = Self> {
  itertools::Itertools::combinations(self.clone().into_iter(), sub_dim + 1).map(Self::from)
}
```
This implementation conveniently provides the subsequences in lexicographical
order with respect to the vertex indices of the original simplex. If the
original simplex was sorted, then the generated subsequences are also
lexicographically ordered.

A standard operation is to generate all subsequence simplices of the standard
simplex $[0, 1, ..., n]$. We call these the standard subsimplices.
```rust
pub fn standard_subsimps(dim_cell: Dim, dim_sub: Dim) -> impl Iterator<Item = Simplex> {
  Simplex::standard(dim_cell).subsequences(dim_sub)
}
```
We can also generate all standard subsimplices for each dimension up to $n$ in a
graded fashion, which is useful for generating a standard simplicial complex.
```rust
pub fn graded_subsimps(dim_cell: Dim) -> impl Iterator<Item = impl Iterator<Item = Simplex>> {
  (0..=dim_cell).map(move |d| standard_subsimps(dim_cell, d))
}
```

Conversely, we can generate supersequences. Given a simplex and a "root" simplex
(which contains both the original simplex and its potential supersequences as
subsequences), we can find all subsequences of the root that have a specific
`super_dim` and contain the original simplex as a subsequence.
```rust
pub fn supersequences(
  &self,
  super_dim: Dim,
  root: &Self,
) -> impl Iterator<Item = Self> + use<'_> {
  root
    .subsequences(super_dim)
    .filter(|sup| self.is_subsequence_of(sup))
}
```

=== Boundary

A special operation related to subsequence simplices is the *boundary operator*
$diff$ @hatcher:algtop. Applied to an $n$-simplex $sigma = [v_0, ..., v_n]$,
the boundary operator is defined as the formal sum:
$
  diff sigma = sum_(i=0)^n (-1)^i [v_0,...,hat(v)_i,...,v_n]
$
Here, $hat(v)_i$ indicates that vertex $v_i$ is omitted. The result is a formal
sum of all $(n-1)$-dimensional subsequence simplices (facets) of $sigma$. Each
facet is assigned a sign $(-1)^i$, giving the boundary an orientation consistent
with the orientation of $sigma$. When viewed as elements of the free Abelian
group generated by all oriented simplices, this operator is linear.

For instance, the boundary of the triangle $sigma = [0,1,2]$ is
$
  diff sigma = (-1)^0 [1,2] + (-1)^1 [0,2] + (-1)^2 [0,1] = [1,2] - [0,2] + [0,1]
$
Rearranging terms to follow a path gives $[0,1] + [1,2] + [2,0]$, which
corresponds to traversing the edges of the triangle.

Our implementation generates the boundary facets using the `subsequences`
method, which yields them in an order based on the index of the *retained*
vertices (lexicographical combinations). This order differs from the summation
index $i$ (based on the *omitted* vertex) in the standard definition $sum_i
(-1)^i [...]$. The code calculates the appropriate sign for each generated
subsequence to ensure consistency with the standard alternating sign convention
of the boundary operator.
```rust
pub fn boundary(&self) -> impl Iterator<Item = SignedSimplex> {
  let mut sign = Sign::from_parity(self.nvertices() - 1);
  self.subsequences(self.dim() - 1).map(move |simp| {
    let this_sign = sign;
    sign.flip();
    SignedSimplex::new(simp, this_sign)
  })
}
```

To represent the terms in the formal sum, we introduce a struct that pairs a
simplex with an explicit sign:
```rust
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct SignedSimplex {
  pub simplex: Simplex,
  pub sign: Sign,
}
```


== Simplicial Skeleton

Simplices are the building blocks of our mesh. If we construct our mesh from
coordinate simplices, we obtain the typical Euclidean extrinsic description
of an embedded mesh, which inherently contains all geometric information. As
an embedding, the union of these coordinate simplices forms an $n$-dimensional
region within an $N$-dimensional Euclidean space.

If we, in contrast, build our mesh using only abstract simplices, we lack
this explicit geometric information, since abstract simplices only specify the
vertices (via indices) they comprise. This information, however, fully defines
the *topology* of our discrete $n$-manifold @hatcher:algtop. The connectivity
between simplices—whether they are adjacent or incident—is determined entirely
by shared vertices. This makes the topology purely combinatorial.

In the following sections, we will study this simplicial topology and implement
the necessary data structures and algorithms.

#v(1cm)

To define the topology of a simplicial $n$-manifold at its highest dimension, we
only need to store the set of $n$-simplices that constitute it. This collection
defines the top-level structure of the mesh. Such a collection of $n$-simplices,
typically sharing vertices from a common pool, is called an $n$-skeleton
@hatcher:algtop.

```rust
/// A container for sorted simplices of the same dimension.
#[derive(Default, Debug, Clone)]
pub struct Skeleton {
  /// Every simplex is sorted.
  simplices: IndexSet<Simplex>,
  nvertices: usize,
}
```

A `Skeleton` fulfills several responsibilities of a mesh data structure.
Primarily, it serves as a container for all $n$-simplices of a specific
dimension $n$. It allows for iteration over all these mesh entities:
```rust
impl Skeleton {
  pub fn iter(&self) -> indexmap::set::Iter<'_, Simplex> {
    self.simplices.iter()
  }
}
impl IntoIterator for Skeleton {
  type Item = Simplex;
  type IntoIter = indexmap::set::IntoIter<Self::Item>;
  fn into_iter(self) -> Self::IntoIter {
    self.simplices.into_iter()
  }
}
```
Crucially, it provides unique identification for each simplex through a
global numbering scheme within that dimension. This establishes a bijective
mapping between an integer index (`KSimplexIdx`) and the abstract simplex
itself. This functionality is achieved using the `IndexSet` data structure
from the `indexmap` crate. `IndexSet` maintains insertion order, allowing
efficient retrieval of a `Simplex` given its index (similar to `Vec`), while
also using hashing internally to support the reverse lookup: retrieving the
index corresponding to a given `Simplex` instance.
```rust
pub fn simplex_by_kidx(&self, idx: KSimplexIdx) -> &Simplex {
  self.simplices.get_index(idx).unwrap()
}
pub fn kidx_by_simplex(&self, simp: &Simplex) -> KSimplexIdx {
  self.simplices.get_index_of(simp).unwrap()
}
```

The `Skeleton` constructor enforces several guarantees about the simplices it
contains:
```rust
pub fn new(simplices: Vec<Simplex>) -> Self {
  assert!(!simplices.is_empty(), "Skeleton must not be empty");
  let dim = simplices[0].dim();
  assert!(
    simplices.iter().map(|simp| simp.dim()).all(|d| d == dim),
    "Skeleton simplices must have same dimension."
  );
  assert!(
    simplices.iter().all(|simp| simp.is_sorted()),
    "Skeleton simplices must be sorted."
  );
  let nvertices = if dim == 0 {
    assert!(simplices.iter().enumerate().all(|(i, simp)| simp[0] == i));
    simplices.len()
  } else {
    simplices
      .iter()
      .map(|simp| simp.iter().max().expect("Simplex is not empty."))
      .max()
      .expect("Simplices is not empty.")
      + 1
  };

  let simplices = IndexSet::from_iter(simplices);
  Self {
    simplices,
    nvertices,
  }
}
```
First, a skeleton cannot be empty, and all contained simplices must have the
same dimension. Furthermore, we enforce that only the canonical representation
(sorted vertex indices) of simplices is stored. This is essential for the
reverse mapping (simplex-to-index lookup) to be consistently useful; regardless
of the initial ordering of a simplex's vertices, converting it to the canonical
sorted form allows retrieval of its unique index within the skeleton. Lastly,
a special requirement applies to 0-skeletons: the simplices must represent the
vertices indexed sequentially as $[0], [1], ..., [N-1]$. The constructor also
determines and stores the total number of vertices (`nvertices`) involved in
the skeleton.


== Simplicial Complex

An $n$-skeleton provides the top-level topological information of a mesh by
defining its $n$-dimensional cells. However, this structure alone is often
insufficient for applications like Finite Element Exterior Calculus (FEEC)
@douglas:feec-book. Many FE methods associate degrees of freedom (DOFs) or
basis functions not only with cells but also with the lower-dimensional entities, in our
case subsimplices.
Therefore, we need a data structure that explicitly represents the topology of
_all_ relevant simplices within the mesh.

The skeleton only stores the top-level simplices $Delta_n (mesh)$. Our FEM
library, however, also needs to reference the lower-level simplices $Delta_k
(mesh)$ for $k < n$, since these are also mesh entities potentially carrying
DOFs.

Enter the *simplicial complex* @hatcher:algtop. A simplicial complex $K$ is a
collection of simplices such that:
+ Every face (subsequence simplex) of a simplex in $K$ is also in $K$.
+ The intersection of any two simplices in $K$ is either empty or a face of both.

In our implementation, we represent an $n$-dimensional simplicial complex by
storing the complete set of $k$-simplices for each dimension $k$ from $0$ to
$n$. Effectively, it comprises $n+1$ distinct skeletons, one for each dimension.

We use standard terminology for low-dimensional simplices within the complex:
-   The $0$-simplices are called *vertices*.
-   The $1$-simplices are called *edges*.
-   The $2$-simplices are called *faces*.
-   The $3$-simplices are called *tets*.
-   The $(n-1)$-simplices are called *facets*.
-   The $n$-simplices are called *cells*.
The `Complex` will serve as the main *topological data structure* passed as an
argument into our FEEC algorithm routines, providing access to all mesh entities
and their relationships.

While general simplicial complexes can represent complex topological spaces,
potentially including non-manifold features @hatcher:algtop, our target
applications in PDE modeling typically require computational domains that
are *manifolds*. A topological $n$-manifold is a space that locally resembles
Euclidean $n$-space. For a simplicial complex, this means the neighborhood of
each point (specifically, the link of each vertex) should be homeomorphic to an
$(n-1)$-sphere or an $(n-1)$-ball (if on the boundary) @hatcher:algtop.

Our `Complex` data structure is designed to represent such manifold domains. We
ensure two properties through construction:
1.  *Closure:* The complex contains all faces (subsequences) of its simplices.
  This is guaranteed by generating the complex from a list of top-level cells and
  explicitly adding all their subsequences.
2.  *Purity:* Every simplex of dimension $k < n$ is a face of at least one
  $n$-simplex (cell). This is also ensured by constructing the complex downwards
  from the cells.

However, the input `cells` skeleton itself might implicitly define a
non-manifold topology. A common example in 2D is when three or more triangles
meet along a single edge, or in 3D when multiple tetrahedra share a common face
like pages of a book. To ensure the represented space is a manifold (potentially
with boundary), we perform a check. A necessary condition for an $n$-dimensional
simplicial complex to be a manifold (without boundary) is that every facet
($(n-1)$-simplex) must be shared by exactly two cells ($(n)$-simplices). If
facets are shared by only one cell, this indicates they lie on the boundary
of the manifold. Our check verifies that each facet is contained in *at most*
two cells. This check is performed *after* building the full complex structure
because the required incidence information (which cells contain each facet) is
naturally computed during the construction process. It fundamentally serves as a
validation step for the input cell skeleton.

```rust
/// A simplicial manifold complex.
#[derive(Default, Debug, Clone)]
pub struct Complex {
  // Stores skeletons for dimensions 0 to n.
  skeletons: Vec<ComplexSkeleton>,
}
impl Complex {
  pub fn dim(&self) -> Dim { self.skeletons.len() - 1 }
}

/// A skeleton inside of a complex, pairing the raw Skeleton
/// with additional topological data computed during complex construction.
#[derive(Default, Debug, Clone)]
pub struct ComplexSkeleton {
  skeleton: Skeleton,
  complex_data: SkeletonComplexData,
}
impl ComplexSkeleton {
  pub fn skeleton(&self) -> &Skeleton {
    &self.skeleton
  }
  /// Accessor for the complex-specific data associated with each simplex
  /// in this skeleton.
  pub fn complex_data(&self) -> &[SimplexComplexData] {
    &self.complex_data
  }
}

/// Complex-specific data for all simplices in a single skeleton.
pub type SkeletonComplexData = Vec<SimplexComplexData>;

/// Complex-specific data associated with a single simplex within the complex.
#[derive(Default, Debug, Clone)]
pub struct SimplexComplexData {
  /// Stores the indices of the top-level cells (n-simplices)
  /// that contain this simplex as a face (subsequence).
  pub cocells: Vec<SimplexIdx>,
}
```

The primary method for creating a `Complex` is by providing the skeleton of
its highest-dimensional cells ($n$-simplices). The `from_cells` constructor
then systematically builds the skeletons for all lower dimensions ($k=0, ...,
n-1$) by generating all subsequence simplices of the input cells. During this
process, it also computes crucial topological incidence information: for each
simplex, it records the list of top-level cells that contain it (its `cocells`).
This information is stored alongside the skeletons in the `ComplexSkeleton`
structure.

After populating all skeletons and computing the incidence data, the
constructor performs the manifold topology check described earlier. It examines
the $(n-1)$-skeleton (facets) and verifies that each facet is listed as a
subsequence of either one cell (indicating a boundary facet) or two cells
(indicating an interior facet). If any facet belongs to more than two cells,
the input `cells` skeleton does not represent a manifold, and the constructor
asserts failure.
```rust
impl Complex {
  pub fn from_cells(cells: Skeleton) -> Self {
    let dim = cells.dim();

    let mut skeletons = vec![ComplexSkeleton::default(); dim + 1];
    skeletons[0] = ComplexSkeleton {
      skeleton: Skeleton::new((0..cells.nvertices()).map(Simplex::single).collect()),
      complex_data: (0..cells.nvertices())
        .map(|_| SimplexComplexData::default())
        .collect(),
    };

    for (icell, cell) in cells.iter().enumerate() {
      for (
        dim_skeleton,
        ComplexSkeleton {
          skeleton,
          complex_data: mesh_data,
        },
      ) in skeletons.iter_mut().enumerate()
      {
        for sub in cell.subsequences(dim_skeleton) {
          let (sub_idx, is_new) = skeleton.insert(sub);
          let sub_data = if is_new {
            mesh_data.push(SimplexComplexData::default());
            mesh_data.last_mut().unwrap()
          } else {
            &mut mesh_data[sub_idx]
          };
          sub_data.cocells.push(SimplexIdx::new(dim, icell));
        }
      }
    }

    // Topology checks.
    if dim >= 1 {
      let facet_data = skeletons[dim - 1].complex_data();
      for SimplexComplexData { cocells } in facet_data {
        let nparents = cocells.len();
        let is_manifold = nparents == 2 || nparents == 1;
        assert!(is_manifold, "Topology must be manifold.");
      }
    }

    Self { skeletons }
  }
}
```

=== Simplices in the Mesh: Simplex Indices and Handles

To identify a simplex inside the mesh, we use an indexing system.
If the context of a concrete dimension is given,
then we only need to know the index inside the skeleton, which
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
data structure for storing our simplices inside a skeleton,
we are able to go both ways. We can not only retrieve the simplex, given the index,
but we can also get the index corresponding to a given simplex.

The `Simplex` struct doesn't reference the mesh and therefore doesn't have access
to any other simplices. But for doing any kind of topological computations,
it is helpful to be able to reference other simplices in the mesh.
For this reason we introduce a new concept, that represents a simplex
inside of a mesh. We create a simplex handle, that is like a more sophisticated
pointer to a simplex, that has a reference to the mesh.
This allows us to interact with these simplices inside the mesh very naturally.

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

For instance the `SimplexHandle::supersimps` gives us all supersimplices that
are also contained in the mesh, which are exactly all the supersequence simplices.
The `Simplex::supersequence` method however expects a `root: &Simplex`, which
gives the context in which we are searching for supersequences. This context
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
          .supersequences(dim_super, parent.raw())
          .map(move |sup| self.complex.skeleton(dim_super).get_by_simplex(&sup))
      })
      .collect()
  }
```
Furthermore these functions always directly access the `IndexSet` and retrieve
the corresponding index of the simplex and construct a new `SimplexHandle` out of
it, such that we can easily apply subsequent method calls on the returned objects.


=== Boundary Operator

We previously defined the boundary operator $diff$ for individual simplices.
This concept extends naturally to the entire complex. For each dimension $k$,
the boundary operator $diff_k$ maps the $k$-skeleton to the $(k-1)$-skeleton.
It can be represented as a linear operator, specifically a matrix $amat(D)_k$,
often called the *incidence matrix* @hatcher:algtop.

The matrix $amat(D)_k$ has dimensions $N_(k-1) times N_k$, where $N_j$ is the
number of $j$-simplices in the complex. The entry $(amat(D)_k)_(i j)$ represents
the signed incidence relation between the $i$-th $(k-1)$-simplex and the $j$-th
$k$-simplex. It is $+1$ or $-1$ if the $i$-th $(k-1)$-simplex is a facet of the
$j$-th $k$-simplex (with the sign determined by the relative orientation from
the boundary definition), and $0$ otherwise.
$
  amat(D)_k in {-1,0,+1}^(N_(k-1) times N_k)
$

The following function computes this sparse incidence matrix for a given
dimension `dim` ($k$).
```rust
impl Complex {
  pub fn boundary_operator(&self, dim: Dim) -> CooMatrix {
    let sups = &self.skeleton(dim);

    if dim == 0 {
      return CooMatrix::zeros(0, sups.len());
    }

    let subs = &self.skeleton(dim - 1);
    let mut mat = CooMatrix::zeros(subs.len(), sups.len());
    for (isup, sup) in sups.handle_iter().enumerate() {
      let sup_boundary = sup.boundary();
      for sub in sup_boundary {
        let sign = sub.sign.as_f64();
        let isub = subs.handle_by_simplex(&sub.simplex).kidx();
        mat.push(isub, isup, sign);
      }
    }
    mat
  }
}
```

== Simplicial Geometry

We have now successfully developed the topological structure of our mesh,
by combining many abstract simplices into skeletons and collecting
all of these skeletons together.

What we are still missing in our mesh data structure now, is any geometry.
The geometry is missing, since we only store abstract simplices and not
something like coordinate simplices.

This was purposefully done, because we want to separate the topology from the geometry.
This allows us to switch between a coordinate-based embedded geometry and a coordinate-free
intrinsic geometry based on a Riemannian metric.


== Coordinate-Based Simplicial Euclidean Geometry

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
store its coordinate exactly once. This is contrast to using a list of `SimplexCoords`, for
which there would have been many duplicate coordinates, since the vertices are
shared by many simplices. So separating topology and geometry is always very natural
even in the case of the typical coordinate-based geometry.

=== Coordinate Function Functors & Barycentric Quadrature

Before differential geometry, calculus was done on euclidean space $RR^n$
instead of on abstract manifolds @frankel:diffgeo. Euclidean space always has global coordinates.
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
in general @frankel:diffgeo. But instead we can rely on ambient coordinates $x in RR^N$, if an
embedding is available, and work with functions defined on them.

One common use-case that is also relevant to us for such a point-evaluable
functor is numerical integration of a real valued function via numerical
quadrature @hiptmair:numpde.
Since we are doing only 1st order FEEC, we restrict ourselves to
quadrature rules of order 1, that integrate affine-linear functions exactly.
The simplest of these that work on arbitrary-dimensional simplices is
the barycentric quadrature rule, that just does a single evaluation
of the function at the barycenter of the simplex and multiplies this
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
  avec(x) in RR^N |-> omega_avec(x) (diff/(diff x^1),...,diff/(diff x^n)) in RR
$


== Metric-Based Riemannian Geometry

The coordinate-based Euclidean geometry we've seen so far, is what
is commonly used in almost all FEM implementations.
In our implementation we go one step further and abstract away
the coordinates of the manifold and instead make use of 
coordinate-free Riemannian geometry @frankel:diffgeo for all of our FE algorithms.
All FE algorithms only depend on this geometry representation and cannot operate
directly on the coordinate-based geometry. Instead one should always derive a
coordinate-free representation from the coordinate-based one.
Most of the time one starts with a coordinate-based representation
that has been constructed by some mesh generator like gmsh @GmshPaper2009 and
then one computes the intrinsic geometry and forgets about the coordinates.
Our library supports this exactly this functionality.

=== Riemannian Metric

Riemannian geometry is an intrinsic description of the manifold,
that doesn't need an ambient space at all. It relies purely on a structure
over the manifold called a *Riemannian metric* $g$ @frankel:diffgeo.

It is a continuous function over the whole manifold, which at each point $p$
gives us an inner product $g_p: T_p M times T_p M -> RR^+$ on the tangent space
$T_p M$ at this point $p$.
It is the analog to the standard euclidean inner product (dot product) in
euclidean geometry. The inner product on tangent vectors allows one to measure
lengths $norm(v)_g = sqrt(g(v, v))$ angles $phi(v, w) = arccos((g_p (v, w))/(norm(v)_g norm(w)_g))$.
While euclidean space is flat and the inner product is the same everywhere, a
manifold is curved in general and therefore the inner product changes from point
to point, reflecting the changing geometry.


Given a basis $diff/(diff x^1),...,diff/(diff x^n)$ of the tangent space
$T_p M$ at a point $p$, induced by a chart map
$phi: p in U subset.eq M |-> (x_1,...,x_n)$, the inner product $g_p$ can be
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
In the context of Riemannian geometry this is called a *metric tensor* @frankel:diffgeo.

The inverse metric $g^(-1)_p$ at a point $p$ provides an inner product
$g^(-1)_p: T^*_p M times T^*_p M -> RR^+$ on the
cotangent space $T^*_p M$. It can be obtained by computing the inverse
Gramian matrix $amat(G)^(-1)$, which is then a new Gramian matrix representing
the inner product on the dual basis of covectors.
$
  amat(G)^(-1) = [g(dif x^i,dif x^j)]_(i,j=1)^(n times n)
$
The inverse metric is very important for us, since differential forms are
covariant tensors, therefore they are measured by the inverse metric tensor @frankel:diffgeo.

Computing the inverse is numerically unstable and instead it would
be better to rely on matrix factorization to do computation
involving the inverse metric @hiptmair:numcse. However this quickly becomes
intractable. For this reason we chose here to rely on the directly
computed inverse matrix nonetheless.

We introduce a struct to represent the Riemannian metric at a particular point
as the Gramian matrix.

```rust
/// A Gram Matrix represents an inner product expressed in a basis.
#[derive(Debug, Clone)]
pub struct Gramian {
  /// S.P.D. matrix
  matrix: na::DMatrix<f64>,
}
impl Gramian {
  pub fn try_new(matrix: na::DMatrix<f64>) -> Option<Self> {
    matrix.is_spd().then_some(Self { matrix })
  }
  pub fn new(matrix: na::DMatrix<f64>) -> Self {
    Self::try_new(matrix).expect("Matrix must be s.p.d.")
  }
  pub fn new_unchecked(matrix: na::DMatrix<f64>) -> Self {
    if cfg!(debug_assertions) {
      Self::new(matrix)
    } else {
      Self { matrix }
    }
  }
  pub fn from_euclidean_vectors(vectors: na::DMatrix<f64>) -> Self {
    assert!(vectors.is_full_rank(1e-9), "Matrix must be full rank.");
    let matrix = vectors.transpose() * vectors;
    Self::new_unchecked(matrix)
  }
  /// Orthonormal euclidean metric.
  pub fn standard(dim: Dim) -> Self {
    let matrix = na::DMatrix::identity(dim, dim);
    Self::new_unchecked(matrix)
  }

  pub fn matrix(&self) -> &na::DMatrix<f64> { &self.matrix }


  pub fn dim(&self) -> Dim { self.matrix.nrows() }
  pub fn det(&self) -> f64 { self.matrix.determinant() }
  pub fn det_sqrt(&self) -> f64 { self.det().sqrt() }

  pub fn inverse(self) -> Self {
    let matrix = self
      .matrix
      .try_inverse()
      .expect("Symmetric Positive Definite is always invertible.");
    Self::new_unchecked(matrix)
  }
}

/// Inner product functionality directly on the basis.
impl Gramian {
  pub fn basis_inner(&self, i: usize, j: usize) -> f64 { self.matrix[(i, j)] }
  pub fn basis_norm_sq(&self, i: usize) -> f64 { self.basis_inner(i, i) }
  pub fn basis_norm(&self, i: usize) -> f64 { self.basis_norm_sq(i).sqrt() }
  pub fn basis_angle_cos(&self, i: usize, j: usize) -> f64 {
    self.basis_inner(i, j) / self.basis_norm(i) / self.basis_norm(j)
  }
  pub fn basis_angle(&self, i: usize, j: usize) -> f64 { self.basis_angle_cos(i, j).acos() }
}
impl std::ops::Index<(usize, usize)> for Gramian {
  type Output = f64;
  fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
    &self.matrix[(i, j)]
  }
}
```

```rust
/// Inner product functionality directly on any element.
impl Gramian {
  pub fn inner(&self, v: &na::DVector<f64>, w: &na::DVector<f64>) -> f64 {
    (v.transpose() * self.matrix() * w).x
  }
  pub fn inner_mat(&self, v: &na::DMatrix<f64>, w: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    v.transpose() * self.matrix() * w
  }
  pub fn norm_sq(&self, v: &na::DVector<f64>) -> f64 {
    self.inner(v, v)
  }
  pub fn norm_sq_mat(&self, v: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    self.inner_mat(v, v)
  }
  pub fn norm(&self, v: &na::DVector<f64>) -> f64 {
    self.inner(v, v).sqrt()
  }
  pub fn norm_mat(&self, v: &na::DMatrix<f64>) -> na::DMatrix<f64> {
    self.inner_mat(v, v).map(|v| v.sqrt())
  }
  pub fn angle_cos(&self, v: &na::DVector<f64>, w: &na::DVector<f64>) -> f64 {
    self.inner(v, w) / self.norm(v) / self.norm(w)
  }
  pub fn angle(&self, v: &na::DVector<f64>, w: &na::DVector<f64>) -> f64 {
    self.angle_cos(v, w).acos()
  }
}
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
into an ambient space $RR^N$ @frankel:diffgeo.
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

== Simplicial Riemannian Geometry & Regge Calculus

We have now discussed the of Riemannian geometry in general.
Now we want to focus on the special case of simplicial geometry,
where the Riemannian metric is defined on our mesh.

We have seen with our coordinate simplices that our geometry is piecewise-flat
over the cells. This means that our metric is constant over each cell
and changes only from one cell to another.

This piecewise-constant metric over the simplicial mesh is known as the *Regge
metric* @regge and comes from Regge calculus @regge, a theory for numerical general relativity
that is about producing simplicial approximations of spacetimes that are
solutions to the Einstein field equation.


=== Deriving the Metric from Coordinate Simplices

Our coordinate simplices are an immersion of an abstract simplex
and as such, we can compute the corresponding constant metric tensor on it.
We've seen that the local metric tensor is exactly the Gramian of the Jacobian of the immersion.
The Jacobian of our affine-linear transformation, is the matrix $amat(E)$,
that has the spanning vectors as columns.
From a different perspective the spanning vectors just constitute our chosen basis of the tangent space
and therefore it's Gramian w.r.t. the Euclidean inner product is the metric tensor.
$
  amat(G) = amat(E)^transp amat(E)
$
```rust
impl SimplexCoords {
  pub fn metric_tensor(&self) -> Gramian {
    Gramian::from_euclidean_vectors(self.spanning_vectors())
  }
}
```

=== Simplex Edge Lengths

We have seen how to derive the metric tensor from an embedding,
but of course the metric is independent of the specific coordinates.
The metric is invariant under isometric transformations, such
as translations, rotations and reflections @frankel:diffgeo.
This begs the question, what the minimal geometric information necessary is
to derive the metric.
It turns out that, while vertex coordinates are over-specified, edge lengths
in contrast are exactly the required information.
Edge lengths are invariant under isometric transformations.
This is known from *Regge Caluclus* @regge.

Instead of giving all vertices a global coordinate, as one would do in extrinsic
geometry, we just give each edge in the mesh a positive length. Just knowing
the lengths doesn't tell you the positioning of the mesh in an ambient space but
it's enough to give the whole mesh it's piecewise-flat geometry.

Mathematically this is just a function on the edges to the positive real numbers.
$
  d: Delta_1 (mesh) -> RR^+
$
that gives each edge $e in Delta_1 (mesh)$ a positive length $l_e in RR^+$. \
We denote the edge length between vertices $i$ and $j$ by $d_(i j)$.

```rust
#[derive(Debug, Clone)]
pub struct SimplexLengths {
  lengths: na::DVector<f64>,
  dim: Dim,
}
```

```rust
pub fn new(lengths: na::DVector<f64>, dim: Dim) -> Self {
  assert_eq!(lengths.len(), nedges(dim), "Wrong number of edges.");
  let this = Self { lengths, dim };
  assert!(
    this.is_coordinate_realizable(),
    "Simplex must be coordinate realizable."
  );
  this
}
pub fn new_unchecked(lengths: na::DVector<f64>, dim: Dim) -> Self {
  if cfg!(debug_assertions) {
    Self::new(lengths, dim)
  } else {
    Self { lengths, dim }
  }
}
pub fn standard(dim: Dim) -> SimplexLengths {
  let nedges = nedges(dim);
  let lengths: Vec<f64> = (0..dim)
    .map(|_| 1.0)
    .chain((dim..nedges).map(|_| SQRT_2))
    .collect();

  Self::new_unchecked(lengths.into(), dim)
}
pub fn from_coords(coords: &SimplexCoords) -> Self {
  let dim = coords.dim_intrinsic();
  let lengths = coords.edges().map(|e| e.vol()).collect_vec().into();
  // SAFETY: Edge lengths stem from a realization already.
  Self::new_unchecked(lengths, dim)
}
```

The edge lengths of a simplex and the metric tensor on it
both fully define the geometry uniquely @regge.
These two geometry representations are completely equivalent.
This means one can be derived from the other.


We can derive the edge lengths, from the metric tensor Gramian.
$
  d_(i j) = sqrt(amat(G)_(i i) + amat(G)_(j j) - 2 amat(G)_(i j))
$
```rust
pub fn from_metric_tensor(metric: &Gramian) -> Self {
  let dim = metric.dim();
  let length = |i, j| {
    (metric.basis_inner(i, i) + metric.basis_inner(j, j) - 2.0 * metric.basis_inner(i, j)).sqrt()
  };

  let mut lengths = na::DVector::zeros(nedges(dim));
  let mut iedge = 0;
  for i in 0..dim {
    for j in i..dim {
      lengths[iedge] = length(i, j);
      iedge += 1;
    }
  }

  Self::new(lengths, dim)
}
```


We can derive the metric, from the edge lengths by using the law
of cosines.
$
  amat(G)_(i j) = 1/2 (d_(0 i)^2 + d_(0 j)^2 - d_(i j)^2)
$
```rust
pub fn to_metric_tensor(&self) -> Gramian {
  let mut metric = Matrix::zeros(self.dim(), self.dim());
  for i in 0..self.dim() {
    metric[(i, i)] = self[i].powi(2);
  }
  for i in 0..self.dim() {
    for j in (i + 1)..self.dim() {
      let l0i = self[i];
      let l0j = self[j];

      let vi = i + 1;
      let vj = j + 1;
      let eij = lex_rank(&[vi, vj], self.nvertices());
      let lij = self[eij];

      let val = 0.5 * (l0i.powi(2) + l0j.powi(2) - lij.powi(2));

      metric[(i, j)] = val;
      metric[(j, i)] = val;
    }
  }
  Gramian::new(metric)
}
```


=== Realizability Conditions

While edge lengths provide a coordinate-free description of a simplex's
intrinsic geometry, not just any assignment of positive numbers to the edges
of an abstract simplex constitutes a valid Euclidean geometry. The assigned
lengths must satisfy certain consistency requirements, known collectively
as *realizability conditions* @distgeo. These conditions ensure that the
abstract simplex, endowed with these edge lengths, could actually be embedded
isometrically as a flat simplex in some Euclidean space $RR^N$. In essence, only
edge lengths for which a corresponding `SimplexCoords` object could exist are
considered valid for defining a Euclidean simplex geometry.

A fundamental necessary condition stems from the geometry of triangles. For
every 2-dimensional face (triangle) $sigma = [i, j, k]$ within the mesh, the
assigned edge lengths $d_(i j)$, $d_(j k)$, and $d_(i k)$ must satisfy the
standard *triangle inequalities*:
$
  d_(i j) + d_(j k) >= d_(i k) \
  d_(j k) + d_(i k) >= d_(i j) \
  d_(i k) + d_(i j) >= d_(j k) \
$
If these inequalities (particularly the strict versions) are violated for any
triangle, the metric tensor $amat(G)$ derived from these lengths via the law
of cosines will not be positive-definite. This would imply a degenerate or
pseudo-Riemannian metric rather than the proper Riemannian metric associated
with Euclidean geometry. If the edge lengths are derived from an actual
coordinate embedding (`SimplexCoords::from_coords`), the triangle inequalities
are automatically satisfied.

While necessary, the triangle inequalities alone are not sufficient for
dimensions $n > 2$. A more comprehensive check involves the squared distances
between all pairs of vertices. We can assemble these into the *Euclidean
distance matrix* (EDM) $amat(A)$ @distgeo, a symmetric matrix with zeros on
the diagonal:
$
  amat(A) = mat(
    0, d_12^2, d_13^2, dots.c, d_(1 n)^2;
    d_21^2, 0, d_23^2, dots.c, d_(2 n)^2;
    d_31^2, d_32^2, 0, dots.c, d_(3 n)^2;
    dots.v, dots.v, dots.v, dots.down, dots.v;
    d_(n 1)^2, d_(n 2)^2, d_(n 3)^2, dots.c, 0;
  )
$
where $d_(i j)$ is the length of the edge between vertex $i$ and vertex $j$.
```rust
pub fn distance_matrix(&self) -> na::DMatrix<f64> {
  let mut mat = na::DMatrix::zeros(self.nvertices(), self.nvertices());

  let mut idx = 0;
  for i in 0..self.nvertices() {
    for j in (i + 1)..self.nvertices() {
      let dist_sqr = self.lengths[idx].powi(2);
      mat[(i, j)] = dist_sqr;
      mat[(j, i)] = dist_sqr;
      idx += 1;
    }
  }
  mat
}
```

Building upon the EDM, the *Cayley-Menger matrix* @distgeo is constructed by
bordering the EDM with a row and column of ones:
$
  amat(C M) = mat(
    0, d_12^2, d_13^2, dots.c, d_(1 n)^2, 1;
    d_21^2, 0, d_23^2, dots.c, d_(2 n)^2, 1;
    d_31^2, d_32^2, 0, dots.c, d_(3 n)^2, 1;
    dots.v, dots.v, dots.v, dots.down, dots.v, dots.v;
    d_(n 1)^2, d_(n 2)^2, d_(n 3)^2, dots.c, 0, 1;
    1, 1, 1, dots.c, 1, 0;
  )
$
```rust
pub fn cayley_menger_matrix(&self) -> na::DMatrix<f64> {
  let mut mat = self.distance_matrix();
  mat = mat.insert_row(self.nvertices(), 1.0);
  mat = mat.insert_column(self.nvertices(), 1.0);
  mat[(self.nvertices(), self.nvertices())] = 0.0;
  mat
}
```

The determinant of this matrix, scaled by a dimension-dependent factor, is the
*Cayley-Menger determinant* $c m$ @distgeo:
$
  c m = ((-1)^(n+1))/((n!)^2 2^n) det (amat(C M))
$
```rust
impl SimplexLengths {
  pub fn cayley_menger_det(&self) -> f64 {
    cayley_menger_factor(self.dim()) * self.cayley_menger_matrix().determinant()
  }
}
pub fn cayley_menger_factor(dim: Dim) -> f64 {
  (-1.0f64).powi(dim as i32 + 1) / factorial(dim).pow(2) as f64 / 2f64.powi(dim as i32)
}
```

A fundamental result states that a set of edge lengths for an $n$-simplex
is realizable in Euclidean space $RR^N$ (for $N >= n$) if and only if the
Cayley-Menger determinant is non-negative: $c m >= 0$ @distgeo. A strictly
positive determinant indicates realizability in exactly $n$ dimensions
(non-degenerate), while a zero determinant implies the simplex is degenerate and
lies within an $(n-1)$-dimensional affine subspace.
```rust
pub fn is_coordinate_realizable(&self) -> bool {
  self.cayley_menger_det() >= 0
}
```

Furthermore, if the simplex is realizable ($c m >= 0$), its $n$-dimensional
volume is directly related to the Cayley-Menger determinant:
$
  vol = sqrt(c m)
$
This provides a way to compute the volume directly from edge lengths, without
needing explicit coordinates or the metric tensor.
```rust
pub fn vol(&self) -> f64 {
  self.cayley_menger_det().sqrt()
}
```

=== Global Geometry

Computationally we represent the edge lengths in a single struct
that has all lengths stored continuously in memory in a nalgebra vector @crate:nalgebra.
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


== Higher Order Geometry

The original PDE problem, before discretization, is posed on a smooth manifold, which
we then discretized in the form of a mesh.
This smooth manifold has a non-zero curvature everywhere in general @frankel:diffgeo.
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
of the mesh @hiptmair:numpde. In general any order polynomial elements can be used.
We will restrict ourselves completely to first-order elements in this thesis.
This approximation is sufficient for us, since it represents an
*admissible geometric variational crime* @holst:gvc: The order of our FE method coincides with
the order of our mesh geometry; both are linear 1st order.
This approximation doesn't affect the order of convergence in a negative way,
and therefore is admissible @holst:gvc.


== Mesh Generation and Loading

=== Tensor-Product Domain Meshing

Formoniq features a meshing algorithm for arbitrary dimensional
tensor-product domains. These domains are $n$-dimensional Cartesian products $[0,1]^n$
of the unit interval $[0,1]$. The simplicial skeleton will be computed based on
a Cartesian grid that subdivides the domain into $l^n$ many $n$-cubes, which are
generalizations of squares and cubes. Here $l$ is the number of subdivisions per axis.
To obtain a simplicial skeleton, we need to split each $n$-cube into non-overlapping $n$-simplices
that make up its volume. In 2D it's very natural to split a square into two triangles
of equal volume. This can be generalized to higher dimensions. The trivial
triangulation of a $n$-cube into $n!$ simplices is based on the $n!$ many permutations
of the $n$ coordinate axes @hiptmair:numpde.

The $n$-cube has $2^n$ vertices, which can all be identified using multiindices
$
  V = {0,1}^n = {(i_1,...,i_n) mid(|) i_j in {0,1}}
$
All $n!$ simplices will be based on this vertex base set. To generate the list
of vertices of the simplex, we start at the origin vertex $v_0 = 0 = (0)^n$.
From there we walk along axis directions from vertex to vertex.
For this we consider all $n!$ permutations of the basis directions $avec(e)_1,...,avec(e)_n$.
A permutation $sigma$ tells in which axis direction we need to walk next.
This gives us the vertices $v_0,...,v_n$ that forms a simplex.
$
  v_k = v_0 + sum_(i=1)^k avec(e)_sigma(i)
$

The algorithm in Rust looks like:
```rust
pub fn compute_cell_skeleton(&self) -> Skeleton {
  let nboxes = self.ncells();
  let nboxes_axis = self.ncells_axis();

  let dim = self.dim();
  let nsimplices = factorial(dim) * nboxes;
  let mut simplices: Vec<SortedSimplex> = Vec::with_capacity(nsimplices);

  // iterate through all boxes that make up the mesh
  for ibox in 0..nboxes {
    let cube_icart = linear_index2cartesian_index(ibox, nboxes_axis, self.dim());

    let vertex_icart_origin = cube_icart;
    let ivertex_origin =
      cartesian_index2linear_index(vertex_icart_origin.clone(), self.nvertices_axis());

    let basisdirs = IndexSet::increasing(dim);

    // Construct all $d!$ simplexes that make up the current box.
    // Each permutation of the basis directions (dimensions) gives rise to one simplex.
    let cube_simplices = basisdirs.permutations().map(|basisdirs| {
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

    simplices.extend(cube_simplices);
  }

  Skeleton::new(simplices)
}
```

We can note here, that the computational complexity of this algorithm, grows extremely fast
in the dimension $n$.
We have a factorial scaling $cal(O)(n!)$ (worse than exponential scaling $cal(O)(e^n)$)
for splitting the cube into simplices. Given $l$ subdivisions per dimensions, we have
$l^n$ cubes. So the overall computational complexity is dominated by $cal(O)(l^n n!)$,
a terrible result, due to the curse of dimensionality.
The memory usage is dictated by the same scaling law.

=== Gmsh Import

The formoniq manifold crate can read gmsh `.msh` files @GmshPaper2009 and turn them
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

