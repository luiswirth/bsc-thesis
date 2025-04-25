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
known as the Regge metric @regge. We also support the optional storage of global
vertex coordinates if an embedding is known.

== Coordinate Simplicies

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
$n$-simplex $sigma$ is defined by $n+1$ vertices $avec(v)_0, dots, avec(v)_n in
RR^N$ in a possibly higher-dimensional space $RR^N$ (where $N >= n$). The
simplex itself is the region bounded by the convex hull of these vertices:
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
pub fn dim_embedded(&self) -> Dim { self.vertices.nrows() }
pub fn is_same_dim(&self) -> bool { self.dim_intrinsic() == self.dim_embedded() }
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

The coordinate transformation $psi: avec(lambda) |-> avec(x)$ from intrinsic
barycentric coordinates $avec(lambda)$ to ambient Cartesian coordinates $avec(x)$
is given by:
$
  avec(x) = psi (avec(lambda)) = sum_(i=0)^n lambda^i avec(v)_i
$

This transformation can be implemented as:
```rust
pub fn bary2global<'a>(&self, bary: impl Into<BaryCoordRef<'a>>) -> EmbeddingCoord {
  let bary = bary.into();
  self
    .vertices
    .column_iter()
    .zip(bary.iter())
    .map(|(vi, &baryi)| baryi * vi)
    .sum()
}
```

The barycentric coordinate representation extends beyond the simplex boundaries
to the entire affine subspace spanned by the vertices. The condition $sum_(i=0)^n
lambda^i = 1$ must still hold, but only points $avec(x) in sigma$ strictly
inside the simplex have all $lambda^i in [0,1]$.
```rust
pub fn is_bary_inside<'a>(bary: impl Into<CoordRef<'a>>) -> bool {
  let bary = bary.into();
  assert_relative_eq!(bary.sum(), 1.0);
  bary.iter().all(|&b| (0.0..=1.0).contains(&b))
}
```

Outside the simplex $avec(x) in.not sigma$, some $lambda^i$ will be greater
than one or negative. The barycenter $avec(m) = 1/(n+1) sum_(i=0)^n avec(v)_i$
always has the special barycentric coordinate
$psi(avec(m)) = avec(lambda) = [1/n]^(n+1)$.
```rust
pub fn barycenter(&self) -> Coord {
  let mut barycenter = na::DVector::zeros(self.dim_embedded());
  self.vertices.column_iter().for_each(|v| barycenter += v);
  barycenter /= self.nvertices() as f64;
  barycenter
}
```

This coordinate system treats all vertices symmetrically, assigning a weight to
each. Consequently, with $n+1$ coordinates ($lambda^0, ..., lambda^n$) for an
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
coordinate system*. In this system, the coordinates $lambda^1, ..., lambda^n$
are unconstrained, providing a unique representation for every point in the
affine subspace via a bijection with $RR^n$.

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
  let mut mat = na::DMatrix::zeros(self.dim_embedded(), self.dim_intrinsic());
  let v0 = self.base_vertex();
  // Skip base vertex (index 0)
  for (i, vi) in self.vertices.coord_iter().skip(1).enumerate() {
    let v0i = vi - v0;
    mat.set_column(i, &v0i);
  }
  mat
}
```

These spanning vectors naturally relate to the reduced barycentric coordinate
system. We can rewrite the coordinate transformation $psi$ using $lambda^0 = 1 -
sum_(i=1)^n lambda^i$:
$
  avec(x)
  = sum_(i=0)^n lambda^i avec(v)_i 
  = (1 - sum_(i=1)^n lambda^i) avec(v)_0 + sum_(i=1)^n lambda^i avec(v)_i
  = avec(v)_0 + sum_(i=1)^n lambda^i (avec(v)_i - avec(v)_0)
  = avec(v)_0 + amat(E) avec(lambda)^-
$
This clearly shows the transformation is an affine map: a translation by
$avec(v)_0$ followed by the linear map represented by $amat(E)$ acting on the
local coordinates $avec(lambda)^-$. It maps the local coordinates to the
Cartesian coordinates $avec(x)$ within the affine subspace spanned by the
vectors $avec(e)_i$ originating at $avec(v)_0$.

We implement functions for this affine transformation:
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


// TODO: introduce chart map, reverse of parametrization
Reversing the transformation (finding local coordinates from global ones) is
more complex due to the potentially higher-dimensional ($N >= n$) ambient space.
The global coordinate point might not lie in the affine subspace.
And due to floating-point inaccuries it almost never exactly will.
This makes the linear system of the reverse transformation $avec(x) -
avec(v)_0 = amat(E) avec(lambda)^-$ underdetermined.
We use the Moore-Penrose pseudo-inverse $amat(E)^dagger$ @hiptmair:numcse,
typically computed via Singular Value Decomposition (SVD), to find the
least-squares solution of smallest norm:
$
  avec(lambda)^- = phi(avec(x))
  = amat(E)^dagger (avec(x) - avec(v)_0)
$

```rust
impl CoordSimplex {
  pub fn global2local<'a>(&self, global: impl Into<EmbeddingCoordRef<'a>>) -> LocalCoord {
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
      .unwrap()
  }
  pub fn pseudo_inverse(&self) -> Self {
    if self.dim_domain() == 0 {
      return Self::new(Vector::zeros(0), Matrix::zeros(0, self.dim_image()));
    }
    let linear = self.linear.clone().pseudo_inverse(1e-12).unwrap();
    let translation = &linear * &self.translation;
    Self { translation, linear, }
  }
}
```

The derivatives of the affine parametrization $avec(x)(avec(lambda)^-)$ reveal
that the spanning vectors $avec(e)_i$ form a natural basis for the tangent space
$T_p sigma$ at any point $p$ within the simplex $sigma$ @frankel:diffgeo. The
Jacobian of the affine map is precisely $amat(E)$:
$
  (diff avec(x))/(diff lambda^i) = avec(e)_i
  quad quad
  (diff avec(x))/(diff avec(lambda)^-) = amat(E)
$

Conversely, the total differentials of all the barycentric coordinate functions
$lambda^i$ can be computed using the pseudo-inverse. The rows of
$amat(E)^dagger$ correspond to the differentials $dif lambda^1, ..., dif lambda^n$.
These form a basis for the cotangent space $T^*_p sigma$, dual to the tangent
basis $avec(e)_1, ..., avec(e)_n$ @frankel:diffgeo. The differential $dif lambda^0$
is determined by the constraint $sum_i dif lambda^i = 0$.
$
  (diff avec(lambda)^-)/(diff avec(x)) = amat(E)^dagger
  quad quad
  dif lambda^i = (diff lambda^i)/(diff avec(x)) = (amat(E)^dagger)_(i,:) quad (i=1...n)
  quad quad
  dif lambda^0 = -sum_(i=1)^n dif lambda^i
  quad quad
  dif lambda^i (diff/(diff lambda^j)) = delta^i_j quad (i,j=1...n)
$
```rust
/// Total differential of barycentric coordinate functions in the rows(!) of
/// a matrix.
pub fn difbarys(&self) -> Matrix {
  let difs = self.inv_linear_transform();
  let mut difs = difs.insert_row(0, 0.0);
  difs.set_row(0, &-difs.row_sum());
  difs
}
```

The spanning vectors also define a parallelepiped. The volume of the $n$-simplex
is $1/n!$ times the $n$-dimensional volume of this parallelepiped. The signed volume
is computed using the determinant of the spanning vectors if $n=N$, or more
generally using the square root of the Gram determinant
$sqrt(det(amat(E)^transp amat(E)))$ @frankel:diffgeo.
```rust
impl SimplexCoords {
  pub fn det(&self) -> f64 {
    let det = if self.is_same_dim() {
      self.spanning_vectors().determinant()
    } else {
      Gramian::from_euclidean_vectors(self.spanning_vectors()).det_sqrt()
    };
    refsimp_vol(self.dim_intrinsic()) * det
  }
  pub fn vol(&self) -> f64 { self.det().abs() }
  pub fn is_degenerate(&self) -> bool { self.vol() <= 1e-12 }
}
pub fn ref_vol(dim: Dim) -> f64 { (factorial(dim) as f64).recip() }
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
```

As a consequence swapping to vertices in the simplex, will swap the orientation of the simplex,
by the properties of the determinant.

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
given dimension $n$, the $n$-dimensional reference simplex $sigma_"ref"^n$ is
defined in $RR^n$ (so $N=n$) using its local
coordinates as its global Cartesian coordinates:
$
  sigma_"ref"^n = {(lambda_1,dots,lambda_n) in RR^n mid(|) lambda_i >= 0, quad sum_(i=1)^n lambda_i <= 1 }
$
Its vertices are the origin $avec(v)_0 = avec(0)$ and the standard basis vectors
$avec(v)_i = nvec(e)_i$ for $i=1...n$. The spanning vectors are simply the
standard basis vectors $avec(e)_i = nvec(e)_i$, so $amat(E) = amat(I)_n$. The
metric tensor is the identity matrix $amat(G) = amat(I)_n$, representing the
standard Euclidean inner product. Its volume is $(n!)^(-1)$.

The affine map $phi$ from the reference simplex's local coordinates to its global
coordinates is the identity map. Any real, non-degenerate $n$-simplex $sigma$
can be viewed as the image of the reference $n$-simplex under the affine map
defined by $sigma$'s spanning vectors and base vertex:
$
  sigma = phi(sigma_"ref"^n)
$
The reference simplex acts as the parameter domain or chart for any real
simplex $sigma$. The map $phi$ is the parametrization, while its inverse $psi:
sigma -> sigma_"ref"^n$ (mapping global points on $sigma$ to local coordinates)
is the chart map. Barycentric coordinates, being intrinsic, remain invariant
under this affine transformation.


== Abstract Simplicies

After studying coordinate simplicies, the reader has hopefully developed
some intuitive understanding of simplicies. We will now shed the coordinates
and represent simplicies in a more abstract way, by just considering
them as a list of vertex indices, without any vertex coordinates.
A $n$-simplex $sigma$ is a $(n+1)$-tuple of natural numbers, which represent vertex
indices @hatcher:algtop.
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
we can call these even and odd permutations @hatcher:algtop.
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

Another important notion is the idea of a subsimplex or a face of a simplex @hatcher:algtop.

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


=== Subsequences

When considering the facets of an $n$-simplex, there are multiple
permutations with the same set of vertices. It would be nice
to instead have only one permutation per subset of vertices.
For this we can instead consider the subsequences of the original simplex.
This then also preservers the vertex order.

We have some methods to check whether a simplex is a subsequence of another,
based on a naive subsequence check algorithm.
```rust
pub fn is_subsequence_of(&self, sup: &Self) -> bool {
  let mut sup_iter = sup.iter();
  self.iter().all(|item| sup_iter.any(|x| x == item))
}
pub fn is_supersequence_of(&self, other: &Self) -> bool {
  other.is_subsequence_of(self)
}
```

We also have a method for generating all $k$-subsimplicies that
are subsequencess of a $n$-simplex. For this we generate all $k+1$-subsequences
of the original $n+1$ vertices.
We use here the implementation of provided by the itertools crate.
```rust
pub fn subsequences(&self, sub_dim: Dim) -> impl Iterator<Item = Self> {
  itertools::Itertools::combinations(self.clone().into_iter(), sub_dim + 1).map(Self::from)
}
```
This implementation is nice, since it provides the subsequences in a lexicographical
order w.r.t. the local indices.
If the original simplex was sorted, then the subsequences are truly lexicographically ordered even
w.r.t. the global indices.

A very standard operation is to generate all subsequences simplicies of the standard simplex.
We call these the standard subsimplicies.
```rust
pub fn standard_subsimps(dim_cell: Dim, dim_sub: Dim) -> impl Iterator<Item = Simplex> {
  Simplex::standard(dim_cell).subsequences(dim_sub)
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

We can also go the other directions and generate the supersequences of a given simplex,
if we are given a root simplex that has both the original simplex and it's subsequences as subsequences.
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

There is a special operation related to subsequence simplicies, called the
boundary operator @hatcher:algtop.
The boundary operator can be applied to any $n$-simplex $sigma in NN^(n+1)$ and
is the defined as.
$
  diff sigma = sum_i (-1)^i [v_0,dots,hat(v)_i,dots,v_n]
$
On the left we have a formal sum of simplicies.
This formal sum consists of all the $(n-1)$-subsequences of a $n$-simplex,
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
  self.subsequences(self.dim() - 1).map(move |simp| {
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
$n$-manifold @hatcher:algtop.
If two simplicies share the same vertices, then these are connected, by either
being adjacent or by being incident.
This makes the topology purely combinatorial.

In the following sections we study this simplicial topology and implement data
structures and algorithms related to it.

#v(1cm)

To define the topology of our simplicial $n$-manifold, we just need
to store the $n$-simplicies that make it up.
This defines the topology of the mesh at the top-level.
We call such a collection of $n$-simplicies that share the same vertices a $n$-skeleton @hatcher:algtop.

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

A $n$-skeleton alone doesn't suffice as data structure for our FEEC implementation @douglas:feec-book,
since it is missing the topology of the lower-dimensional subsimplicies of our cells.
But our FE basis functions are associated with these subsimplicies, so we need to represent them.

The skeleton only stores the top-level simplicies $Delta_n (mesh)$, but our FEM library
also needs to reference the lower-level simplicies $Delta_k (mesh)$, since these are also
also mesh entities on which the DOFs of our FE space live.


Enter the simplicial complex @hatcher:algtop. It stores not only the top-level cells, but also all
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
represent more general topological spaces beyond manifolds @hatcher:algtop.
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
to a $n$-ball @hatcher:algtop.


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
cell-skeleton. For this we generate the subsequences of all lengths of the cells.
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
      for sub in cell.subsequences(dim_sub) {
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
from the $k$-skeleton to the $(k-1)$-skeleton @hatcher:algtop.
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
are also contained in the mesh, which are exactly all the supersequence simplicies.
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
store it's coordinate exactly once. This is contrast to use a list of `SimplexCoords`, for
which there would have been many duplicate coordinates, since the vertices are
shared by many simplicies. So separating topology and geometry is always very natural
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

One common use-case that is also relevant to us for a such a point-evaluable
functor is numerical integration of a real valued function via numerical
quadrature @hiptmair:numpde.
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
  avec(x) in RR^N |-> omega_avec(x) (diff/(diff x^1),dots,diff/(diff x^n)) in RR
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

This is called a Gram matrix or Gramian @hiptmair:numcse and is the discretization of a
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
invovling the inverse metric @hiptmair:numcse. However this quickly becomes
intractable. For this reason we chose here to rely on the directly
computed inverse matrix nontheless.

We introduce a struct to represent the Riemannian metric at a particular point
as the Gramian matrix.

```rust
/// A Gram Matrix represent an inner product expressed in a basis.
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

  pub fn matrix(&self) -> &na::DMatrix<f64> {
    &self.matrix
  }
  pub fn dim(&self) -> Dim {
    self.matrix.nrows()
  }
  pub fn det(&self) -> f64 {
    self.matrix.determinant()
  }
  pub fn det_sqrt(&self) -> f64 {
    self.det().sqrt()
  }
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
  pub fn basis_inner(&self, i: usize, j: usize) -> f64 {
    self.matrix[(i, j)]
  }
  pub fn basis_norm_sq(&self, i: usize) -> f64 {
    self.basis_inner(i, i)
  }
  pub fn basis_norm(&self, i: usize) -> f64 {
    self.basis_norm_sq(i).sqrt()
  }
  pub fn basis_angle_cos(&self, i: usize, j: usize) -> f64 {
    self.basis_inner(i, j) / self.basis_norm(i) / self.basis_norm(j)
  }
  pub fn basis_angle(&self, i: usize, j: usize) -> f64 {
    self.basis_angle_cos(i, j).acos()
  }
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

== Simplicial Riemannian Geometry & Regge Metric

We have now discussed the of Riemannian geometry in general.
Now we want to focus on the special case of simplicial geometry,
where the Riemannian metric is defined on our mesh.

We have seen with our coordinate simplicies that our geometry is piecewise-flat
over the cells. This means that our metric is constant over each cell
and changes only from one cell to another.

This piecewise-constant metric over the simplicial mesh is known as the *Regge
metric* @regge and comes from Regge calculus @regge, a theory for numerical general relativity
that is about producing simplicial approximations of spacetimes that are
solutions to the Einstein field equation.


=== Deriving the Regge Metric from Coordinate Simplicies

Our coordinate simplicies are an immersion of an abstract simplex
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

We have seen how to derive the Regge metric from coordinates,
but of course the metric is independent of the specific coordinates.
The metric is invariant under isometric transformations, such
as translations, rotations and reflections @frankel:diffgeo.
This begs the question, what the minimal geometric information necessary is
to derive the metric.
It turns out that, while vertex coordinates are over-specified, edge lengths
in contrast are exactly the required information @regge.
Edge lengths are also invariant under isometric transformations.

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
    "Simplex must be coordiante realizable."
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

The Regge metric on a single simplex and it's edge lengths
are exactly equivalent and both define the geometry uniquely @regge.
This means one can be derived from the other.


We can derive the edge lengths, from the Regge metric Gramian.
$
  d_(i j) = sqrt(amat(G)_(i i) + amat(G)_(j j) - 2 amat(G)_(i j))
$
```rust
pub fn from_regge_metric(metric: &Gramian) -> Self {
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


We can derive the Regge metric Gramian, from the edge lengths by using the law
of cosines.
$
  amat(G)_(i j) = 1/2 (d_(0 i)^2 + d_(0 j)^2 - d_(i j)^2)
$
```rust
pub fn into_regge_metric(&self) -> Gramian {
  let mut metric_tensor = na::DMatrix::zeros(self.dim(), self.dim());
  for i in 0..self.dim() {
    metric_tensor[(i, i)] = self[i].powi(2);
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

      metric_tensor[(i, j)] = val;
      metric_tensor[(j, i)] = val;
    }
  }
  Gramian::try_new(metric_tensor).expect("Edge Lengths must be coordinate realizable.")
}
```


=== Realizability Conditions


HIPTMAIR TODO: die Kantenlängen müssen die Dreiecksungleichung erfüllen, damit sie eine lokal-konstante Riemannsche Metrik induzieren. Ansonsten erhält man eine pseudo-Riemannsche Metrik

Of course not all possible length assignment are valid. They need to fulfill certain
critieria that evolve around the possibility of realizing these edge lengths
as a real euclidean simplex. These are so called *Realizability Conditions* @distgeo.
Only edge lengths for which a coordinate simplex exists that actually
produces these edge lengths are valid.

One important condition on these edge lengths, is that they fulfill the
triangle inequality: \
All 2-simplicies (triangles) $sigma = [i, j, k] in Delta_2 (mesh)$ in the mesh $mesh$,
must fulfill the usual triangle inequality
$
  d_(i j) + d_(j k) >= d_(i k) \
  d_(j k) + d_(i k) >= d_(i j) \
  d_(i k) + d_(i j) >= d_(j k) \
$
Otherwise our metric tensor is not positive-definite and we obtain a
degenerate pseudo-Riemannian metric instead of a proper Riemannian metric. If we start from
an immersed mesh, this is always the case.


*Euclidean distance matrix* @distgeo
$
  amat(A) = mat(
    0, d_12^2, d_13^2, dots.c, d_(1 n)^2;
    d_21^2, 0, d_23^2, dots.c, d_(2 n)^2;
    d_31^2, d_32^2, 0, dots.c, d_(3 n)^2;
    dots.v, dots.v, dots.v, dots.down, dots.v;
    d_(n 1)^2, d_(n 2)^2, d_(n 3)^2, dots.c, 0;
  )
$

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

*Cayley-Menger Matrix* @distgeo

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

*Cayley-Menger Determinant* @distgeo
$
  c m = ((-1)^(n+1))/((n!)^2 2^n) det (amat(C M))
$

```rust
impl SimplexLengths {
pub fn cayley_menger_det(&self) -> f64 {
  cayley_menger_factor(self.dim()) * self.cayley_menger_matrix().determinant()
}
pub fn cayley_menger_factor(dim: Dim) -> f64 {
  (-1.0f64).powi(dim as i32 + 1) / factorial(dim).pow(2) as f64 / 2f64.powi(dim as i32)
}
```

Realizability is equivalent to $c m >= 0$ @distgeo.

```rust
pub fn is_coordinate_realizable(&self) -> bool {
  self.cayley_menger_det() >= 0.0
}
```

If the simplex is realizable, then the positive volume of it is:
$
  vol = sqrt(c m)
$

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
To obtain a simplicial skeleton, we need to split each $n$-cube into non-overlapping $n$-simplicies
that make up it's volume. In 2D it's very natural to split a square into two triangles
of equal volume. This can be generalized to higher dimensions. The trivial
triangulation of a $n$-cube into $n!$ simplicies is based on the $n!$ many permutations
of the $n$ coordinate axes @hiptmair:numpde.

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

