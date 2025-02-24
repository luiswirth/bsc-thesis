#import "setup-math.typ": *
#import "layout.typ": *
#import "setup.typ": *

= Software Design & Implementation Choices

The product of this thesis is the implementation of a FEEC library, which
has the name formoniq.
Formoniq is a suite of multiple libraries that have a focus on modularity.

== Why Rust? Safety, performance, and expressiveness!

- Modern programming language.
- Amazing Build System and Package Manager: Cargo
- Official Tooling: rustdoc, rustfmt, cargo
- Expressive language -> Strong type system, Traits, Enums
- Memory-Safety Proof-checker -> Ownership system and Borrowchecker
- fearless concurrency

== External libraries
=== nalgebra (linear algebra)

nalgebra and nalgebra-sparse

Unfortunatly the rust sparse linear algebra ecosystem is rather immature.

=== PETSc & SLEPc (solvers)

Sparse matrix direct solvers

eigensolvers

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

Thanks to Rust's amazing build system package manager cargo, (in stark contrast to Make and CMake)
the setup of a rather complicated project such as ours, is extremly simple.
We rely on a Cargo workspace to organize the various parts of our library ecosystem.

We have multiple crates (libraries):
- multi-index
- manifold
- exterior
- whitney
- formoniq

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
Applications in simplices and exterior algebra

= Topology & Geometry of Simplicial Riemannian Manifolds

In this chapter we will develop a mesh data structure for our finite element library.
It will store the topological and geometrical properties of our discrete PDE domain.
This will be in the form of a discretized Riemannian manifold, which will be represented
as a simplicial complex (with manifold topology) equipped with the Regge metric.
Our data structure will be special in two senses:
- It will support arbitrary dimensions.
- It will not only support extrinsinc euclidean geometry (based on some
  embedding, providing global coordinates), but also intrinsic Riemannian
  geometry.
We will restrict ourselves to simplicial piecewise-flat meshes.


Our mesh data structure needs to provide the following functionality:
- Container for mesh entities: Simplicies.
- Global numbering for unique identification of the entities.
- Entity iteration.
- Topological Information: Incidence and Adjacency.
- Geometric Information: Angles, Lengths and Volumes.


== The PDE Domain as a Manifold

PDEs are posed over a domain, a patch of space on which the solution, we are
searching for, lives.
Usually one thinks of this domain as just being a subset of euclidean space,
but the more general way is to describe the domain as a manifold.
This is the first way in which differential geometry enters our theory.

One can intuitively think of a manifold $Omega$ just as a subspace of euclidean space $RR^N$.
#footnote[
This is a valid perspective thanks to the Whitney Embedding Theorem, telling us that
every $n$-dimensional topological manifold can be embedded in $RR^(2n)$.
]
An easy example for a manifold would be the unit sphere $SS^2$ inside of $RR^3$.
It's the set of points ${ xv in RR^3 mid(|) norm(xv) = 1 }$, distance 1 from the origin.
So it's just the surface and not the ball, that has it's volume filled in.
This perspective is called the extrinsic view of a manifold, since
we study the manifold from the ambient space $RR^3$.
The manifold is placed into it through an embedding $Phi: Omega -> RR^N$.
Using it, we can study this manifold with the tools from extrinsic euclidean geometry.
If the manifold is curved the ambient space must be higher dimensional than the manifold itself.
For our example the ambient space is 3 dimensional, while the dimension of the manifold is just
2D, since the sphere is just a surface.

But the manifold is an object in it's own right, that exists independent from the ambient space.
We can take an intrinsic view and study the manifold without ever referring to the surroundings.
This is the perspective that differential geometry takes and is the one we will be mainly using.
Here we forget about the embedding $Phi$ and study the manifold as an abstract mathematical object.

This is also the way in which Einstein's Theory of General Relativity describe the spacetime continuum.
Spacetime is a manifold of intrinsic dimension 4 and the theory doesn't say anything about
the ambient space in which the manifold is embeddable into. It might as well not exist!
Spacetime is purely intrinsic and there is no outside.

Let's dive into the intrinsic description of manifolds.
A manifold is a type of topological space. Topology tells us how our space is
connected and gives us a notion of closeness.
The topology being manifold, tells us that our space is at each point locally
homeomorphic to $n$-dimensional euclidean space. Here $n$ is the intrinsic dimension.
In the case of $SS^2$, we have $n=2$.
Intuitively this means that if you zoom close enough the space looks just like flat space.

Differentiable/Smooth structure?
Topology gives us a notion of continuity and makes our manifold have $C^0$ regualarity,
but in order to solve PDEs on manifolds, we need them to be differentiable.
For this we need an additional differentiable/smooth structure. Given this we are able
to do calculus on the manifold. This is done on the tangent spaces of the manifold.
These are vector spaces at each point of the manifold, which make the curvy manifold
look like a flat linear vector space locally.

Next to topological and differentiable structures, we also need geometrical information.
Geometry lets us measure angles, lengths, areas, volumes and so on.
This is not possible with differential topology alone.
In the extrinsic view, we do geometry using the global coordinates
$(x_1,dots,x_N)$ of the manifold, provided by the embedding $Phi: p |-> (x_1,dots,x_N)$.
These coordinates describe the position of the manifold in the ambient space.
Using coordinates all geometrical properties can easily be calculated, using euclidean geometry.

The intrinsic way to describe the geometry of the manifold is called Riemannian geometry.
For this we need to equipe our toplogical manifold with an additional structure called
a Riemannian metric. It allows us to measure angles, lengths, areas, etc in a fashion
that is completly independent of the ambient space and the embedding.
The Riemannian metric is an inner product on the tangent spaces that continuously varies
across the manifold. This inner product induces a norm on the tangent vectors, that
let's us measure their lengths and the inner product gives us angles. Using the metric
we can compute all geometric properties that one could compute with an embedding.
This turns our differentiable manifold into a Riemannian manifold.

In the most general sense our PDE domain is a piecewise smooth oriented and
bounded $n$-dimensional (pseudo) Riemannian manifold, $n in NN$ with a piecewise
smooth boundary.

Orientability is the property of a manifold that we can choose a consistent
orientation. Famous examples of manifolds that are not orientable are the möbius strip and the
klein bottle, both of which only have one side. So they are lacking a backside.

== The mesh as a discretized Riemannian manifold

Finite Element Techniques heavily rely on the existance of a mesh, a
discretization of the PDE domain. So our mesh is giong to be a discrete
variant of a Riemannian manifold. Our implementation will embrace the intrinsic
differential geometry view as much as possible. This will manifest in two specialities
of our mesh data structure: Firstly our implementation is dimensionality independent.
Instead of hardcoding the mesh to e.g. 3 dimensions the dimension will just be a property
chosen at runtime. Secondly we will not assign any global coordiantes to the geometric entities
of our mesh, but instead only rely on a discrete Riemannian metric, instead of an embedding.
All finite element calculations will solely rely on this metric.

== The Simplex

Finite Element methods benefit from their ability to work on unstructured meshes.
So instead of subdividing a domain into a regular grid FEM works on potentially very non-uniform
meshes. The simplest type of mesh that works for such non-uniform meshes are simplicial meshes.
In 2D these are triangular and in 3D we have meshes made from tetrahedron. These building blocks
need to be generalized to arbitrary dimensions for our implementation.

The generalization of triangles in 2D and tetrahedron in 3D to arbitray dimensions
is called a simplex.
There is a type of simplex for every dimension:
A 0-simplex is a vertex, a 1-simplex is an edge, a 2-simplex is a triangle and
a 3-simplex is a tetrahedon.
The idea here is that an $n$-simplex is the polytope with the fewest vertices
that lives in $n$ dimensions. It's the simplest $n$-polytope there is.
A $n$-simplex always has $n+1$ vertices and the simplex is the patch of space
bounded by the convex hull of the vertices.

=== The Reference Simplex

The most natural simplex to consider is the orthogonal simplex, basically a corner of a n-cube.
This simplex can be defined as it's own coordindate realisation as an actual convex hull of
some points.
$
  Delta_"ref"^n = {(lambda_1,dots,lambda_n) in RR^n mid(|) lambda_i >= 0, quad sum_(i=1)^n lambda_i <= 1 }
$

This is the reference simplex.
It has vertices $v_0 = avec(0)$ and $v_i = v_0 + avec(e)_(i-1)$.
Vertex 0 is special because it's the origin. The edges that include the origin
are the spanning edges. They are the standard basis vectors.
They give rise to an euclidean orthonormal tangent space basis. Which manifests
as a metric tensor that is equal to the identity matrix $amat(G) = amat(I)_n$.

One can extend these coordinates by one more
$
  lambda_0 = 1 - sum_(i=1)^n lambda_i
$
To obtain what are called the *barycentric coordinates* ${lambda_i}_(i=0)^n$.

Barycentric coordinates exist for all $k$-simplicies.\
They are a coordinate system relative to the simplex.\
Where the barycenter of the simplex has coordinates $(1/k)^k$.\
$
  x = sum_(i=0)^k lambda^i (x) space v_i
$
with $sum_(i=0)^k lambda_i(x) = 1$ and inside the simplex $lambda_i (x) in [0,1]$ (partition of unity).

$
  lambda^i (x) = det[v_0,dots,v_(i-1),x,v_(i+1),dots,v_k] / det[x_0,dots,v_k]
$

Linear functions on simplicies can be expressed as
$
  u(x) = sum_(i=0)^k lambda^i (x) space u(v_i)
$


By applying an affine linear transformation to the reference simplex, we can obtain
any other coordinate realization simplex.

If we forget about coordinates, we can obtain any metric simplex by applying
a linear transformation to the standard simplex.

Just as for a coordinate-based geometry the coordinates of the vertices are sufficent
information for the full geometry of the simplex,
the edge lengths of a simplex are sufficent information for the full information of
a coordinate-free simplex.

== Simplicial Manifold Topology

We now construct a topological manifold that consists of simplicies.

In our coordinate-independent framework, we consider abstract simplicies,
which are just finite ordered sets of vertex indicies.
An example for a simplex simplex could be the list $[12, 27, 4]$, which
represent a 2-simplex, so a triangle with the three vertices 12, 27 and 4 in
this very order.
This makes our simplicies just combinatorial objects and these combinatorics
will be heart of our mesh datastructure.

In our code we will represent a simplex as
```rust
pub type VertexIdx = usize;
pub struct Simplex {
  vertices: Vec<VertexIdx>,
}
```

Abstract simplicies only convey topological information.
Which vertices are connected to each other to form this simplex and which
simplicies share the same vertices, giving us adjacency information.

=== Orientation

Simplicies have an orientation, that is determined through the order
of the vertices. For any simplex there are exactly two orientations, which
are opposites of each other.
In euclidean geometry the orientation is absolute and can be determined based
on the sign of determinant of the spanning vectors.

#table(
  columns: 4,
  stroke: fgcolor,
  $n$, [Simplex], [Positive], [Negative],
  $1$, [Edge], [left-to-right], [right-to-left],
  $2$, [Triangle], [counterclockwise], [clockwise],
  $3$, [Tetrahedron], [right-handed], [left-handed],
)

All permutations can be divided into two equivalence classes.
Given a reference ordering of the vertices of a simplex,
we can call these even and odd permutations, relative to this reference.
We call simplicies with even permutations, positively oriented and
simplicies with odd permutations negatively oriented.
For example we have: $[12, 27, 4] = -[27, 12, 4]$.

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
For this we consider all $n!$ permutations of the basis directions $vvec(e)_1,dots,vvec(e)_n$.
A permutation $sigma$ tells in which axis direction we need to walk next.
This gives us the vertices $v_0,dots,v_n$ that forms a simplex.
$
  v_k = v_0 + sum_(i=1)^k vvec(e)_sigma(i)
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
- The $n$-simplicies are called cells.
- The $(n-1)$-simplicies are called facets.
- The $0$-simplicies are called vertices.
- The $1$-simplicies are called edges.
- The $2$-simplicies are called triangles.
- The $3$-simplicies are called tetrahedrons.

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

Let's start with an informal outline of how homology works.

A topological space can have various holes of different dimensions.
A 1-dimensional hole (circular hole) is the kind of hole a circle has.
A sphere has a 2D hole (void).
The number of 0 dimensional holes, is just the number of connected components.

We want to count the numbers of these holes in a space.

The $k$-th Betti number $beta_k (X)$ is the number of $k$ dimensional holes of a topological space $X$.
So for the circle, we have $beta_0 (SS^1) = 1, beta_1 (SS^1) = 1$
and the sphere has $beta_0 (SS^2)=1, beta_1 (SS^2)=0, beta_2 (SS^2)=1$.
The torus has $beta_0 (TT^2) = 1, beta_1 (TT^2) = 2, beta_2 (TT^2) = 1$.

Homology is all about computing these numbers of any topological space.
There are various different homology theories, such as
singular homology and cellular homolgy, but the one relevant to us
is simplicial homology, since we are working with simplicial complexes.
So we study the connectivity of our simplicial complex.


More algebra is possible with simplicies.
For this we take the free albelian group generated by taking
a collection of simplicies as basis.
This allows us to express formal $ZZ$-linear combinations of simplicies,
such as $[0,1] - [1,2] + [2,3]$

We first define what a $k$-chain is.
A $k$-chain is a formal $ZZ$-linear combination of $k$-simplicies in the simplicial complex.
We have a linear structure over $ZZ$, which is a ring but not a field.
Therefore this is not a vector space, but a free albelian group generated by the $k$-simplicies.
This is also gives rise to a space of $k$-chains. If we take all of these spaces together
for all $k<=n$, we obtain a graded algebraic strucuture containing all chains of the simplicial complex.

On this graded structure we can now introduce the boundary operator $diff$.
It is a graded $ZZ$-linear operator $diff = plus.big.circle_k diff_k$ of order -1 with each individual
$diff_k: Delta_k -> Delta_(k-1)$, being defined by
$
  diff_k: sigma = [sigma_0,dots,sigma_n] |-> sum_i (-1)^i [sigma_0,dots,hat(sigma)_i,dots,sigma_n]
$
where the hat $hat(sigma)_i$ indicates that vertex $sigma_i$ is omitted.

This is the standard textbook definition of the boundary opertator and we will also be
using it, but the sum sign here suggests an order for the boundary simplicies that
gives a reversed lexicographical ordering, relative to the input, which is not optimal
for implementation. So instead we reverse this order.

The boundary operator has the important property that $diff^2$ = $diff compose diff = 0$.
This can be easily shown with some algebra.
$
  diff^2 sigma &= diff_(n-1) diff_n sigma
  \ &= diff (sum_i (-1)^i [sigma_0,dots,hat(sigma)_i,dots,sigma_n])
  \ &= sum_i (-1)^i diff [sigma_0,dots,hat(sigma)_i,dots,sigma_n]
  \ &= sum_i (-1)^i sum_j (-1)^j [sigma_0,dots,hat(sigma)_i,dots,hat(sigma)_j,dots,sigma_n]
  \ &= sum_(i,j) (-1)^i (-1)^j [sigma_0,dots,hat(sigma)_i,dots,hat(sigma)_j,dots,sigma_n]
  \ &= sum_(i<j) (-1)^i (-1)^(j-1) [sigma_0,dots,hat(sigma)_i,dots,hat(sigma)_j,dots,sigma_n]
   + sum_(i>j) (-1)^i (-1)^j [sigma_0,dots,hat(sigma)_j,dots,hat(sigma)_i,dots,sigma_n]
  \ &= 0
$

Simplicial Chain Complex
$
  0 limits(<-) Delta_0 (mesh) limits(<-)^diff dots.c limits(<-)^diff Delta_n (mesh) limits(<-) 0
  \
  diff^2 = diff compose diff = 0
$

== Atlas and Differential Structure

Until now we only studied the manifold as a topological space.
We now start studying additional structure.
We start with coordinate charts and the atlas.

Since simplicies are flat, our manifold is piecewise-flat.

We define a reference chart, a homeomorphism between the reference simplex
$Delta_"ref"^n subset.eq RR^n$ and the real simplex $sigma_i$.

$
  phi_i: Delta_"ref"^n -> sigma_i
$

$
  phi_i (lambda_1,dots,lambda_n)
  &= v_0 +
  mat(
    |,  , |;
    v_1,dots.c,v_n;
    |,  , |;
  ) avec(lambda)
$

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

Our refrence charts induce a natural basis on each tangent space.
$
  (diff phi_i)/(diff lambda_j) = v_j - v_0 = e_j
$

So the edge vectors $e_j$ eminating from the origin vertex $v_0$
form a basis of the tangent space. This is very intuitive.

By lucky coincidence, the notation for edge vectors and basis vectors
just happens to be the same. Both times denoted by a $e_i$.

The element of the tangent space are the vectors with which we can do
vector calculus or in our case the more general exterior calculus.

Vectors are *contravariant* tensors.
In exterior algebra we will extend them to multivectors that
are rank $k$ fully contravariant tensors.

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
What we mean by this is that the finite element algorithms will only rely on the
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

== Exterior Algebra as Generalization of Vector Algebra


Vectors are a fundamental algebraic object thought at a high-school level.
They are very geometric in their nature and represent oriented magnitudes.
They represent oriented line segments and are in this sense 1 dimensional objects.

The idea can be generalized to oriented $k$-dimensional segments of a $n$-dimensional space.
In 3D for instance we have
- Vector
- Bivectors
- Trivectors

The $k$-th exterior algebra $wedgespace^k V$ over the vector space $V$ is
called the space of $k$-vectors.\

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


== Musical Isomorphism

There is a geometric connection between $k$-vectors and $k$-forms, through the
musical isomorphisms.
This defines two unary operators.
- Flat #flat to move from $k$-vector to $k$-form.
- Sharp #sharp to move from $k$-form to $k$-vector.
This is inspired by musical notation. It moves the tensor index down (#flat) and up (#sharp).
$
  v^flat = w |-> g(v, w)
  \
  omega^sharp = ?
$

== Inner product

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


== Hodge star operator

Computationally we are working in some basis. The following
formulas are relevant for the implementation.
They are written in tensor index notation and make
use of the einstein sum convention.

For multivectors we use the metric tensor. For multiforms we use the inverse metric tensor.

This is the formula for the hodge star of basis k-forms.
$
  hodge (dif x^(i_1) wedge dots.c wedge dif x^(i_k))
  = sqrt(abs(det[g_(a b)])) / (n-k)! g^(i_1 j_1) dots.c g^(i_k j_k)
  epsilon_(j_1 dots j_n) dif x^(j_(k+1)) wedge dots.c wedge dif x^(j_n)
$

Here with restricted to increasing indices $j_(k+1) < dots < j_n$
$
  hodge (dif x^(i_1) wedge dots.c wedge dif x^(i_k))
  = sqrt(abs(det[g_(a b)])) sum_(j_(k+1) < dots < j_n)
  g^(i_1 j_1) dots.c g^(i_k j_k)
  epsilon_(j_1 dots j_n) dif x^(j_(k+1)) wedge dots.c wedge dif x^(j_n)
$

For an arbitrary differential k-form $alpha$, we have
$
  hodge alpha = sum_(j_(k+1) < dots < j_n)
  (hodge alpha)_(j_(k+1) dots j_n) dif x^(j_(k+1)) wedge dots.c wedge dif x^(j_n)
$

$
  (hodge alpha)_(j_(k+1) dots j_n)
  = sqrt(det[g_(a b)]) / k!
  alpha_(i_1 dots i_k) g^(i_1 j_1) dots.c g^(i_k j_k) epsilon_(j_1 dots j_n)
$



Given a Riemannian metric $g$, we get an inner product on each fiber
$wedge.big^k T^*_p (Omega)$.


Computationally this is done using the basis and we compute an extended
gramian matrix for the inner product on $k$-forms using the determinant.
This can be further extended to an inner product on #strike[differential] $k$-forms
with basis $dif x_i_1 wedge dots wedge dif x_i_k$.
$
  inner(dif x_I, dif x_J) = det [inner(dif x_I_i, dif x_I_j)]_(i,j)^k
$

// Section with most math compared to code
= Exterior Calculus of Differential Forms



== Exterior Calculus as Generalization of Vector Calculus



You can think of $k$-vector field as a *density* of infinitesimal oriented $k$-dimensional.
The differential $k$-form is just a $k$-form field, which is the dual measuring object.

Exterior Calculus exclusively cares about multiform-fields and not really about
multivector-fields. This is because multiforms can naturally be defined as integrands.


An arbitrary differential form can be written as (with Einstein sum convention)
$
  alpha = 1/k!
  alpha_(i_1 dots i_k) dif x^(i_1) wedge dots.c wedge dif x^(i_k)
  = sum_(i_1 < dots < i_k) 
  alpha_(i_1 dots i_k) dif x^(i_1) wedge dots.c wedge dif x^(i_k)
$


Differential Forms are sections of the exterior cotangent bundle.
$
  Lambda^k (Omega) = Gamma (wedge.big^k T^* (Omega))
$

== Integration

WIKIPEDIA:
A differential k-form can be integrated over an oriented k-dimensional manifold.
When the k-form is defined on an n-dimensional manifold with n > k, then the
k-form can be integrated over oriented k-dimensional submanifolds. If k = 0,
integration over oriented 0-dimensional submanifolds is just the summation
of the integrand evaluated at points, according to the orientation of those
points. Other values of k = 1, 2, 3, ... correspond to line integrals, surface
integrals, volume integrals, and so on. There are several equivalent ways to
formally define the integral of a differential form, all of which depend on
reducing to the case of Euclidean space.


- $k$-dimensional ruler $omega in Lambda^k (Omega)$
- ruler $omega: p in Omega |-> omega_p$ varies continuously  across manifold according to coefficent functions.
- locally measures tangential $k$-vectors $omega_p: (T_p M)^k -> RR$
- globally measures $k$-dimensional submanifold $integral_M omega in RR$

$
  phi: [0,1]^k -> Omega
  quad quad
  M = "Image" phi
  \
  integral_M omega =
  limits(integral dots.c integral)_([0,1]^k) quad
  omega_(avec(phi)(t))
  ((diff avec(phi))/(diff t_1) wedge dots.c wedge (diff avec(phi))/(diff t_k))
  dif t_1 dots dif t_k
$

== Exterior Derivative

The exterior derivative unifies all the derivatives from vector calculus.
In 3D we have:

$
  grad &=^~ dif_0
  quad quad
  &&grad f = (dif f)^sharp
  \
  curl &=^~ dif_1
  quad quad
  &&curl avec(F) = (hodge dif avec(F)^flat)^sharp
  \
  div &=^~ dif_2
  quad quad
  &&"div" avec(F) = hodge dif hodge avec(F)^flat
$

- $dif_0$: Measures how much a 0-form (scalar field) changes linearly,
  producing a 1-form (line field).
- $dif_1$: Measures how much a 1-form (line field) circulates areally,
  producing a 2-form (areal field).
- $dif_2$: Measures how much a 2-form (areal flux field) diverges volumetrically,
  producing a 3-form (volume field).
  

Purely topological, no geometry.

== Stokes' Theorem

Stokes' theorem unifies the main theorems from vector calculus.

Gradient Theorem
$
  integral_C grad f dot dif avec(s) =
  phi(avec(b)) - phi(avec(a))
$

Curl Theorem (Ordinary Stokes' Theorem)
$
  integral.double_S curl avec(F) dot dif avec(S) =
  integral.cont_(diff S) avec(F) dot dif avec(s)
$

Divergence Theorem (Gauss theorem)
$
  integral.triple_V "div" avec(F) dif V =
  integral.surf_(diff V) avec(F) dot nvec(n) dif A
$


$
  integral_Omega dif omega = integral_(diff Omega) trace omega
$
for all $omega in Lambda^l_1 (Omega)$


== Leibniz Product rule
$
  dif (alpha wedge beta) = dif alpha wedge beta + (-1)^abs(alpha) alpha wedge dif beta
$

Using the Leibniz Rule we can derive what the exterior derivative of a 1-form
term $alpha_j dif x^j$ must be, if we interpret this term as a wedge $alpha_j
wedge dif x^j$ between a 0-form $alpha_j$ and a 1-form $dif x^j$.
$
  dif (alpha_j dif x^j)
  = dif (alpha_j wedge dif x^j)
  = (dif alpha_j) wedge dif x^j + alpha_j wedge (dif dif x^j)
  = (diff alpha_j)/(diff x^i) dif x^i wedge dif x^j
$

== Integration by parts
$
  integral_Omega dif omega wedge eta
  + (-1)^l integral_Omega omega wedge dif eta
  = integral_(diff Omega) omega wedge eta
$
for $omega in Lambda^l (Omega), eta in Lambda^k (Omega), 0 <= l, k < n − 1, l + k = n − 1$.


$
  integral_Omega dif omega wedge eta
  =
  (-1)^(k-1)
  integral_Omega omega wedge dif eta
  +
  integral_(diff Omega) "Tr" omega wedge "Tr" eta
$

$
  inner(dif omega, eta) = inner(omega, delta eta) + integral_(diff Omega) "Tr" omega wedge "Tr" hodge eta
$


== Hodge Star operator and $L^2$-inner product

This can be extended to an $L^2$-inner product on $Lambda^k (Omega)$
by integrating the pointwise inner product with respect to the volume
from $vol$ associated to $g$.

$
  (omega, tau) |-> inner(omega, tau)_(L^2 Lambda^k) :=
  integral_M inner(omega(p), tau(p))_p vol
  = integral_M omega wedge hodge tau
$

The Hodge star operator is a linear operator
$
  hodge: Lambda^k (Omega) -> Lambda^(n-k) (Omega)
$
s.t.
$
  alpha wedge (hodge beta) = inner(alpha, beta)_(Lambda^k) vol
  quad forall alpha in Lambda^k (Omega)
$
where $inner(alpha, beta)$ is the pointwise inner product on #strike[differential] $k$-forms
meaning it's a scalar function on $Omega$.\
$vol = sqrt(abs(g)) dif x^1 dots dif x^n$ is the volume form (top-level form $k=n$).

Given a basis for $Lambda^k (Omega)$, we can get an LSE by replacing $alpha$ with each basis element.\
This allows us to solve for $hodge beta$.\
For a inner product on an orthonormal basis on euclidean space, the solution is explicit and doesn't involve solving an LSE.

In general:\
- $hodge 1 = vol$
- $hodge vol = 1$

== Codifferential


Coderivative operator $delta: Lambda^k (Omega) -> Lambda^(k-1) (Omega)$
defined such that
$
  hodge delta omega = (-1)^k dif hodge omega
  \
  delta_k := (dif_(k-1))^* = (-1)^k space (hodge_(k-1))^(-1) compose dif_(n-k) compose hodge_k
$

For vanishing boundary it's the formal $L^2$-adjoint of the exterior derivative.


== Exact vs Closed

A fundamental fact about exterior differentiation is that $dif(dif omega) = 0$
for any sufficiently smooth differential form $omega$.

Under some restrictions on the topology of $Omega$ the converse is
also true, which is called the exact sequence property:

$
  omega in Lambda^k: quad
  dif omega = 0 => omega = dif eta
  quad beta_k = 0
$

$
  grad F = 0 &=> F = "const"
  quad beta_0 = 0
  \
  curl F = 0 &=> F = grad f
  quad beta_1 = 0
  \
  div F = 0  &=> F = curl A
  quad beta_2 = 0
$

$
  frak(H)^k = {omega | dif omega = 0}/{omega | dif eta = omega}
$

== Poincaré's lemma

For a contractible domain $Omega subset.eq RR^n$ every
$omega in Lambda^l_1 (Omega), l >= 1$, with $dif omega = 0$ is the exterior
derivative of an ($l − 1$)–form over $Omega$.

- Constant vector field has zero gradient
- Curlfree vector field has a scalar potential
- divergencefree vector field has a vector potential (take curl of it)

== De Rham Cohomology

There is a dual notion to homology, called cohomology.
The most important of which is going to be de Rham cohomology.
Which makes statements about the existance of the existance of anti-derivaties of
differential forms and differential forms that have derivative 0.
It will turn out that the homology of PDE domain and the cohomology
of the differential forms is isomorphic.

Okay let's formally define what homology is.
The main object of study is a chain complex.



This gives us a cell complex.

Chain Complex: Sequence of algebras and linear maps

$
  dots.c -> V_(k+1) ->^(diff_(k+1)) V_k ->^(diff_k) V_(k-1) -> dots.c
  quad "with" quad
  diff_k compose diff_(k+1) = 0
$

Graded algebra $V = plus.big.circle_k V_k$ with graded linear operator $diff = plus.big.circle_k diff_k$ of degree -1,
such that $diff compose diff = 0$.

$V_k$: $k$-chains \
$diff_k$: $k$-th boundary operator \
$frak(Z)_k = ker diff_k$: $k$-cycles \
$frak(B)_k = im diff_(k+1)$: $k$-boundaries \
$frak(H)_k = frak(Z)_k \/ frak(B)_k$: $k$-th homology space \

The main star of the show is the homology space $frak(H)_k$, which is a quotient
space of the $k$-cycles divided by the $k$-boundaries.

The dimension of the $k$-th homology space is equal to the $k$-th Betti numbers.
$
  dim frak(H)_k = B_k
$

Therefore knowing the homology space of a topological space gives us the information
about all the holes of the space.

Dual to homology there is also cohomology, which is basically just homology
on the dual space of $k$-chains, which are the $k$-cochains. These are functions
on the simplicies to the integeres $ZZ$.

The homology and cohomology are isomorphic.

Homology and Cohomology will be very important to the proper treatment of FEEC.

== De Rham Complex
== De Rham Cohomology
== Hodge Theory

Wikipedia:\
A method for studying the cohomology groups of a smooth manifold M using partial
differential equations. The key observation is that, given a Riemannian metric
on M, every cohomology class has a canonical representative, a differential form
that vanishes under the Laplacian operator of the metric. Such forms are called
harmonic.
built on the work of Georges de Rham on de Rham cohomology.

== De Rham Theorem

Singular cohomology with real coefficients is isomorphic to de Rham cohomology.

The de Rham map is important for us as discretization of differential forms.
It is the projection of differential $k$-forms onto $k$-cochains,
which are functions defined on the $k$-simplicies of the mesh.

= Discrete Differential Forms: Simplicial Cochains and Whitney Forms

Cochains are isomorphic to our 1st order FEEC basis functions.
The basis we will be working with is called the Whitney basis.
The Whitney space is the space of piecewise-linear (over the cells)
differential forms.

== Discrete Differential Forms

The discretization of differential forms on a mesh is of outmost importance.
Luckily the discretization is really simple in the case of 1st order FEEC, which
gives us the same discretization as in DEC.

A discrete differential $k$-form on a mesh is a $k$-cochain on this mesh.
So just a real-valued function $omega: Delta_k (mesh) -> RR$ defined on all
$k$-simplicies $Delta_k (mesh)$ of the mesh $mesh$.

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

== Discrete Exterior Derivative via Stokes' Theorem

The continuous exterior derivative does not reference the metric, which
makes it purely topological. The same should hold true for the discrete exterior derivative.

In discrete settings defined as coboundary operator, through Stokes' theorem.\
So the discrete exterior derivative is just the transpose of the boundary operator / incidence matrix.

$
  dif^k = diff_(k+1)^transp
$
Stokes' Theorem is fullfilled by definition.


Extension Trait in whitney crate.
```rust
pub trait ManifoldComplexExt {
  fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> SparseMatrix;
}
impl ManifoldComplexExt for Complex {
  /// $dif^k: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$
  fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> SparseMatrix {
    self.boundary_operator(grade + 1).transpose()
  }
}
```

The exterior derivative is closed in the space of Whitney forms, because of the de Rham complex.

The local (on a single cell) exterior derivative is always the same for any cell.
Therefore we can compute it on the reference cell.


== Cochain-Projection & Discretization


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


== Whitney-Interpolation & Reconstruction

```rust
/// Whitney Form on a coordinate complex.
///
/// Can be evaluated on local coordinates.
pub struct WhitneyForm<O: SetOrder> {
  cell_coords: SimplexCoords,
  associated_subsimp: Simplex<O>,
  difbarys: Vec<MultiForm>,
}
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

  /// The constant exterior derivative of the Whitney form.
  pub fn dif(&self) -> MultiForm {
    if self.grade() == self.dim() {
      return MultiForm::zero(self.dim(), self.grade() + 1);
    }
    let factorial = factorial(self.grade() + 1) as f64;
    let difbarys = self.difbarys.clone();
    factorial * MultiForm::wedge_big(difbarys).unwrap()
  }
}
impl<O: SetOrder> ExteriorField for WhitneyForm<O> {
  type Variance = variance::Co;
  fn dim(&self) -> Dim {
    self.cell_coords.dim_embedded()
  }
  fn grade(&self) -> ExteriorGrade {
    self.associated_subsimp.dim()
  }
  fn at_point<'a>(&self, coord_global: impl Into<CoordRef<'a>>) -> ExteriorElement<Self::Variance> {
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


== Whitney Forms and Whitney Basis
#v(1cm)

Whitney $k$-forms $cal(W) Lambda^k (mesh)$ are piecewise-constant (over cells $Delta_k (mesh)$)
differential $k$-forms.

The defining property of the Whitney basis is a from pointwise to integral
generalized Lagrange basis property (from interpolation):\
For any two $k$-simplicies $sigma, tau in Delta_k (mesh)$, we have
$
  integral_sigma lambda_tau = cases(
    +&1 quad &"if" sigma = +tau,
    -&1 quad &"if" sigma = -tau,
     &0 quad &"if" sigma != plus.minus tau,
  )
$

The Whitney $k$-form basis function live on all $k$-simplicies of the mesh $mesh$.
$
  cal(W) Lambda^k (mesh) = "span" {lambda_sigma : sigma in Delta_k (mesh)}
$

This is a true generalization of the Lagrange Space and it's Basis.

If we have a triangulation $mesh$, then the barycentric coordinate functions
can be collected to form the lagrange basis.

We can represent piecewiese-linear (over simplicial cells) functions on the mesh.
$
  u(x) = sum_(i=0)^N b^i (x) space u(v_i)
$


Fullfills Lagrange basis property basis.
$
  b^i (v_j) = delta_(i j)
$



There is a isomorphism between Whitney $k$-forms and cochains.\
Represented through the de Rham map (discretization) and Whitney interpolation:\
- The integration of each Whitney $k$-form over its associated $k$-simplex yields a $k$-cochain.
- The interpolation of a $k$-cochain yields a Whitney $k$-form.\


Whitney forms are affine invariant. \
Let $sigma = [x_0 dots x_n]$ and $tau = [y_0 dots y_n]$ and $phi: sigma -> tau$
affine map, such that $phi(x_i) = y_i$, then
$
  cal(W)[x_0 dots x_n] = phi^* (cal(W)[y_0 dots y_n])
$

The Whitney basis ${lambda_sigma}$ is constructed from barycentric coordinate functions ${lambda_i}$.

$
  lambda_(i_0 dots i_k) =
  k! sum_(l=0)^k (-1)^l lambda_i_l
  (dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k)
$

#align(center)[#grid(
  columns: 2,
  gutter: 10%,
  $
    cal(W)[v_0 v_1] =
    &-lambda_1 dif lambda_0
     + lambda_0 dif lambda_1 
    \
    cal(W)[v_0 v_1 v_2] =
    &+2 lambda_2 (dif lambda_0 wedge dif lambda_1) \
    &-2 lambda_1 (dif lambda_0 wedge dif lambda_2) \
    &+2 lambda_0 (dif lambda_1 wedge dif lambda_2) \
  $,
  $
    cal(W)[v_0 v_1 v_2 v_3] =
    - &6 lambda_3 (dif lambda_0 wedge dif lambda_1 wedge dif lambda_2) \
    + &6 lambda_2 (dif lambda_0 wedge dif lambda_1 wedge dif lambda_3) \
    - &6 lambda_1 (dif lambda_0 wedge dif lambda_2 wedge dif lambda_3) \
      &6 lambda_0 (dif lambda_1 wedge dif lambda_2 wedge dif lambda_3) \
  $
)]

From this definition we can easily derive the constant exterior derivative
of a whitney form!

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


== Whitney Forms

The DOFs of an element of the Whitney $k$-form space are the $k$-simplicies of the mesh.

$omega in cal(W) Lambda^k (mesh)$ has DOFs on $Delta_k (mesh)$

The Whitney space unifies and generalizes the Lagrangian, Raviart-Thomas and Nédélec
Finite Element spaces.
$
  cal(W) Lambda^0 (mesh) &=^~ cal(S)^0_1 (mesh) \
  cal(W) Lambda^1 (mesh) &=^~ bold(cal(N)) (mesh) \
  cal(W) Lambda^2 (mesh) &=^~ bold(cal(R T)) (mesh) \
$

The Whitney Subcomplex
$
  0 -> cal(W) Lambda^0 (mesh) limits(->)^dif dots.c limits(->)^dif cal(W) Lambda^n (mesh) -> 0
$

It generalizes the discrete subcomplex from vector calculus.
$
  0 -> cal(S)^0_1 (mesh) limits(->)^grad bold(cal(N)) (mesh) limits(->)^curl bold(cal(R T)) (mesh) limits(->)^div cal(S)^(-1)_0 (mesh) -> 0
$


= Finite Element Methods for Differential Forms

We have now arrived at the chapter talking about the
actual finite element library formoniq. \
Here we will derive and implement the formulas for computing the element matrices
of the various weak differential operators in FEEC.
Furthermore we implement the assembly algorithm that will give us the
final galerkin matrices.

== Sobolev Space of Differential Forms

$H Lambda^k (Omega)$ is the sobolev space of differential forms.
It is defined as the space of differential forms that have a square integrable
exterior derivative.

$
  H Lambda^k (Omega) = { omega in L^2 Lambda^k (Omega) mid(|) dif omega in L^2 Lambda^(k+1) (Omega) }
$

This is a very general definition that unifies the sobolev spaces known
from vector calculus.
In $RR^3$ we have the following isomorphisms.

$
  H Lambda^0 (Omega)
  &=^~
  H (grad; Omega)
  \
  H Lambda^1 (Omega)
  &=^~
  Hvec (curl; Omega)
  \
  H Lambda^2 (Omega)
  &=^~
  Hvec (div ; Omega)
$


== De Rham Complex of Differential Forms

These sobolev space together with their respective exterior derivatives
form a cochain complex, called the de Rham complex of differential forms.
$
  0 -> H Lambda^0 (Omega) limits(->)^dif dots.c limits(->)^dif H Lambda^n (Omega) -> 0
  \
  dif^2 = dif compose dif = 0
$


//#diagram(
//  edge-stroke: fgcolor,
//  cell-size: 15mm,
//  $
//    0 edge(->) &H(grad; Omega) edge(grad, ->) &Hvec (curl; Omega) edge(curl, ->) &Hvec (div; Omega) edge(div, ->) &L^2(Omega) edge(->) &0
//  $
//)

It generalizes the 3D vector calculus de Rham complex.

$
  0 -> H (grad; Omega) limits(->)^grad Hvec (curl; Omega) limits(->)^curl Hvec (div; Omega) limits(->)^div L^2(Omega) -> 0
  \
  curl compose grad = 0
  quad quad
  div compose curl = 0
$

== Variational Formulation & Element Matrix Computation

There are only a few variational differential operators that
exist in FEEC. These are all just variants of the exterior derivative and
the codifferential.


All bilinear forms that occur in the mixed weak formulation of Hodge-Laplacian
are just a variant of the inner product on Whitney forms.

$
  c(u, v) &= inner(delta u, v)_(L^2 Lambda^k (Omega)) = inner(u, dif v)_(L^2 Lambda^k (Omega)) \
$

$
  m^k (u, v) &= inner(u, v)_(L^2 Lambda^k (Omega)) \
  d^k (u, v) &= inner(dif u, v)_(L^2 Lambda^k (Omega)) \
  c^k (u, v) &= inner(u, dif v)_(L^2 Lambda^k (Omega)) \
  l^k (u, v) &= inner(dif u, dif v)_(L^2 Lambda^k (Omega)) \
$

After Galerkin discretization we arrive at these Galerkin matrices for our
four weak operators.
$
  amat(M)^k &= [inner(phi^k_i, phi^k_j)]_(i j) \
  amat(D)^k &= [inner(phi^k_i, dif phi^(k-1)_j)]_(i j) \
  amat(C)^k &= [inner(dif phi^(k-1)_i, phi^k_j)]_(i j) \
  amat(L)^k &= [inner(dif phi^k_i, dif phi^k_j)]_(i j) \
$


As we can see all bilinear forms are just inner product with a potential exterior derivative
on either argument. Since the exterior derivative is purely topological it only involves a
signed incidende matrix.

$
  amat(D)^k &= amat(M)^k amat(dif)^(k-1) \
  amat(C)^k &= (amat(dif)^(k-1))^transp amat(M)^k \
  amat(L)^k &= (amat(dif)^(k-1))^transp amat(M)^k amat(dif)^(k-1) \
$

We first define a element matrix provider trait
```rust
pub type ElMat = na::DMatrix<f64>;
pub trait ElMatProvider {
  fn row_grade(&self) -> ExteriorGrade;
  fn col_grade(&self) -> ExteriorGrade;
  fn eval(&self, geometry: &SimplexGeometry) -> ElMat;
}
```

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


For this reason we really just need a formula for the element matrix
of this inner product. This galerkin matrix is called the mass matrix
and the inner product could also be called the mass bilinear form.

$
  M = [inner(lambda_tau, lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))
$


=== Mass bilinear form

Integrating the equation over $Omega$ we get
$
  integral_Omega alpha wedge (hodge beta)
  = integral_Omega inner(alpha, beta)_(Lambda^k) vol
  = inner(alpha, beta)_(L^2 Lambda^k)
  quad forall alpha in Lambda^k (Omega)
$

This equation can be used to find the discretized weak Hodge star operator.\
The weak variational form of our hodge star operator is the mass bilinear form
$
  m(u, v) = integral_Omega hodge u wedge v
$
After Galerkin discretization we get the mass matrix for our discretized weak Hodge star operator
as the $L^2$-inner product on differential $k$-forms.
$
  amat(M)_(i j) = integral_Omega phi_j wedge hodge phi_i = inner(phi_j, phi_i)_(L^2 Lambda^k)
$

This is called the mass bilinear form / matrix, since for 0-forms, it really coincides with
the mass matrix from Lagrangian FEM.

The Hodge star operator captures geometry of problem through this inner product,
which depends on Riemannian metric.\
Let's see what this dependence looks like.

For (1st order) FEEC we have piecewise-linear differential $k$-forms with the
Whitney basis $lambda_sigma$.\
Therefore our discretized weak hodge star operator is the mass matrix, which is the Gramian matrix
on all Whitney $k$-forms.

$
  amat(M)^k = [inner(lambda_sigma_j, lambda_sigma_i)_(L^2 Lambda^k)]_(0 <= i,j < binom(n,k))
  = [inner(lambda_I, lambda_J)_(L^2 Lambda^k)]_(I,J in hat(cal(I))^n_k)
$


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

In Rust this is implemented as the following element matrix provider
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

Here we make use of the scalar mass element matrix


The following integral formula for powers of barycentric coordinate functions holds (NUMPDE):
$
  integral_K lambda_0^(alpha_0) dots.c lambda_n^(alpha_n) vol
  =
  n! abs(K) (alpha_0 ! space dots.c space alpha_n !)/(alpha_0 + dots.c + alpha_n + n)!
$
where $K in Delta_n, avec(alpha) in NN^(n+1)$.\
The formula treats all barycoords symmetrically.

For piecewise linear FE, the only relevant results are:
$
  integral_K lambda_i lambda_j vol
  = abs(K)/((n+2)(n+1)) (1 + delta_(i j))
$

$
  integral_K lambda_i vol = abs(K)/(n+1)
$



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

The Hodge-Laplacian operator generalizes the ordinary scalar Laplacian operator.

$
  Delta f = div grad f = hodge dif hodge dif f
$

=== Primal Strong form
$
  Delta u = f
$
with $u,f in Lambda^k (Omega)$

Hodge-Laplace operator
$
  Delta: Lambda^k (Omega) -> Lambda^k (Omega)
  \
  Delta = dif delta + delta dif
$


=== Primal Weak Form

The primal weak form cannot be implemented. It lacks the necessary regularity
to give a meaningful codifferential.


=== Mixed Strong Formulation

Given $f in Lambda^k$, find $(sigma,u,p) in (Lambda^(k-1) times Lambda^k times frak(h)^k)$ s.t.
$
  sigma - delta u &= 0
  quad &&"in" Omega
  \
  dif sigma + delta dif u &= f - p
  quad &&"in" Omega
  \
  tr hodge u &= 0
  quad &&"on" diff Omega
  \
  tr hodge dif u &= 0
  quad &&"on" diff Omega
  \
  u perp frak(h)
$


=== Mixed Weak Form

Given $f in L^2 Lambda^k$, find $(sigma,u,p) in (H Lambda^(k-1) times H Lambda^k times frak(h)^k)$ s.t.
$
  inner(sigma,tau) - inner(u,dif tau) &= 0
  quad &&forall tau in H Lambda^(k-1)
  \
  inner(dif sigma,v) + inner(dif u,dif v) + inner(p,v) &= inner(f,v)
  quad &&forall v in H Lambda^k
  \
  inner(u,q) &= 0
  quad &&forall q in frak(h)^k
$

=== Galerkin Mixed Weak Form
$
  sum_j sigma_j inner(phi^(k-1)_j,phi^(k-1)_i) - sum_j u_j inner(phi^k_j,dif phi^(k-1)_i) &= 0
  \
  sum_j sigma_j inner(dif phi^(k-1)_j,phi^k_i) + sum_j u_j inner(dif phi^k_j,dif phi^k_i) + sum_j p_j inner(eta^k_j,phi^k_i) &= sum_j f_j inner(psi_j,phi^k_i)
  \
  sum_j u_j inner(phi^k_j,eta^k_i) &= 0
$

$
  hodge sigma - dif^transp hodge u &= 0
  \
  hodge dif sigma + dif^transp hodge dif u + hodge H p &= hodge f
  \
  H^transp hodge u &= 0
$

$
  mat(
    hodge, -dif^transp hodge, 0;
    hodge dif, dif^transp hodge dif, hodge H;
    0, H^transp hodge, 0;
  )
  vec(sigma, u, p)
  =
  vec(0, hodge f, 0)
$

== Hodge-Laplace Eigenvalue Problem

=== Primal Strong Form
$
  (delta dif + dif delta) u = lambda u
$

=== Mixed Weak Form

Find $lambda in RR$, $(sigma, u) in (H Lambda^(k-1) times H Lambda^k \\ {0})$, s.t.
$
  inner(sigma, tau) - inner(u, dif tau) &= 0
  quad &&forall tau in H Lambda^(k-1)
  \
  inner(dif sigma, v) + inner(dif u, dif v) &= lambda inner(u,v)
  quad &&forall v in H Lambda^k
$


=== Galerkin Mixed Weak Form
$
  sum_j sigma_j inner(phi^(k-1)_j, phi^(k-1)_i) - sum_j u_j inner(phi^k_j, dif phi^(k-1)_i) &= 0
  \
  sum_j sigma_j inner(dif phi^(k-1)_j, phi^k_i) + sum_j u_j inner(dif phi^k_j, dif phi^k_i) &= lambda sum_j u_j inner(phi^k_j,phi^k_i)
$


$
  amat(M)^(k-1) vvec(sigma) - amat(C) vvec(u) = 0
  \
  amat(D) vvec(sigma) + amat(L) vvec(u) = lambda amat(M)^k vvec(u)
$


$
  mat(
    amat(M)^(k-1), -amat(C);
    amat(D), amat(L);
  )
  vec(vvec(sigma), vvec(u))
  =
  lambda
  mat(
    0,0;
    0,amat(M)^k
  )
  vec(vvec(sigma), vvec(u))
$

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


This is a symmetric indefinite sparse generalized matrix eigenvalue problem,
that can be solved by an iterative eigensolver such as Krylov-Schur.
This is also called a GHIEP problem.



// Probably move this to post-face
= Conclusion and Outlook

- Summary of key contributions
- Possible improvements and future work (e.g., efficiency, higher-order elements, more general manifolds)
- Broader impact (e.g., Rust in scientific computing, FEEC extensions)
- Discarded ideas and failed apporaches (generic dimensionality à la nalgebra/eigen)
