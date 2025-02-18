#import "setup-math.typ": *
#import "layout.typ": *
#import "setup.typ": *

= The Mesh


In this chapter we will develop a mesh data structure for our finite element library.
It will store the topological and geometrical properties of our discrete PDE domain.
This will be in the form of a discretized Riemannian manifold, which will be represented
as a simplicial complex (with manifold topology) equipped with the Regge metric.
Our data structure will be special in two senses:
- It will support arbitrary dimensions.
- It will use intrinsic Riemannian geometry, instead of an embedding providing global coordinates.
We will restrict ourselves to simplicial meshes.

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
This perspective is called the extrinsic view of a manifold.
$RR^3$ plays the role of an ambient space, in which the actual manifold lives in.
We can study this manifold using the tools from euclidean geometry.
This extrinsic perspective studies the manifold through it's embedding $Phi: Omega -> RR^N$.
The ambient space is usually higher dimensional than the manifold itself. For our example
the ambient space is 3 dimensional, while the dimension of the manifold is just
2D, since the sphere is just a surface.

But the manifold is an object in it's own right, that exists independent from the ambient space.
We can take an intrinsic view and study the manifold without ever referring to the surroundings.
This is the perspective that differential geometry takes and is the one we will be using.
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

== Simplicial Topology

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
For this we consider all $n!$ permutations of the basis directions $vvec(e_1),dots,vvec(e)_n$.
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

The skeleton only stores the top-level simplicies $Delta_n (mesh)$, but our FEM library
also needs to reference the lower-level simplicies $Delta_k (mesh)$, because DOFs
live on these.

For this reason these should also be stored in our mesh data structure.

Our topological mesh data structure needs to provide the following functionality:
- Container for mesh entities (simplicies).
- Global numbering for unique identification of the entities.
- Entity iteration.
- Incidence Information.
- Adjacency Information.


An abstract simplicial complex is a familiy of abstract simplicies,
which are set-like, that are closed under the subset operation.

More algebra is possible with simplicies.
For this we take the free albelian group generated by taking
a collection of simplicies as basis.
This allows us to express formal $ZZ$-linear combinations of simplicies,
such as $[0,1] - [1,2] + [2,3]$



A name triangulation stems from the 2D case, where a surface (2D topology)
is approximated by a collection of triangles.

When we collect all top-level simplicies together with their subsimplicies,
we obtain what is called a simplicial complex.
It's topology is not necessarily manifold and can represent the discrete variant of any
triangulatable topology.

But since we start with a manifold, the topology of our simplicial complex
will still be manifold. This means:

For a simplicial complex to manifold, the neighborhood of each vertex (i.e. the
set of simplices that contain that point as a vertex) needs to be homeomorphic
to a $n$-ball.

- The $n$-simplicies of the manifold are called cells.
- The $n-1$-simplicies of the manifold are called facets.

Therefore we are working with a simplicial manifold.

Topology is all about the connectivity of a space.
For the simplicial compelx is therefore just about how simplicies
are connected to other simplicies. This is a combinatorial theory.

Since we are doing a coordinate-free treatment, we consider abstract simplicies,
which are not really geometric in their nature, but only combinatorial.

They can be seen as lists of vertices, where vertices. Where vertices really
are just numbers. They will later on be equipped with geometry by
their edge lengths or equivalently a riemannian metric.

Given a $n$ dimensional manifold it's triangulation is a collection of $n$ simplicies.

We can extend this collection to a simplicial complex, which not only contains the
top level $n$-simplicies, but also all $k$-subsimplicies, $k <= n$.

Simplicial complexes are objects that are studied in algebraic topology.
They are very general and also allow for topological spaces that are not manifolds.
But for the application of them for PDE domains we restrict ourself to only
simplicial complexes with manifold topology, meaning every $k$-subsimplex is contained
in some $n$-simplex. And there are at most 2 facets ($(n-1)$-simplices) for each cell ($n$-simplex).

So to summarize the topology of our PDE domain is represented as a simplicial complex
that is manifold.

== Simplicial Homology


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

= Riemannian Geometry

Okay now that we've established the topology of the mesh, we need to equipe it
with some geometry as well.

The standard way of doing so is using coordinates. Usually PDE solvers always rely
on coordinates. This would just require that all vertices of the mesh have a known
coordinates.


The euclidean geometry
of a simplex would be fully specified by giving the coordinates of the $n+1$ vertices.

```rust
pub struct VertexCoords(na::DMatrix<f64>);

```

This is something formoniq supports, but it's not what is used by
the actual finite element code.

Instead we want to do a coordinate-free treatment and rely souly on intrinsic
geometry as known from differential geometry.

A manifold with intrinsic geometry is called a Riemannian manifold and it relies
on what is called a Riemannian metric $g$. It is a continuous function on the manifold
that gives at each point an inner product $g_p$ on the tangent space at this point.


If we have an immersion (maybe also embedding) $f: M -> RR^n$,
then it's differential $dif f: T_p M -> T_p RR^n$, is called the push-forward
and tells us how our intrinsic tangential vectors are being stretched when viewed geometrically.

Computationally this differential $dif f$ can be represented, since it is a linear map, by
a Jacobi Matrix $J$.

The differential tells us also how to take an inner product of our tangent
vectors, by inducing a metric
$
  g(u, v) = dif f(u) dot dif f(v)
$

Computationally this metric is represented, since it's a bilinear form, by a
metric tensor $G$.\
The above relationship then becomes
$
  G = J^transp J
$

Given a basis $diff/(diff x^1),dots,diff/(diff x^n)$ of the tangent space $T_p M$ (induced by a chart)
the metric at a point p can be fully represented by a 2-tensor called the metric tensor $amat(G)$, since it's just a bilinear form.
$
  amat(G) = [g(diff/(diff x^i),diff/(diff x^j))]_(i,j=1)^(n times n)
$



An inner product on a vector space can be represented as a Gramian matrix given a basis.

The Riemannian metric (tensor) is an inner product on the tangent vectors with
basis $diff/(diff x^i)$.\
The inverse metric is an inner product on covectors / 1-forms with basis $dif x^i$.\

Our mesh should be a piecewise flat approximation of the manifold.
Meaning that each simplex is considered to be flat and all the curvature is concentrated
in the faces $Delta_(n-1)$.
This means that the metric tensor only changes from cell to cell and therefore
is constant on each $n$-simplex.

The most natural simplex to consider is the orthogonal simplex, basically a corner of a n-cube.
This simplex can be defined as it's own coordindate realisation as an actual convex hull of
some points.
$
  Delta_perp^n = {(t_1,dots,t_n) in RR^n mid(|) sum_i t_i <= 1 "and" t_i >= 0}
$

This is the reference simplex.
It has vertices $v_0 = avec(0)$ and $v_i = avec(e)_(i-1)$.
Vertex 0 is special because it's the origin. The edges that include the origin
are the spanning edges. They are the standard basis vectors.
They give rise to an euclidean orthonormal tangent space basis. Which manifests
as a metric tensor that is equal to the identity matrix $amat(G) = amat(I)_n$.

By applying an affine linear transformation to the reference simplex, we can obtain
any other coordinate realization simplex.

If we forget about coordinates, we can obtain any metric simplex by applying
a linear transformation to the standard simplex.

Just as for a coordinate-based geometry the coordinates of the vertices are sufficent
information for the full geometry of the simplex,
the edge lengths of a simplex are sufficent information for the full information of
a coordinate-free simplex.

The geometry of the manifold will also get discretized by means of the Regge metric,
which is a piecewise-flat (over the cells) metric on the simplicial complex.

The piecewise-flat metric for a simplicial manifold, is called the Regge metric.
It only depends on the edge lengths of the simplicies. It's a concept that comes
from Regge calculus, which is a theory developed for numerical algorithms
for general relativity.
Storing only the edge lengths of the whole mesh is a more memory efficent
representation of the geometry than storing all the metric tensors.
These can be derived from the edge lengths via the law of cosines:
$
  amat(G)_(i j) = 1/2 (e_(0 i)^2 + e_(0 j)^2 - e_(i j)^2)
$

So one way of equipping our simplicial complex with geometric
information, is by specifying a function
$
  f: Delta_1 (mesh) -> RR^+
$
that gives each edge $e in Delta_1 (mesh)$ a positive length $f(e) in RR^+$.

From this one can derive the cell-piecewise constant Regge metric.

This gives us a simplicial Riemannian manifold.

= Exterior Algebra

FEEC makes use of Exterior Calculus and Differential Forms. To develop
these notions a good starting point is exterior algebra.
This is just like how one first needs to learn about vector algebra, before
one can do vector calculus.

We have an vector space $V$ or a field $KK$.\
We first define the tensor algebra
$
  T(V) = plus.circle.big_(k=0)^oo V^(times.circle k)
  = K plus.circle V plus.circle (V times.circle V) plus.circle dots.c
$

Now we define the two-sides ideal $I = { x times.circle x : x in V }$.\
The exterior algebra is now the quotient algebra of the tensor algebra by the ideal
$
  wedgespace(V) = T(V)\/I
  = wedgespace^0(V) plus.circle wedgespace^1(V) plus.circle dots.c plus.circle wedgespace^n (V)
$
The exterior product $wedge$ of two element in $wedgespace(V)$ is then
$
  alpha wedge beta = alpha times.circle beta quad (mod I)
$

We have dimensionality given by the binomial coefficent.
$
  dim wedgespace^k (V) = binom(n,k)
$

The $k$-th exterior algebra $wedgespace^k V$ over the vector space $V$ is
called the space of $k$-vectors.\
The $k$-th exterior algebra $wedgespace^k (V^*)$ over the dual space $V^*$ of $V$ is
called the space of $k$-forms.\


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

== Hodge Star operator

Computationally we are working in some basis. The following
formulas are relevant for the implementation.
They are written in tensor index notation and make
use of the einstein sum convention.

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

= Exterior Calculus

You can think of $k$-vector field as a *density* of infinitesimal oriented $k$-dimensional.

The differential $k$-form is just a $k$-form field, which is the dual measuring object.


== Differential Forms

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

An arbitrary differential form can be written as (with Einstein sum convention)
$
  alpha = 1/k!
  alpha_(i_1 dots i_k) dif x^(i_1) wedge dots.c wedge dif x^(i_k)
  = sum_(i_1 < dots < i_k) 
  alpha_(i_1 dots i_k) dif x^(i_1) wedge dots.c wedge dif x^(i_k)
$


$
  Lambda^k (Omega) = Gamma (wedge.big^k T^* (Omega))
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

This can be extended to an $L^2$-inner product on $Lambda^k (Omega)$
by integrating the pointwise inner product with respect to the volume
from $vol$ associated to $g$.

$
  (omega, tau) |-> inner(omega, tau)_(L^2 Lambda^k) :=
  integral_M inner(omega(p), tau(p))_p vol
  = integral_M omega wedge hodge tau
$

== Hodge Star on Differential Forms


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

In discrete settings defined as coboundary operator, through Stokes' theorem.\
So the discrete exterior derivative is just the transpose of the boundary operator / incidence matrix.

There is a discrete theory on exterior calculus
that is related to FEEC, called DEC.

We make use of some of the ideas from DEC.

The discrete exterior derivative is defined using the boundary
operator. Stokes' Theorem is fullfilled by definition.
$
  dif^k = diff_(k+1)^transp

$

The exterior derivative is closed in the space of Whitney forms, because of the de Rham complex.

The local (on a single cell) exterior derivative is always the same for any cell.
Therefore we can compute it on the reference cell.


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


== Codifferential


Coderivative operator $delta: Lambda^k (Omega) -> Lambda^(k-1) (Omega)$
defined such that
$
  hodge delta omega = (-1)^k dif hodge omega
  \
  delta_k := (dif_(k-1))^* = (-1)^k space (hodge_(k-1))^(-1) compose dif_(n-k) compose hodge_k
$

For vanishing boundary it's the formal $L^2$-adjoint of the exterior derivative.


= Homology

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

= Finite Element Exterior Calculus

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



== Mass bilinear form

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


== Hodge-Laplacian

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

Form the $L^2$-inner product with a test "function" $v in Lambda^k (Omega)$.
$
  Delta u = f
$

We obtain the variational equation
$
  u in H Lambda^k (Omega): quad quad
  inner(Delta u, v) = inner(f, v)
  quad quad forall v in H Lambda^k (Omega)
$

Or in integral form
$
  integral_Omega ((dif delta + delta dif) u) wedge hodge v = integral_Omega f wedge hodge v
$


If $omega$ or $eta$ vanishes on the boundary, then
$delta$ is the formal adjoint of $dif$ w.r.t. the $L^2$-inner product.
$
  inner(dif omega, eta) = inner(omega, delta eta)
$

$
  inner(Delta u, v) = inner(f, v)
  \
  inner((dif delta + delta dif) u, v) = inner(f, v)
  \
  inner((dif delta + delta dif) u, v) = inner(f, v)
  \
  inner(dif delta u, v) + inner(delta dif u, v) = inner(f, v)
  \
  inner(delta u, delta v) + inner(dif u, dif v) = inner(f, v)
$

#v(1cm)

$
  u in H Lambda^k (Omega): quad quad
  inner(delta u, delta v) + inner(dif u, dif v) = inner(f, v)
  quad
  forall v in H Lambda^k (Omega)
$

$
  u in H Lambda^k (Omega): quad
  integral_Omega (delta u) wedge hodge (delta v) + integral_Omega (dif u) wedge hodge (dif v) = integral_Omega f wedge hodge v
  quad
  forall v in H Lambda^k (Omega)
$

=== Galerkin Primal Weak Form

$
  u_h = sum_(i=1)^N mu_i phi_i
  quad quad
  v_h = phi_j
  \
  u in H Lambda^k (Omega): quad quad
  inner(delta u, delta v) + inner(dif u, dif v) = inner(f, v)
  quad quad forall v in H Lambda^k (Omega)
  \
  vvec(mu) in RR^N: quad
  sum_(i=1)^N mu_i (integral_Omega (delta phi_i) wedge hodge (delta phi_j) + integral_Omega (dif phi_i) wedge hodge (dif phi_j))
  =
  sum_(i=1)^N mu_i integral_Omega f wedge hodge phi_j
  quad forall j in {1,dots,N}
$

$
  amat(A) vvec(mu) = 0
  \
  A =
  [integral_Omega (delta phi_i) wedge hodge (delta phi_j)]_(i,j=1)^N
  +
  [integral_Omega (dif phi_i) wedge hodge (dif phi_j)]_(i,j=1)^N
  \
  vvec(phi) = [integral_Omega f wedge hodge phi_j]_(j=1)^N
$


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



$
  hodge sigma - dif^transp hodge u &= 0
  \
  hodge dif sigma + dif^transp hodge dif u &= lambda hodge u
$

$
  mat(
    hodge, -dif^transp hodge;
    hodge dif, dif^transp hodge dif;
  )
  vec(sigma, u)
  =
  lambda
  mat(
    0,0;
    0,hodge
  )
  vec(sigma, u)
$

This is a symmetric indefinite sparse generalized matrix eigenvalue problem,
that can be solved by an iterative eigensolver such as Krylov-Schur.
This is also called a GHIEP problem.


== Barycentric Coordinates

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

== Lagrange Basis
#v(1cm)

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

== Mass Bilinear Form

All bilinear forms that occur in the mixed weak formulation of Hodge-Laplacian
are just a variant of the inner product on Whitney forms.
For this reason we really just need a formula for the element matrix
of this inner product. This galerkin matrix is called the mass matrix
and the inner product could also be called the mass bilinear form.

$
  M = [inner(lambda_tau, lambda_sigma)_(L^2 Lambda^k (K))]_(sigma,tau in Delta_k (K))
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
