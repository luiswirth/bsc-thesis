#import "../setup.typ": *
#import "../setup-math.typ": *
#import "../layout.typ": *

= Discrete Differential Forms: Simplicial Cochains and Whitney Forms

Having discretized the smooth domain $Omega$ into a simplicial complex $mesh$,
we now require a corresponding discretization for the differential forms
$Lambda(Omega)$ defined upon it @whitney:geointegration.
FEEC relies on discrete counterparts, called discrete differential forms (DDF)
$Lambda_h (mesh)$ defined on the simplicial complex $mesh$, that faithfully
represent the structure of the continuous forms @douglas:feec-book, @douglas:feec-article.

Central to FEEC are two fundamental, closely related concepts: @whitney:geointegration
- *Simplicial Cochains*: These are the primary discrete objects. A $k$-cochain
  assigns a real value (a degree of freedom) to each $k$-simplex of the mesh
  $mesh$. They form the algebraic backbone, capturing the combinatorial topology
  and enabling discrete versions of operators like the exterior derivative @crane:ddg.
- *Whitney Forms*: These constitute the lowest-order finite element space
  for differential forms @whitney:geointegration. Each Whitney $k$-form is
  a piecewise polynomial basis function associated with a $k$-simplex. They
  provide a means to reconstruct a field across the mesh by interpolating the
  cochain values and are essential for defining the integrals required in weak
  formulations @douglas:feec-book.

We will demonstrate that the space of $k$-cochains and the space spanned by
Whitney $k$-forms are isomorphic. The *projection* from
continuous forms to cochains is realized by the *integration map*, while the
*interpolation* from cochains to the Whitney space is achieved via the *Whitney
map* @whitney:geointegration.

This chapter explores the representation and manipulation of these discrete
differential forms. We will focus on two key processes:
- *Discretization*: Projecting continuous differential forms onto the discrete
  setting, yielding degrees of freedom associated with the mesh simplices.
- *Reconstruction*: Interpolating these discrete degrees of freedom using basis
  functions to obtain a continuous (albeit piecewise polynomial) representation
  over the mesh, suitable for variational methods.


Understanding these discrete structures and the maps connecting them is crucial,
as they provide the foundation for constructing the finite element spaces and
structure-preserving discrete operators used throughout FEEC @douglas:feec-book.

This chapter lays the foundation for the discrete variational formulations used
in FEEC.

== Cochains

A $k$-cochain $omega$ is a real-valued function $omega: Delta_k (mesh) -> RR$
on the $k$-skeleton $Delta_k (mesh)$ of the mesh $mesh$ @whitney:geointegration.

A rank $k$ differential $k$-form, becomes a $k$-cochain, defined on the simplices
of dimension $k$ of the mesh.

Simplicial cochains arise naturally from the combinatorial structure of a
simplical complex. A simplicial cochain is dual to a simplicial chain.

Simplicial cochains are also the fundamental combinatorial object
in *discrete exterior calculus* (DEC) @crane:ddg.

One can represent this function on the simplices, using a list of real values
that are ordered according to the global numbering of the simplices.
```rust
pub struct Cochain {
  pub coeffs: na::DVector<f64>,
  pub dim: Dim,
}
```

Simplicial cochains preserve the structure of the de Rham complex at a discrete
level and therefore retain the key topological and geometrical properties from
differential topology/geometry @douglas:feec-article.

Cochains can be seen as the coefficents or DOFs of our FE spaces.

=== Discretization: Cochain-Projection via Integration

What is the interpretation of these cochain values?
This question can be answered by looking at the discretization procedure
of a continuous differential form to this discrete form.

The discretization of differential forms happens by projection
onto the cochain space. This cochain-projection is the simple operation
of integration the given continuous differential $k$-form over every $k$-simplex
of the mesh. This gives a real number for each $k$-simplex, which is exactly
the cochain values.
This projection is called the *integration map* @whitney:geointegration
$
  I: Lambda^k (Omega) -> C^k (mesh; RR)
$

$
  I(omega) = (sigma |-> c_sigma) quad "where" quad c_sigma = integral_sigma omega quad forall sigma in Delta_k (mesh)
$

The integral of a differential $k$-form over a $k$-simplex is defined using the pullback
to the reference $k$-simplex.
$
  integral_sigma omega
  &= integral_hat(sigma) phi^* omega \
  &= integral_hat(sigma) omega_(phi(lambda^1,dots,lambda^k)) (phi_* nvec(e)_1, dots, phi_* nvec(e)_k) dif lambda^1 dots dif lambda^k \
  &= integral_hat(sigma) omega_(phi(lambda^1,dots,lambda^k)) ((diff avec(x))/(diff lambda_1), dots, (diff avec(x))/(diff lambda_k)) dif lambda^1 dots dif lambda^k
$

The last expression is a traditional pre-differential geometry integral over a
subset $hat(sigma) in RR^k$. No exterior calculus required.

We approximate this integral using barycentric quadrature.
$
  integral_sigma omega approx |hat(sigma)| omega_(phi(avec(m)_hat(sigma))) (avec(e)_1,dots,avec(e)_n)
$

And the implementation just looks like this:
```rust
pub fn integrate_form_simplex(form: &impl DifferentialMultiForm, simplex: &SimplexCoords) -> f64 {
  let multivector = simplex.spanning_multivector();
  let f = |coord: CoordRef| {
    form
      .at_point(simplex.local2global(coord).as_view())
      .apply_form_on_multivector(&multivector)
  };
  let std_simp = SimplexCoords::standard(simplex.dim_intrinsic());
  barycentric_quadrature(&f, &std_simp)
}
```

And for the full cochain-projection, we just repeat this integration
for each simplex in the $k$-skeleton. The implementation the looks like this.
```rust
pub fn cochain_projection(
  form: &impl DifferentialMultiForm,
  topology: &Complex,
  coords: &MeshCoords,
) -> Cochain {
  let cochain = topology
    .skeleton(form.grade())
    .handle_iter()
    .map(|simp| SimplexCoords::from_simplex_and_coords(&simp, coords))
    .map(|simp| integrate_form_simplex(form, &simp))
    .collect::<Vec<_>>()
    .into();
  Cochain::new(form.grade(), cochain)
}
```


=== Discrete Exterior Derivative via Stokes' Theorem

In exterior calculus, the exterior derivative is a fundamental operator
that generalizes the standard derivatives (gradient, curl, divergence) to
differential forms @frankel:diffgeo.
$
  dif: Lambda^k (Omega) -> Lambda^(k+1) (Omega)
$

For our discrete setting, we require a
discrete counterpart that acts on cochains. This can be derived through the lens
of cochain calculus.
$
  dif_h: C^k (mesh) -> C^(k+1) (mesh)
$

A crucial property of the continuous exterior derivative is captured by *Stokes'
Theorem*. For a differential form $omega$ and a chain $c$, this theorem relates
the integral of the exterior derivative over the chain to the integral of the
form over the chain's boundary @frankel:diffgeo, @hatcher:algtop:
$
  integral_c dif omega = integral_(diff c) omega
$

This relationship is particularly insightful when viewed through the framework
of dual pairings. If we define a natural pairing $inner(dot, dot)$ between
differential forms and chains as integration over the chain:
$
  inner(omega, c) := integral_c omega
$

Now Stokes' Theorem can be expressed in a more abstract form:
$
  inner(dif omega, c) = inner(omega, diff c)
$
This equation reveals that, with respect to this dual pairing, the exterior
derivative operator $dif$ is the adjoint of the boundary operator $diff$.
$
  dif = diff^*
$

This adjoint relationship provides the direct motivation for defining the
discrete exterior derivative.
On a simplicial complex, we use cochains as discrete differential forms
and the boundary operator
is a fundamental combinatorial operator acting on chains. The discrete
exterior derivative, often called the *coboundary operator*, is thus defined
as the adjoint of the boundary operator. By definition, this discrete
operator preserves the structure of Stokes' Theorem at the discrete level
@crane:ddg.

From a computational perspective, the boundary operator $diff^k$ mapping
$k$-chains to $(k-1)$-chains can be represented as a signed incidence matrix
between the simplices in the $k$-skeleton and $(k-1)$-skeleton @crane:ddg.
The adjoint property then translates directly to the discrete exterior
derivative $dif^k$ (mapping $k$-cochains to $(k+1)$-cochains) being the
transpose of the boundary operator $diff_(k+1)$:
// TODO: notation??
$
  amat(dif)^k = amat(D)_(k+1)^transp
$

This definition highlights a key feature of the discrete exterior derivative: it
is a purely topological operator. Its definition and computation depend only on
the combinatorial structure of the simplicial complex (captured by the incidence
relationships in the boundary operator), and not on the geometry or metric of
the underlying manifold @crane:ddg.

In our implementation, we represent the discrete exterior derivative as a sparse
matrix. Using an extension trait, we provide a method to compute this matrix for
a given grade, leveraging the already implemented boundary operator:
```rust
pub trait ManifoldComplexExt { ... }
impl ManifoldComplexExt for Complex {
  /// $dif^k: cal(W) Lambda^k -> cal(W) Lambda^(k+1)$
  fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> SparseMatrix {
    self.boundary_operator(grade + 1).transpose()
  }
}
```


== Whitney Forms

*Whitney forms* are the *finite element differential forms* that correspond to
cochains. They are the piecewise-linear (over the cells) differential
forms defined over the simplicial manifold @whitney:geointegration.

The Whitney space $cal(W) Lambda^k (mesh)$ is the space of all Whitney forms
over our mesh $mesh$

For a $mesh$ with topological dimension $n$, the Whitney $0$-form space coincides with
the Lagrangian space and the $n$-forms coincides with piecewise-constant discontinuous elements.
$
  cal(W) Lambda^0 (mesh) &=^~ cal(S)^0_1 (mesh) \
  cal(W) Lambda^n (mesh) &=^~ cal(S)^(-1)_0 (mesh) \
$

For a 3D mesh, we additionaly have a correspondance with Raviart-Thomas
$bold(cal(R T))(mesh)$ and Nédélec $bold(cal(N))(mesh)$ Finite Element spaces.
$
  cal(W) Lambda^0 (mesh) &=^~ cal(S)^0_1 (mesh) \
  cal(W) Lambda^1 (mesh) &=^~ bold(cal(N)) (mesh) \
  cal(W) Lambda^2 (mesh) &=^~ bold(cal(R T)) (mesh) \
  cal(W) Lambda^3 (mesh) &=^~ cal(S)^(-1)_0 (mesh) \
$

THis is the famous discrete subcomplex of the de Rham complex.
$
  0 -> cal(S)^0_1 (mesh) limits(->)^grad bold(cal(N)) (mesh) limits(->)^curl bold(cal(R T)) (mesh) limits(->)^div cal(S)^(-1)_0 (mesh) -> 0
$

The *Whitney subcomplex* generalize it to arbitrary dimensions.
$
  0 -> cal(W) Lambda^0 (mesh) limits(->)^dif dots.c limits(->)^dif cal(W) Lambda^n (mesh) -> 0
$


=== Whitney Basis

There is a special basis for the space of Whitney forms, called the *Whitney
basis* @whitney:geointegration, @douglas:feec-article. Just like there is a
cochain value for each $k$-simplex, there is a Whitney basis function for each
$k$-simplex. They have their DOF on this $K$-simplex.
$
  cal(W) Lambda^k (mesh) = "span" {lambda_sigma : sigma in Delta_k (mesh)}
$

Let's take a look at the local shape functions (LSF).

The local Whitney form $lambda_(i_0 dots i_k)$ associated with the DOF simplex
$sigma = [i_0 dots i_k] subset.eq tau$ on the cell $tau = [j_0 dots j_n]$ is
defined using the barycentric coordinate functions $lambda_i_s$ of the cell.
$
  lambda_(i_0 dots i_k) =
  k! sum_(l=0)^k (-1)^l lambda_i_l
  (dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k)
$ <def:whitney>

To get some intuition for the kind of fields this produces, let's look at some
visualizations. We do this specifically for 1-forms, since we can visualize
these using vector field proxies.

We have the following formula for Whitney 1-forms.
$
  lambda_(i j) = lambda_i dif lambda_j - lambda_j dif lambda_i
$

For the reference 2-simplex, we get the following Whitney basis 1-forms.
$
  lambda_01 &= (1-y) dif x + x dif y
  \
  lambda_02 &= y dif x + (1-x) dif y
  \
  lambda_12 &= -y dif x + x dif y
$

Visualized on the reference triangle they look like:
#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    rows: 1,
    gutter: 3pt,
    image("../../res/ref_lambda01.png", width: 100%),
    image("../../res/ref_lambda02.png", width: 100%),
    image("../../res/ref_lambda12.png", width: 100%),
  ),
  caption: [
    Vector proxies of Reference Local Shape Functions
    $lambda_01, lambda_02, lambda_12 in cal(W) Lambda^1 (Delta_2^"ref")$.
  ],
) <img:ref_whitneys>

The global shape functions are obtained by combing the various
local shape functions that are associated with the same DOF simplex.
In general the GSF are discontinuous over cell boundaries.

We visualize here the 3 GSF on a equilateral triangle mesh associated
with the edges of the middle triangle.

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    rows: 1,
    gutter: 3pt,
    image("../../res/eq_phi01.png", width: 100%),
    image("../../res/eq_phi02.png", width: 100%),
    image("../../res/eq_phi12.png", width: 100%),
  ),
  caption: [
    Vector proxies of Global Shape Functions
    $phi_01, phi_02, phi_12 in cal(W) Lambda^1 (mesh)$ \
    on equilateral triangle mesh $mesh$.
  ],
) <img:global_whitneys>

Via linear combination of these GSF we can obtain
any Whitney form.

We can for instance construct the following constant, purely divergent,
and purely rotational vector fields.
#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    rows: 1,
    gutter: 3pt,
    image("../../res/triforce_constant.cochain.png", width: 100%),
    image("../../res/triforce_div.cochain.png", width: 100%),
    image("../../res/triforce_rot.cochain.png", width: 100%),
  ),
  caption: [
    Vector proxies of some example Whitney forms on equilateral triangle mesh
    $mesh$.
  ],
) <img:fe_whitneys>

We now implement some functionality to work with Whitney forms.
The most important of which is the point evaluation of Whitney basis forms
on a cell associated with one of it's subsimplices. Which means
implementing @def:whitney.

We do this implementation based on a coordinate simplex. But this doesn't
mean that this relies on an embedding. It is multi-purpose!
If we just choose the reference simplex as the coordinate simplex,
then we just compute the reference Whitney basis forms.
If we choose a coordinate simplex from an embedding, we compute the
representation of a Whitney form in this embedding.

We define a struct that stores the coordinate of the cell we are working on
as well as the DOF subsimplex (in local indices) that the Whitney basis function
is associated with.

```rust
#[derive(Debug, Clone)]
pub struct WhitneyLsf {
  cell_coords: SimplexCoords,
  dof_simp: Simplex,
}
impl WhitneyLsf {
  pub fn from_coords(cell_coords: SimplexCoords, dof_simp: Simplex) -> Self {
    Self {
      cell_coords,
      dof_simp,
    }
  }
  pub fn standard(cell_dim: Dim, dof_simp: Simplex) -> Self {
    Self::from_coords(SimplexCoords::standard(cell_dim), dof_simp)
  }
  pub fn grade(&self) -> ExteriorGrade { self.dof_simp.dim() }
```

The defining formula of a Whitney basis form, relies on the barycentric
coordinate functions of the cell, but only those that are on the vertices
of the DOF simplex.
We write a function to get an iterator on exactly these.
```rust
  /// The difbarys of the vertices of the DOF simplex.
  pub fn difbarys(&self) -> impl Iterator<Item = MultiForm> + use<'_> {
    self
      .cell_coords
      .difbarys_ext()
      .into_iter()
      .enumerate()
      .filter_map(|(ibary, difbary)| self.dof_simp.contains(ibary).then_some(difbary))
  }
```

We can observe that the big wedge terms
$dif lambda_i_0 wedge dots.c wedge hat(dif lambda)_i_l wedge dots.c wedge dif lambda_i_k$
in @def:whitney are constant multiforms.
We write a function that computes these constants.
```rust
  /// d𝜆_i_0 ∧⋯∧̂ omit(d𝜆_i_iwedge) ∧⋯∧ d𝜆_i_dim
  pub fn wedge_term(&self, iterm: usize) -> MultiForm {
    let dim_cell = self.cell_coords.dim_intrinsic();
    let wedge = self
      .difbarys()
      .enumerate()
      // leave off i'th difbary
      .filter_map(|(pos, difbary)| (pos != iterm).then_some(difbary));
    MultiForm::wedge_big(wedge).unwrap_or(MultiForm::one(dim_cell))
  }
  pub fn wedge_terms(&self) -> impl ExactSizeIterator<Item = MultiForm> + use<'_> {
    (0..self.dof_simp.nvertices()).map(move |iwedge| self.wedge_term(iwedge))
  }
```

Now we can implement the `ExteriorField` trait for our `WhitneyLsf`
struct to make it point evaluable, based on a coordiante.
```rust
impl ExteriorField for WhitneyLsf {
  fn dim_ambient(&self) -> exterior::Dim { self.cell_coords.dim_ambient() }
  fn dim_intrinsic(&self) -> exterior::Dim { self.cell_coords.dim_intrinsic() }
  fn grade(&self) -> ExteriorGrade { self.grade() }
  fn at_point<'a>(&self, coord: impl Into<CoordRef<'a>>) -> MultiForm {
    let barys = self.cell_coords.global2bary(coord);
    assert!(is_bary_inside(&barys), "Point is outside cell.");

    let dim = self.dim_intrinsic();
    let grade = self.grade();
    let mut form = MultiForm::zero(dim, grade);
    for (iterm, &vertex) in self.dof_simp.vertices.iter().enumerate() {
      let sign = Sign::from_parity(iterm);
      let wedge = self.wedge_term(iterm);

      let bary = barys[vertex];
      form += sign.as_f64() * bary * wedge;
    }
    (factorial(grade) as f64) * form
  }
}
```

We can implement one more functionality.

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

This is the corresponding implementation.
```rust
  pub fn dif(&self) -> MultiForm {
    let dim = self.cell_coords.dim_intrinsic();
    let grade = self.grade();
    if grade == dim {
      return MultiForm::zero(dim, grade + 1);
    }
    factorialf(grade + 1) * MultiForm::wedge_big(self.difbarys()).unwrap()
  }
}
```

The defining property of the Whitney basis is a from pointwise to integral
generalized Lagrange basis property @whitney:geointegration:\
For any two $k$-simplices $sigma, tau in Delta_k (mesh)$, we have
$
  integral_sigma lambda_tau = cases(
    +&1 quad &"if" sigma = +tau,
    -&1 quad &"if" sigma = -tau,
     &0 quad &"if" sigma != plus.minus tau,
  )
$

We can write a test that verifies our implementation by checking this property.
```rust
#[test]
fn whitney_basis_property() {
  for dim in 0..=4 {
    let topology = Complex::standard(dim);
    let coords = MeshCoords::standard(dim);

    for grade in 0..=dim {
      for dof_simp in topology.skeleton(grade).handle_iter() {
        let whitney_form = WhitneyLsf::standard(dim, (*dof_simp).clone());

        for other_simp in topology.skeleton(grade).handle_iter() {
          let are_same_simp = dof_simp == other_simp;
          let other_simplex = other_simp.coord_simplex(&coords);
          let discret = integrate_form_simplex(&whitney_form, &other_simplex);
          let expected = are_same_simp as u8 as f64;
          let diff = (discret - expected).abs();
          const TOL: f64 = 10e-9;
          let equal = diff <= TOL;
          assert!(equal, "for: computed={discret} expected={expected}");
          if other_simplex.nvertices() >= 2 {
            let other_simplex_rev = other_simplex.clone().flipped_orientation();
            let discret_rev = integrate_form_simplex(&whitney_form, &other_simplex_rev);
            let expected_rev = Sign::Neg.as_f64() * are_same_simp as usize as f64;
            let diff_rev = (discret_rev - expected_rev).abs();
            let equal_rev = diff_rev <= TOL;
            assert!(
              equal_rev,
              "rev: computed={discret_rev} expected={expected_rev}"
            );
          }
        }
      }
    }
  }
}
```

=== Reconstruction: Whitney-Interpolation via the Whitney map

There is a one-to-one correspondance between $k$-cochain and Whitney $k$-form @whitney:geointegration

We have already seen how to obtain a cochain from a continuous differential form via
cochain-projection. This is how we can obtain the cochain corresponding to a Whitney
form.
But what about the other way? How can we reconstruct a continuous differential form
from a cochain. How can we obtain the Whitney form corresponding to this cochain?


This reconstruction is achieved by the so called *Whitney map* @whitney:geointegration.

$
  W: C^k (mesh; RR) -> cal(W)^k (mesh) subset.eq Lambda^k (Omega)
$

$
  W(c) = sum_(sigma in mesh_k) c(sigma) omega_sigma
$
It can be seen as a generalized interpolation operator.
Instead of pointwise interpolation, we have interpolation in an integral sense.
It takes cochains to differential forms that have the cochains values as integral
values @whitney:geointegration.

The isomorphism between Whitney forms and cochains can now be constructed
$
  W compose I = id_(cal(W)^k (mesh)) \
  I compose W = id_(C^k (mesh; RR))
$

On our implementation side, we introduce a routine that takes
a cochain and let's us evaluate the corresponding Whitney form
at any point on the simplicial manifold.
```rust
pub fn whitney_form_eval<'a>(
  coord: impl Into<CoordRef<'a>>,
  cochain: &Cochain,
  mesh_cell: SimplexHandle,
  mesh_coords: &MeshCoords,
) -> MultiForm {
  let coord = coord.into();

  let cell_coords = mesh_cell.coord_simplex(mesh_coords);

  let dim_intrinsic = mesh_cell.dim();
  let grade = cochain.dim;

  let mut value = MultiForm::zero(dim_intrinsic, grade);
  for dof_simp in mesh_cell.mesh_subsimps(grade) {
    let local_dof_simp = dof_simp.relative_to(&mesh_cell);

    let lsf = WhitneyLsf::from_coords(cell_coords.clone(), local_dof_simp);
    let lsf_value = lsf.at_point(coord);
    let dof_value = cochain[dof_simp];
    value += dof_value * lsf_value;
  }
  value
}
```
We shall provide the cell on which the point lives,
to avoid searching over all the cells for the one containing
the point.

== Higher-Order Discrete Differential Forms

The theoretical construction of finite element differential forms
exist for all polynomial degrees @douglas:feec-book, @hiptmair:whitneyforms.
We don't support them in this implementation, but this
a very obvious possible future extension to this implementation.
One just needs to keep in mind that then higher-order manifold
approximations are also needed to not incur any non-admissible geometric
variational crimes @holst:gvc.
