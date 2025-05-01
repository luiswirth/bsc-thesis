#import "../setup.typ": *
#import "../setup-math.typ": *
#import "../layout.typ": *

= Finite Element Methods for Differential Forms

We have now arrived at the chapter discussing the actual `formoniq` crate. \
Here we will derive and implement the formulas for computing the element
matrices of the various weak differential operators in Finite Element Exterior
Calculus (FEEC) @douglas:feec-article, @douglas:feec-book. Furthermore we
implement the assembly algorithm that will give us the final Galerkin matrices
@hiptmair:numpde.

== Variational Formulation & Element Matrix Computation

There are only 4 types of variational operators that
are relevant to the mixed weak formulation of the Hodge-Laplace operator @douglas:feec-book.
All of them are based on the $L^2$ inner product on Whitney forms @whitney:geointegration, @douglas:feec-article.

Above all is the mass bilinear form, which is
the $L^2$ inner product @douglas:feec-book.
$
  m^k (u, v) &= inner(u, v)_(L^2 Lambda^k (Omega))
  quad
  u in L^2 Lambda^k, v in L^2 Lambda^k
$

The next bilinear form involves the exterior derivative.
$
  d^k (u, v) &= inner(dif u, v)_(L^2 Lambda^k (Omega))
  quad
  u in H Lambda^(k-1), v in L^2 Lambda^k
$

The bilinear form involving the codifferential is also relevant @douglas:feec-article.
$
  c(u, v) &= inner(delta u, v)_(L^2 Lambda^k (Omega))
$
Using the adjoint property of the codifferential relative to the exterior
derivative under the $L^2$ inner product @douglas:feec-book, it can be rewritten
using the exterior derivative applied to the test function.
$
  c^k (u, v) &= inner(u, dif v)_(L^2 Lambda^k (Omega))
  quad
  u in L^2 Lambda^k, v in H Lambda^(k-1)
$

Lastly there is the bilinear form analogous to the scalar Laplacian, involving
exterior derivatives on both arguments. It corresponds to the $delta dif$ part
of the Hodge-Laplacian @douglas:feec-book.
$
  l^k (u, v) &= inner(dif u, dif v)_(L^2 Lambda^(k+1) (Omega))
  quad
  u in H Lambda^k, v in H Lambda^k
$

After Galerkin discretization @hiptmair:numpde, by means of the Whitney finite
element space @whitney:geointegration with the Whitney basis ${phi_i^k}$, we
arrive at the following Galerkin matrices for our four weak operators.
$
  amat(M)^k &= [inner(phi^k_i, phi^k_j)]_(i j) \
  amat(D)^k &= [inner(phi^k_i, dif phi^(k-1)_j)]_(i j) \
  amat(C)^k &= [inner(dif phi^(k-1)_i, phi^k_j)]_(i j) \
  amat(L)^k &= [inner(dif phi^k_i, dif phi^k_j)]_(i j) \
$

We can rewrite the 3 operators that involve the exterior derivative using the
mass matrix and the discrete exterior derivative (coboundary/incidence matrix)
@douglas:feec-article, @crane:ddg.
$
  amat(D)^k &= amat(M)^k amat(dif)^(k-1) \
  amat(C)^k &= (amat(dif)^(k-1))^transp amat(M)^k \
  amat(L)^k &= (amat(dif)^k)^transp amat(M)^(k+1) amat(dif)^k \
$

As usual in a FEM library @hiptmair:numpde, we define element matrix providers,
that compute the element matrices on each cell of mesh and later on assemble the
full Galerkin matrices.

We first define a element matrix provider trait
```rust
pub type ElMat = Matrix;
pub trait ElMatProvider: Sync {
  fn row_grade(&self) -> ExteriorGrade;
  fn col_grade(&self) -> ExteriorGrade;
  fn eval(&self, geometry: &SimplexLengths) -> ElMat;
}
```
The `eval` method provides us with the element matrix on a
cell, given it's geometry. But we also need to know the exterior grade
of the Whitney forms that correspond to the rows and columns.
This information will be used by the assembly routine.

We will now implement the 3 operators involving exterior derivatives
based on the element matrix provider of the mass bilinear form.

The local exterior derivative only depends on the local topology, which is
the same for any simplex of the same dimension. So we use a global variable
that stores the transposed incidence matrix (coboundary operator) for any
$k$-skeleton of a $n$-complex @crane:ddg.

```rust
pub struct DifElmat {
  mass: HodgeMassElmat,
  dif: Matrix,
}
impl DifElmat {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    let mass = HodgeMassElmat::new(dim, grade);
    let dif = Complex::standard(dim).exterior_derivative_operator(grade - 1);
    let dif = Matrix::from(&dif);
    Self { mass, dif }
  }
}
impl ElMatProvider for DifElmat {
  fn row_grade(&self) -> ExteriorGrade { self.mass.grade }
  fn col_grade(&self) -> ExteriorGrade { self.mass.grade - 1 }
  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    let mass = self.mass.eval(geometry);
    mass * &self.dif
  }
}

pub struct CodifElmat {
  mass: HodgeMassElmat,
  codif: Matrix,
}
impl CodifElmat {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    let mass = HodgeMassElmat::new(dim, grade);
    let dif = Complex::standard(dim).exterior_derivative_operator(grade - 1);
    let dif = Matrix::from(&dif);
    let codif = dif.transpose();
    Self { mass, codif }
  }
}
impl ElMatProvider for CodifElmat {
  fn row_grade(&self) -> ExteriorGrade { self.mass.grade - 1 }
  fn col_grade(&self) -> ExteriorGrade { self.mass.grade }
  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    let mass = self.mass.eval(geometry);
    &self.codif * mass
  }
}

pub struct CodifDifElmat {
  mass: HodgeMassElmat,
  dif: Matrix,
  codif: Matrix,
}
impl CodifDifElmat {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    let mass = HodgeMassElmat::new(dim, grade + 1);
    let dif = Complex::standard(dim).exterior_derivative_operator(grade);
    let dif = Matrix::from(&dif);
    let codif = dif.transpose();
    Self { mass, dif, codif }
  }
}
impl ElMatProvider for CodifDifElmat {
  fn row_grade(&self) -> ExteriorGrade { self.mass.grade - 1 }
  fn col_grade(&self) -> ExteriorGrade { self.mass.grade - 1 }
  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    let mass = self.mass.eval(geometry);
    &self.codif * mass * &self.dif
  }
}
```


=== Mass bilinear form

Now we need to implement the element matrix provider to the mass bilinear form.
Here is where the geometry of the domain comes in, through the inner product, which
depends on the Riemannian metric @frankel:diffgeo.

One could also understand the mass bilinear form as a weak Hodge star operator
@douglas:feec-book.
$
  amat(M)_(i j) = integral_Omega phi_j wedge hodge phi_i
  = inner(phi_j, phi_i)_(L^2 Lambda^k (Omega))
$

We will not compute this using the Hodge star operator, but instead directly
using the inner product.

We already have an inner product on constant multiforms. We now need to
extend it to an $L^2$ inner product on Whitney forms @douglas:feec-article.
This can be done by inserting the definition of a Whitney form (in terms of barycentric
coordinate functions) into the inner product.

// TODO: FIND REFERENCE
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
  integral_K lambda_i_l lambda_j_m vol_g \
$


We can now make use of the fact that the exterior derivative of the barycentric
coordinate functions are constant @hiptmair:numpde. This makes the big wedge
terms also constant. We can therefore pull them out of the $L^2$ integral and
now it's just an inner product on constant multiforms. What remains in the in
the integral is the product of two barycentric coordinate functions.

Using this we can now implement the element matrix provider
to the mass bilinear form in Rust.
```rust
pub struct HodgeMassElmat {
  dim: Dim,
  grade: ExteriorGrade,
  simplices: Vec<Simplex>,
  wedge_terms: Vec<ExteriorElementList>,
}
impl HodgeMassElmat {
  pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
    let simplices: Vec<_> = standard_subsimps(dim, grade).collect();
    let wedge_terms: Vec<ExteriorElementList> = simplices
      .iter()
      .cloned()
      .map(|simp| WhitneyLsf::standard(dim, simp).wedge_terms().collect())
      .collect();

    Self { dim, grade, simplices, wedge_terms }
  }
}
impl ElMatProvider for HodgeMassElmat {
  fn row_grade(&self) -> ExteriorGrade { self.grade }
  fn col_grade(&self) -> ExteriorGrade { self.grade }

  fn eval(&self, geometry: &SimplexLengths) -> Matrix {
    assert_eq!(self.dim, geometry.dim());

    let scalar_mass = ScalarMassElmat.eval(geometry);

    let mut elmat = Matrix::zeros(self.simplices.len(), self.simplices.len());
    for (i, asimp) in self.simplices.iter().enumerate() {
      for (j, bsimp) in self.simplices.iter().enumerate() {
        let wedge_terms_a = &self.wedge_terms[i];
        let wedge_terms_b = &self.wedge_terms[j];
        let wedge_inners = multi_gramian(&geometry.to_metric_tensor().inverse(), self.grade)
          .inner_mat(wedge_terms_a.coeffs(), wedge_terms_b.coeffs());

        let nvertices = self.grade + 1;
        let mut sum = 0.0;
        for avertex in 0..nvertices {
          for bvertex in 0..nvertices {
            let sign = Sign::from_parity(avertex + bvertex);
            let inner = wedge_inners[(avertex, bvertex)];
            sum += sign.as_f64() * inner * scalar_mass[(asimp[avertex], bsimp[bvertex])];
          }
        }
        elmat[(i, j)] = sum;
      }
    }
    factorial(self.grade).pow(2) as f64 * elmat
  }
}
```

Now we are just missing an element matrix provider for the scalar mass
bilinear form.
Luckily there exists a closed form solution, for
this integral, which only depends on the volume of the cell @hiptmair:numpde.
$
  integral_K lambda_i lambda_j vol
  = abs(K)/((n+2)(n+1)) (1 + delta_(i j))
$
derived from this more general integral formula for powers of barycentric coordinate functions @hiptmair:numpde
$
  integral_K lambda_0^(alpha_0) dots.c lambda_n^(alpha_n) vol
  =
  n! abs(K) (alpha_0 ! space dots.c space alpha_n !)/(alpha_0 + dots.c + alpha_n + n)!
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

== Source Terms & Quadrature

Next to bilinear forms and element matrix providers, there are also linear forms
and corresponding element vector providers.

We define a trait for element vector providers, just like we did for the element
matrices.
```rust
pub type ElVec = Vector;
pub trait ElVecProvider: Sync {
  fn grade(&self) -> ExteriorGrade;
  fn eval(&self, geometry: &SimplexLengths, topology: &Simplex) -> ElVec;
}
```
Here in contrast to the element matrix provider, we also pass the topology as an argument,
to the evaluation method. This might come as a surprise, as this element vector should
be purely local and therefore not work with any global topological information.
However the element matrix provider for the source term will need global coordinates, but since
we don't want to include coordinates in our `ElVecProvider` trait, we need to supply
information on which topological simplex we are on, in order to reconstruct the corresponding coordinates.

We only have a single implementor of the `ElVecProvider` trait, which is the
element vector that corresponds to the right-hand-side source term, as it appears for instance
in the Hodge-Laplace source problem.
$
  avec(b)_K = [inner(f, phi^k_j)_(L^2 Lambda^k (K))]_(i=1)^(N_k)
$

It takes as input an `ExteriorField` that corresponds to $f$, represented as
a functor that takes a global coordinate $avec(x) in RR^N$ (based on an embedding) and produces the
multiform $f_avec(x)$ at that position. It constitutes the coordinate-based
representation of the arbitrary continuous differential form $f$.

Since only point-evaluation is available to us, we need to rely on quadrature @hiptmair:numpde
to compute the element vector corresponding to the source term. For this we
pullback the `ExteriorField` representing `f` to the reference cell using
`precompose_form` and compute the pointwise inner product.
$
  inner(f, phi^k_j)_(L^2 Lambda^k)
  = integral_K inner(f (avec(x)), phi^k_j (avec(x)))_(Lambda^k) vol_g
  approx abs(K) sum_l w_l inner(phi_K^* f (avec(q)_l), phi^k_j (avec(q)_l))_(Lambda^k)
$

The choice of quadrature is free, but simple barycentric quadrature suffices, to get
an admissible variational crime for 1st order FE. @hiptmair:numpde

This is what the full implementation of this element vector provider then looks like.
```rust
pub struct SourceElVec<'a, F>
where
  F: ExteriorField,
{
  source: &'a F,
  mesh_coords: &'a MeshCoords,
  qr: SimplexQuadRule,
}
impl<'a, F> SourceElVec<'a, F>
where
  F: ExteriorField,
{
  pub fn new(source: &'a F, mesh_coords: &'a MeshCoords, qr: Option<SimplexQuadRule>) -> Self {
    let qr = qr.unwrap_or(SimplexQuadRule::barycentric(source.dim_intrinsic()));
    Self { source, mesh_coords, qr }
  }
}
impl<F> ElVecProvider for SourceElVec<'_, F>
where
  F: Sync + ExteriorField,
{
  fn grade(&self) -> ExteriorGrade {
    self.source.grade()
  }
  fn eval(&self, geometry: &SimplexLengths, topology: &Simplex) -> ElVec {
    let cell_coords = SimplexCoords::from_simplex_and_coords(topology, self.mesh_coords);

    let dim = self.source.dim_intrinsic();
    let grade = self.grade();
    let dof_simps: Vec<_> = standard_subsimps(dim, grade).collect();
    let whitneys: Vec<_> = dof_simps
      .iter()
      .cloned()
      .map(|dof_simp| WhitneyLsf::standard(dim, dof_simp))
      .collect();

    let inner = multi_gramian(&geometry.to_metric_tensor().inverse(), grade);

    let mut elvec = ElVec::zeros(whitneys.len());
    for (iwhitney, whitney) in whitneys.iter().enumerate() {
      let inner_pointwise = |local: CoordRef| {
        let global = cell_coords.local2global(local);
        inner.inner(
          self
            .source
            .at_point(&global)
            .precompose_form(&cell_coords.linear_transform())
            .coeffs(),
          whitney.at_point(local).coeffs(),
        )
      };
      let value = self.qr.integrate_local(&inner_pointwise, geometry.vol());
      elvec[iwhitney] = value;
    }
    elvec
  }
}
```


== Assembly

The element matrix provider provides the exterior grade of the bilinear
form arguments. Based on this the corresponding dimension of simplices are
used to assemble. The assembly process itself is a standard FEM technique
@hiptmair:numpde.

We use rayon @crate:rayon to parallelize the assembly process over all the different cells,
since these computations are always independent.

```rust
pub type GalMat = CooMatrix;
pub fn assemble_galmat(
  topology: &Complex,
  geometry: &MeshLengths,
  elmat: impl ElMatProvider,
) -> GalMat {
  let row_grade = elmat.row_grade();
  let col_grade = elmat.col_grade();
  let nsimps_row = topology.skeleton(row_grade).len();
  let nsimps_col = topology.skeleton(col_grade).len();

  let triplets: Vec<(usize, usize, f64)> = topology
    .cells()
    .handle_iter()
    // parallelization via rayon
    .par_bridge()
    .flat_map(|cell| {
      let geo = geometry.simplex_lengths(cell);
      let elmat = elmat.eval(&geo);

      let row_subs: Vec<_> = cell.mesh_subsimps(row_grade).collect();
      let col_subs: Vec<_> = cell.mesh_subsimps(col_grade).collect();

      let mut local_triplets = Vec::new();
      for (ilocal, &iglobal) in row_subs.iter().enumerate() {
        for (jlocal, &jglobal) in col_subs.iter().enumerate() {
          let val = elmat[(ilocal, jlocal)];
          if val != 0.0 {
            local_triplets.push((iglobal.kidx(), jglobal.kidx(), val));
          }
        }
      }

      local_triplets
    })
    .collect();
  let (rows, cols, values) = triplets.into_iter().multiunzip();
  GalMat::try_from_triplets(nsimps_row, nsimps_col, rows, cols, values).unwrap()
}
```

We also have an assembly algorithm for Galerkin vectors.
```rust
pub type GalVec = Vector;
pub fn assemble_galvec(
  topology: &Complex,
  geometry: &MeshLengths,
  elvec: impl ElVecProvider,
) -> GalVec {
  let grade = elvec.grade();
  let nsimps = topology.skeleton(grade).len();

  let entries: Vec<(usize, f64)> = topology
    .cells()
    .handle_iter()
    // parallelization via rayon
    .par_bridge()
    .flat_map(|cell| {
      let geo = geometry.simplex_lengths(cell);
      let elvec = elvec.eval(&geo, &cell);
      let subs: Vec<_> = cell.mesh_subsimps(grade).collect();

      let mut local_entires = Vec::new();
      for (ilocal, &iglobal) in subs.iter().enumerate() {
        if elvec[ilocal] != 0.0 {
          local_entires.push((iglobal.kidx(), elvec[ilocal]));
        }
      }
      local_entires
    })
    .collect();

  let mut galvec = Vector::zeros(nsimps);
  for (irow, val) in entries {
    galvec[irow] += val;
  }
  galvec
}
```
