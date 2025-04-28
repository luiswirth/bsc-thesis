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
the $L^2$ inner product @frankel:diffgeo.
$
  m^k (u, v) &= inner(u, v)_(L^2 Lambda^k (Omega))
  quad
  u in L^2 Lambda^k, v in L^2 Lambda^k
$

The next bilinear form involves the exterior derivative @frankel:diffgeo
$
  d^k (u, v) &= inner(dif u, v)_(L^2 Lambda^k (Omega))
  quad
  u in H Lambda^(k-1), v in L^2 Lambda^k
$

The bilinear form involving the codifferential is also relevant @frankel:diffgeo
$
  c(u, v) &= inner(delta u, v)_(L^2 Lambda^k (Omega))
$
Using the adjoint property of the codifferential relative to the exterior
derivative under the $L^2$ inner product @frankel:diffgeo, @douglas:feec-book,
it can be rewritten using the exterior derivative
applied to the test function.
$
  c^k (u, v) &= inner(u, dif v)_(L^2 Lambda^k (Omega))
  quad
  u in L^2 Lambda^k, v in H Lambda^(k-1)
$

Lastly there is the bilinear form analogous to the scalar Laplacian, involving
exterior derivatives on both arguments. It corresponds to the $delta dif$ part of the Hodge-Laplacian @frankel:diffgeo.
$
  l^k (u, v) &= inner(dif u, dif v)_(L^2 Lambda^(k+1) (Omega))
  quad
  u in H Lambda^k, v in H Lambda^k
$

After Galerkin discretization @hiptmair:numpde, by means of the Whitney finite element space @whitney:geointegration, @douglas:feec-article
with the Whitney basis ${phi_i^k}$, we arrive at the following Galerkin matrices for our
four weak operators.
$
  amat(M)^k &= [inner(phi^k_i, phi^k_j)]_(i j) \
  amat(D)^k &= [inner(phi^k_i, dif phi^(k-1)_j)]_(i j) \
  amat(C)^k &= [inner(dif phi^(k-1)_i, phi^k_j)]_(i j) \
  amat(L)^k &= [inner(dif phi^k_i, dif phi^k_j)]_(i j) \
$

We can rewrite the 3 operators that involve the exterior derivative
using the mass matrix and the discrete exterior derivative (coboundary/incidence matrix) @douglas:feec-article, @crane:ddg.
$
  amat(D)^k &= amat(M)^k amat(dif)^(k-1) \
  amat(C)^k &= (amat(dif)^(k-1))^transp amat(M)^k \
  amat(L)^k &= (amat(dif)^k)^transp amat(M)^(k+1) amat(dif)^k \
$

As usual in a FEM library @hiptmair:numpde, we define element matrix providers,
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
the transposed incidence matrix (coboundary operator) for any $k$-skeleton of a $n$-complex @crane:ddg.

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


=== Mass bilinear form

Now we need to implement the element matrix provider to the mass bilinear form.
Here is where the geometry of the domain comes in, through the inner product, which
depends on the Riemannian metric @frankel:diffgeo.

One could also understand the mass bilinear form as a weak Hodge star operator @douglas:feec-book.
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
coordinate functions are constant @douglas:feec-article. This makes the wedge big terms also constant.
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
    let simplices: Vec<_> = subsimplices(dim, grade).collect();

    let wedge_terms: Vec<_> = simplices
      .iter()
      .cloned()
      .map(|simp| WhitneyForm::new(SimplexCoords::standard(dim), simp).wedge_terms())
      .collect();

    let scalar_mass = ScalarMassElmat.eval(geometry);

    let mut elmat = na::DMatrix::zeros(simplices.len(), simplices.len());
    for (i, asimp) in simplices.iter().enumerate() {
      for (j, bsimp) in simplices.iter().enumerate() {
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
this integral, which only depends on the volume of the cell @hiptmair:numpde, @douglas:feec-book.
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
based on this the right dimension of simplices are used to assemble.
The assembly process itself is a standard FEM technique @hiptmair:numpde.

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
