#import "../setup.typ": *
#import "../setup-math.typ": *
#import "../layout.typ": *

= Hodge-Laplacian

In this chapter we now solve some PDEs based on the Hodge-Laplace operator @douglas:feec-book.
We consider the Hodge-Laplace eigenvalue problem and the Hodge-Laplace source
problem (analog of Poisson equation).

The Hodge-Laplace operator generalizes the scalar (0-form) Laplace-Beltrami
operator, to an operator acting on any differential $k$-form @frankel:diffgeo,
@douglas:feec-book. As such the 0-form Hodge-Laplacian $Delta^0$ is exactly
the Laplace-Beltrami operator and we can write it using the exterior derivative
$dif$ and the codifferential $delta$.
$
  Delta^0 f = -div grad f = delta^1 dif^0 f
$

The $k$-form Hodge-Laplacian $Delta^k$ is defined as @douglas:feec-book
$
  Delta^k: Lambda^k (Omega) -> Lambda^k (Omega)
  \
  Delta^k = dif^(k-1) delta^k + delta^(k+1) dif^k
$

== Eigenvalue Problem

We first consider the Eigenvalue problem, because it's a bit simpler
and the source problem, relies on the eigenvalue problem.

The strong primal form of the Hodge-Laplace eigenvalue problem is @douglas:feec-book\
Find $lambda in RR, u in Lambda^k (Omega)$, s.t.
$
  (dif delta + delta dif) u = lambda u
$


In FEM we don't solve the PDE based on the strong form, but instead we rely
on a weak variational form.
The primal weak form is not suited for discretization, so instead we make use of
a mixed variational form that includes an auxiliary variable $sigma$ @douglas:feec-article, @douglas:feec-book.

The mixed weak form is @douglas:feec-article, @douglas:feec-book\
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
as finite dimensional subspace of our function space $H Lambda^k$ the space
of Whitney forms $cal(W) lambda^k subset.eq H Lambda^k$ @douglas:feec-article
and as basis the Whitney basis ${phi^k_i}$ @whitney:geointegration,
@hiptmair:whitneyforms. We then replace $sigma$ and $u$ by basis expansions
$sigma = sum_j sigma_j phi^(k-1)_j$, $u = sum_i u_i phi^k_i$ and arrive at the
linear system of equations.
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
In SLEPc @SLEPcPaper2005 terminology this is called a GHIEP problem.

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

// TODO: RHS vector discussion

The Hodge-Laplace Source Problem is the generalization of the Poisson equation
to arbitrary differential $k$-forms @douglas:feec-book. In strong form it is\
Find $u in Lambda^k (Omega)$, given $f in Lambda^k (Omega)$, s.t.
$
  Delta u = f - P_frak(H) f, quad u perp frak(H)
$
This equation is not quite as simple as the normal Poisson equation $Delta u = f$.
Instead it includes two additional parts involving $frak(H)$, which is the space
of harmonic forms $frak(H)^k = ker Delta = { v in Lambda^k mid(|) Delta v = 0}$ @douglas:feec-book.
The first change is that we remove the harmonic part $P_frak(H) f$ of $f$. The second
difference is that we require that our solution $u$ is orthogonal to harmonic forms.

The harmonic forms give a concrete realizations of the cohomology @frankel:diffgeo @douglas:feec-book.
They are a representative of the cohomology quotient group $cal(H)^k = (ker dif)/(im dif)$
and as such they are isomorphic $frak(H)^k =^~ cal(H)^k$.

We once again tackle a mixed weak formulation based on the auxiliary variable
$sigma$ and this time a second one $p$ that represents $f$ without harmonic
component @douglas:feec-article, @douglas:feec-book.\ Given $f in L^2 Lambda^k$,
find $(sigma,u,p) in (H Lambda^(k-1) times H Lambda^k times frak(H)^k)$ s.t.
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

We once again perform Galerkin discretization @douglas:feec-article.
$
  sum_j sigma_j inner(phi^(k-1)_j,phi^(k-1)_i) - sum_j u_j inner(phi^k_j,dif phi^(k-1)_i) &= 0
  \
  sum_j sigma_j inner(dif phi^(k-1)_j,phi^k_i) + sum_j u_j inner(dif phi^k_j,dif phi^k_i) + sum_j p_j inner(eta^k_j,phi^k_i) &= inner(f,phi^k_i)
  \
  sum_j u_j inner(phi^k_j,eta^k_i) &= 0
$

By inserting our known Galerkin matrices, we obtain.


$
  amat(M)^(k-1) avec(sigma) - amat(C) avec(u) = 0
  \
  amat(D) avec(sigma) + amat(L) avec(u) + amat(M) amat(H) avec(p) = avec(b)
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
  vec(0, avec(b), 0)
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

Where the right-hand side corresponding to the source term is approximated via quadrature.
$
  avec(b) = sum_(sigma in mesh) abs(sigma) sum_l w_l inner(f(avec(q)_l), phi^k_i (avec(q)_l))_(Lambda^k)
  approx integral_mesh inner(f(x), phi^k_i (x))_(Lambda^k) vol
$

Compute harmonics
```rust
pub fn solve_hodge_laplace_harmonics(
  topology: &Complex,
  geometry: &MeshEdgeLengths,
  grade: ExteriorGrade,
  homology_dim: usize,
) -> na::DMatrix<f64> {
  if homology_dim == 0 {
    let nwhitneys = topology.nsimplices(grade);
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
  homology_dim: usize,
) -> (Cochain, Cochain, Cochain) {
  let harmonics = solve_hodge_laplace_harmonics(topology, geometry, grade, homology_dim);

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

