#import "../setup.typ": *
#import "../setup-math.typ": *
#import "../layout.typ": *

= Results


In this chapter, we present numerical results to verify the functionality and
validate the implementation of the `formoniq` library.

Specifically, we examine the Hodge-Laplacian eigenvalue problem on a domain with
non-trivial topology and perform a convergence study for the Hodge-Laplacian
source problem using the Method of Manufactured Solutions on a $n$-dimensional
hypercube domain.

== 1-form Eigenvalue Problem on Torus

To assess the library's ability to handle *non-trivial topologies* as well
as *globally curved geometry*, we solve a eigenvalue problem to compute the
spectrum of the 1-form Hodge-Laplacian $Delta^1$ on a torus $TT^2$.

A mesh of a torus with major radius $r_1=0.5$ and minor radius $r_2=0.2$ was
generated using Gmsh @GmshPaper2009, by specifying a `.geo` file with the line:
```
Torus(1) = {-0, -0, 0, 0.5, 0.2, 2*Pi};
```

The topology of the torus is characterized by its first *Betti number* $b_1 =
2$, which predicts a two-dimensional kernel for $Delta^1$ spanned by harmonic
1-forms representing the two fundamental cycles of the domain.

The theoretical eigenvalues of $Delta^1$ on an idealized flat torus with
circumferences $L_1 = 2 pi r_1 = pi$ and $L_2 = 2 pi r_2 = 0.4 pi$ are given by
$lambda_(m,n) = ((2 pi m)/L_1)^2 + ((2 pi n)/L_2)^2 = 4m^2 + 25n^2$ for integers $m, n in ZZ$.

The lowest computed eigenvalues obtained using formoniq and
the SLEPc @SLEPcPaper2005 eigensolver are:
```
ieigen=0, eigenval=-0.000
ieigen=1, eigenval=0.000
ieigen=2, eigenval=4.116
ieigen=3, eigenval=4.116
ieigen=4, eigenval=4.116
ieigen=5, eigenval=4.116
ieigen=6, eigenval=14.447
ieigen=7, eigenval=14.447
ieigen=8, eigenval=14.449
ieigen=9, eigenval=14.449
ieigen=10, eigenval=24.649
```

These numerical results show excellent agreement with theoretical expectations:
- The computed spectrum correctly identifies the two zero eigenvalues
  $lambda=0.000$ corresponding to the two harmonic 1-forms $m=n=0$, accurately
  capturing the torus's topology ($b_1=2$).
- The first non-zero eigenvalue group is computed as $lambda approx 4.116$
  with multiplicity 4. This closely matches the theoretical value
  $lambda_(1,0)=4$ and its expected multiplicity.
- The second non-zero group is computed around $lambda approx 14.447 approx 14.449$ with
  multiplicity 4, corresponding well to the theoretical value
  $lambda_(2,0)=16$ and its expected multiplicity.
- The next computed eigenvalue $lambda approx 24.649$ aligns closely with
  the theoretical value $lambda_(0,1)=25$.

@img:evp_torus shows visualizations of the two harmonic 1-forms, represented by
their vector proxies.

#figure(
  grid(
    columns: (1fr, 1fr),
    rows: 1,
    gutter: 3pt,
    image("../../res/torus_eigen0_full.png", width: 100%),
    image("../../res/torus_eigen1_full.png", width: 100%),
  ),
  caption: [
    The two harmonic forms on the torus, representing the 1-cohomology spaces.
  ],
) <img:evp_torus>

The slight deviations between computed values and the ideal flat torus
eigenvalues are expected due to discretization error. However, the accurate
recovery of the zero eigenvalues and the correct multiplicities for the lowest
eigenvalue groups strongly validates the FEEC implementation for eigenvalue
problems on domains with non-trivial topology.

This example program which computes the eigenvalue works on any mesh. The
program asks for the `.obj` or `.msh` (gmsh) file, then loads it in and computes
the eigenpairs. The FE eigenfunction is then sampled at each barycenter and the
values are written into a file, which can then be visualized, with for instance
paraview.

The example program can be run using
```sh
cargo run --release --example hodge_laplace_source
```

== Source Problem

We verify the source problem by means of the *method of manufactured solution*
@hiptmair:numpde.
We restrict ourselves to a simple setup in globally flat geometry on a subset
of $RR^n$. We validate only the 1-form source problem, but we do this in arbitrary
dimensions $n >= 2$.

Our domain is
$
  Omega = [0,pi]^n
$

For our manufactured solution, we've chosen a simple 1-form $u in Lambda^1
(Omega)$ that generalizes easily to arbitrary dimensions.
$
  u = sum_(i=1)^n u_i dif x^i
  quad "with" quad
  u_i = sin^2 (x^i) product_(j != i) cos(x^j)
$

When using vector proxies $u^sharp$ we get the following vector fields in the 2D
and 3D case.
$
  n=2 ==> u^sharp = vec(
    sin^2(x) cos(y),
    cos(x) sin^2(y),
  )
  quad quad
  n=3 ==> u^sharp = vec(
    sin^2(x) cos(y) cos(z),
    cos(x) sin^2(y) cos(z),
    cos(x) cos(y) sin^2(z),
  )
$

We can analytically derive the exterior derivative $dif u$ of this solution $u$.
$
  dif u = sum_(k<i) [(product_(j !=i,k) cos(x^j)) sin(x^i) sin(x^k) (sin(x^k) - sin(x^i))] dif x^k wedge dif x^i
$

In the method of manufactured solution, the source term $f$ is set
to equal the 1-form Hodge-Laplacian $Delta^1 u$ of the exact solution $u$.
$
  f = Delta^1 u
$

The corresponding source term $f = Delta^1 u$ is computed analytically:
$
  (Delta^1 avec(u))_i = Delta^0 u_i = -(2 cos(2 x^i) - (n-1) sin^2(x^i)) product_(j != i) cos(x^j)
$

Our solution and it's derivative have boundary traces that are equal to zero.
This leads to homogeneous natural boundary conditions, meaning no additional
terms are required in the variational formulation.
$
  trace_(diff Omega) u = 0
  quad quad
  trace_(diff Omega) dif u = 0
$


We use formoniq to solve this problem and determine the rate of convergence for
dimensions 2 and 3, but even higher dimensions would work.

For each dimension we generate finer and finer meshes with mesh width $h$ halved in each
step. The meshes are generated using our custom tensor-product triangulation
algorithm. We compute the right hand side vector by assembling the element
matrix provider for the source term based on the exact Laplacian.
Then we just call the `solve_hodge_laplace_source` routine.

We compute the $L^2$ norm of the error in the function value $norm(u -
u_h)_(L^2)$. we do this by computing the pointwise error norm $norm(u -
u_h)_(Lambda^k)$ and integrating it using quadrature of order 3, which is
sufficient for the quadrature error to not dominate the finite element error
@hiptmair:numpde. We also compute the $L^2$ error in the exterior derivative,
$norm(dif u - dif u_h)_(L^2)$ using the same approach.

```rust
pub fn fe_l2_error<E: ExteriorField>(
  fe_cochain: &Cochain,
  exact: &E,
  topology: &Complex,
  coords: &MeshCoords,
) -> f64 {
  let dim = topology.dim();
  let qr = SimplexQuadRule::order3(dim);
  let fe_whitney = WhitneyForm::new(fe_cochain.clone(), topology, coords);
  let inner = multi_gramian(&Gramian::standard(dim), fe_cochain.dim());
  let error_pointwise = |x: CoordRef, cell: SimplexHandle| {
    inner.norm_sq((exact.at_point(x) - fe_whitney.eval_known_cell(cell, x)).coeffs())
  };
  qr.integrate_mesh(&error_pointwise, topology, coords).sqrt()
}
```

This is the code for our convergence test.
It can be run using `cargo run --release --example hodge_laplace_source`.
```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
  tracing_subscriber::fmt::init();
  let path = "out/laplacian_source";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

  let grade = 1;
  let homology_dim = 0;

  for dim in 2_usize..=3 {
    println!("Solving Hodge-Laplace in {dim}d.");

    let solution_exact = DiffFormClosure::one_form(
      |p| {
        Vector::from_iterator(
          p.len(),
          (0..p.len()).map(|i| {
            let prod = p.remove_row(i).map(|a| a.cos()).product();
            p[i].sin().powi(2) * prod
          }),
        )
      },
      dim,
    );

    let dif_solution_exact = DiffFormClosure::new(
      Box::new(move |p: CoordRef| {
        let dim = p.len();
        let ncomponents = if dim > 1 { dim * (dim - 1) / 2 } else { 0 };
        let mut components = Vec::with_capacity(ncomponents);

        let sin_p: Vec<_> = p.iter().map(|&pi| pi.sin()).collect();
        let cos_p: Vec<_> = p.iter().map(|&pi| pi.cos()).collect();

        for k in 0..dim {
          for i in (k + 1)..dim {
            let mut prod_cos = 1.0;
            #[allow(clippy::needless_range_loop)]
            for j in 0..dim {
              if j != i && j != k {
                prod_cos *= cos_p[j];
              }
            }
            let coeff = prod_cos * sin_p[i] * sin_p[k] * (sin_p[k] - sin_p[i]);
            components.push(coeff);
          }
        }
        ExteriorElement::new(components.into(), dim, 2)
      }),
      dim,
      2,
    );

    let laplacian_exact = DiffFormClosure::one_form(
      |p| {
        Vector::from_iterator(
          p.len(),
          (0..p.len()).map(|i| {
            let prod: f64 = p.remove_row(i).map(|a| a.cos()).product();
            -(2.0 * (2.0 * p[i]).cos() - (p.len() - 1) as f64 * p[i].sin().powi(2)) * prod
          }),
        )
      },
      dim,
    );

    println!(
      "| {:>2} | {:8} | {:>7} | {:>8} | {:>7} |",
      "k", "L2 err", "L2 conv", "Hd err", "Hd conv",
    );

    let mut errors_l2 = Vec::new();
    let mut errors_hd = Vec::new();
    for irefine in 0..=(15 / dim as u32) {
      let refine_path = &format!("{path}/refine{irefine}");
      fs::create_dir_all(refine_path).unwrap();

      let nboxes_per_dim = 2usize.pow(irefine);
      let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, PI);
      let (topology, coords) = box_mesh.compute_coord_complex();
      let metric = coords.to_edge_lengths(&topology);

      let source_data = assemble_galvec(
        &topology,
        &metric,
        SourceElVec::new(&laplacian_exact, &coords, None),
      );

      let (_, galsol, _) = hodge_laplace::solve_hodge_laplace_source(
        &topology,
        &metric,
        source_data,
        grade,
        homology_dim,
      );

      let conv_rate = |errors: &[f64], curr: f64| {
        errors
          .last()
          .map(|&prev| algebraic_convergence_rate(curr, prev))
          .unwrap_or(f64::INFINITY)
      };

      let error_l2 = fe_l2_error(&galsol, &solution_exact, &topology, &coords);
      let conv_rate_l2 = conv_rate(&errors_l2, error_l2);
      errors_l2.push(error_l2);

      let dif_galsol = galsol.dif(&topology);
      let error_hd = fe_l2_error(&dif_galsol, &dif_solution_exact, &topology, &coords);
      let conv_rate_hd = conv_rate(&errors_hd, error_hd);
      errors_hd.push(error_hd);

      println!(
        "| {:>2} | {:<8.2e} | {:>7.2} | {:<8.2e} | {:>7.2} |",
        irefine, error_l2, conv_rate_l2, error_hd, conv_rate_hd
      );
    }
  }

  Ok(())
}
```


The output is
```
Solving Hodge-Laplace in 2d.
|  k | L2 err   | L2 conv |   Hd err | Hd conv |
|  0 | 1.96e0   |     inf | 6.51e-1  |     inf |
|  1 | 1.57e0   |    0.31 | 4.31e-1  |    0.60 |
|  2 | 8.02e-1  |    0.97 | 3.51e-1  |    0.30 |
|  3 | 4.03e-1  |    0.99 | 1.35e-1  |    1.37 |
|  4 | 1.99e-1  |    1.02 | 6.00e-2  |    1.17 |
|  5 | 9.91e-2  |    1.01 | 2.89e-2  |    1.05 |
|  6 | 4.95e-2  |    1.00 | 1.43e-2  |    1.01 |
|  7 | 2.48e-2  |    1.00 | 7.15e-3  |    1.00 |
Solving Hodge-Laplace in 3d.
|  k | L2 err   | L2 conv |   Hd err | Hd conv |
|  0 | 3.66e0   |     inf | 1.09e0   |     inf |
|  1 | 2.56e0   |    0.52 | 1.76e0   |   -0.69 |
|  2 | 1.46e0   |    0.80 | 7.49e-1  |    1.23 |
|  3 | 7.71e-1  |    0.93 | 3.08e-1  |    1.28 |
|  4 | 3.85e-1  |    1.00 | 1.39e-1  |    1.15 |
|  5 | 1.92e-1  |    1.00 | 6.73e-2  |    1.04 |
```

We observe experimental convergence $cal(O) (h^1)$ for the $L^2$ error of
the exterior derivative $norm(dif u - dif u_h)_(L^2)$ and converge $cal(O) (h^1)$ for
the $L^2$ error of the solution value itself $norm(u - u_h)_(L^2)$.

The observed rate $alpha_(H(dif)) = 1$ is promising. It
matches the convergence rate expected for the $norm(dif dot)_(L^2)$
component within the $H Lambda$ energy norm for first-order Whitney
elements, as predicted by theory @douglas:feec-book. This suggests a valid
implementation of the discretization related to the exterior derivative
operator.

The interpretation of the $alpha_(L^2) = 1$ rate is less
straightforward. While this rate is compatible with an overall $cal(O)(h^1)$
convergence in the full $H Lambda$ norm, standard FEEC theory often predicts
$cal(O)(h^2)$ convergence for the $L^2$ error itself via an Aubin-Nitsche duality
argument @douglas:feec-book. The reason why this higher rate is not observed
in our results requires further investigation, potentially related to the
quadrature rules used for error computation or other implementation details.

It should be noted that this study constitutes only a partial validation of the
implementation. A complete verification would require measuring the error in
the codifferential, $norm(delta(u - u_h))_(L^2)$, to assess convergence in
the the full $H Lambda$ norm or the associated energy norm.
$
  norm(dot)_(H Lambda) =
  norm(dot)_(L^2) + norm(dot)_(H (dif)) + norm(dot)_(H (delta))
$
This analysis was not performed here, due to time constraints.

In @img:source_problem we provide a visualization of the 2D finite element
solution at refinement level 4 our library has produced in the form of a vector
field proxy together with a heat map of the magnitude.
#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    rows: 1,
    gutter: 3pt,
    [],
    image("../../res/source_problem.png", width: 100%),
    [],
  ),
  caption: [
    Finite element solution to manufactured source problem in 2D at refinement level 4.
  ],
) <img:source_problem>

