#import "../setup.typ": *
#import "../setup-math.typ": *
#import "../layout.typ": *

= Results

To verify the function of the library we solve a EVP and a source problem based on the Hodge-Laplacian operator, using the Finite Element Exterior Calculus framework @douglas:feec-book, @douglas:feec-article.

== 1-Form EVP on Annulus

We meshed a 2D annulus $BB_1 (0) \\ BB_(1\/4) (0)$ using Gmsh @GmshPaper2009.

The eigenvalues computed on the annulus correspond to the actual eigenvalues.

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    rows: 1,
    gutter: 3pt,
    image("../../res/evp0.png", width: 100%),
    image("../../res/evp5.png", width: 100%),
    image("../../res/evp6.png", width: 100%),
  ),
  caption: [
    Three interesting computed eigenfunctions for the Hodge-Laplacian eigenvalue problem on an annulus.
  ],
) <img:evp>

== 1-Form Source Problem on $RR^n, n >= 1$

We verify the source problem by means of the method of manufactured solution @hiptmair:numpde.
Our manufactured solution is a 1-form that follows the same pattern for any
dimensions.

$
  Omega = [0,pi]^n
$

$
  u = sum_(i=1)^n u_i dif x^i
  quad "with" quad
  u_i = sin^2 (x^i) product_(j != i) cos(x^j)
$

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

$
  dif u = sum_(k<i) [(product_(j !=i,k) cos(x^j)) sin(x^i) sin(x^k) (sin(x^k) - sin(x^i))] dif x^k wedge dif x^i
$


The corresponding source term $f = Delta^1 u$ is computed analytically:
$
  (Delta^1 avec(u))_i = Delta^0 u_i = -(2 cos(2 x^i) - (n-1) sin^2(x^i)) product_(j != i) cos(x^j)
$

Homogeneous boundary conditions are imposed.
$
  trace_(diff Omega) u = 0
  quad quad
  trace_(diff Omega) dif u = 0
$

The manufactured solution is chosen such that it is non-trivial, meaning it is neither curl-free nor divergence-free in general.
$
  curl avec(u) != 0
  quad quad
  div avec(u) != 0
$

The implementation uses the libraries `nalgebra` @crate:nalgebra, `PETSc` @PETScManualRecent, and `SLEPc` @SLEPcPaper2005 for numerical computations and solving the resulting linear systems.
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
theory predicts for lowest-order Whitney form elements @douglas:feec-article, confirming the correct implementation.


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
    Computed solution $u$ to the 1-form source problem using the method of manufactured solutions.
  ],
) <img:source_problem>
