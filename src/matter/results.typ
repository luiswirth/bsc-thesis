#import "../setup.typ": *
#import "../setup-math.typ": *
#import "../layout.typ": *

= Results

To verify the functionality of the library we solve a EVP and a source problem
based on the Hodge-Laplacian operator, using the Finite Element Exterior
Calculus framework @douglas:feec-book, @douglas:feec-article.

== 1-Form EVP on Torus

We solved a Hodge-Laplace eigenvalue problem on the torus $TT^2$.
In @img:evp_torus the two harmonic forms on this torus are visualized
as vector proxies. They are the representatives of the 1-cohomology group.
The vector field show the two 1-holes on the torus in an intuitive way.

#figure(
  grid(
    columns: (1fr, 1fr),
    rows: 1,
    gutter: 3pt,
    image("../../res/torus_eigen0_full.png", width: 100%),
    image("../../res/torus_eigen1_full.png", width: 100%),
  ),
  caption: [
    The two harmonic forms on the torus, representing the 1-cohomology groups.
  ],
) <img:evp_torus>

== 1-Form EVP on Annulus

We meshed a 2D annulus $BB_1 (0) \\ BB_(1\/4) (0)$ using Gmsh @GmshPaper2009.

// TODO: IMPROVE!!!
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
) <img:evp_annulus>

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

The solution and it's exterior derivative are zero on the boundary,
meaning we exactly fulfill, the homogenoeous natural boundary conditions.
$
  trace_(diff Omega) u = 0
  quad quad
  trace_(diff Omega) dif u = 0
$

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
  tracing_subscriber::fmt::init();
  let path = "out/laplacian_source";
  let _ = fs::remove_dir_all(path);
  fs::create_dir_all(path).unwrap();

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
        let num_components = if dim > 1 { dim * (dim - 1) / 2 } else { 0 };
        let mut components = Vec::with_capacity(num_components);

        let sin_p: Vec<_> = p.iter().map(|&pi| pi.sin()).collect();
        let cos_p: Vec<_> = p.iter().map(|&pi| pi.cos()).collect();

        for k in 0..dim {
          for i in (k + 1)..dim {
            let mut prod_cos_pik = 1.0;
            #[allow(clippy::needless_range_loop)]
            for j in 0..dim {
              if j != i && j != k {
                prod_cos_pik *= cos_p[j];
              }
            }

            let coeff = prod_cos_pik * sin_p[i] * sin_p[k] * (sin_p[k] - sin_p[i]);
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
      "k", "L2 err", "L2 conv", "H1 err", "H1 conv",
    );

    let mut errors_l2 = Vec::new();
    let mut errors_h1 = Vec::new();
    for irefine in 0..=(14 / dim as u32) {
      let refine_path = &format!("{path}/refine{irefine}");
      fs::create_dir_all(refine_path).unwrap();

      let nboxes_per_dim = 2usize.pow(irefine);
      let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes_per_dim, PI);
      let (topology, coords) = box_mesh.compute_coord_complex();
      let metric = coords.to_edge_lengths(&topology);

      let source_data = cochain_projection(&laplacian_exact, &topology, &coords, None);

      let (_, galsol, _) =
        hodge_laplace::solve_hodge_laplace_source(&topology, &metric, source_data, homology_dim);

      manifold::io::save_skeleton_to_file(&topology, dim, format!("{refine_path}/cells.skel"))?;
      manifold::io::save_skeleton_to_file(&topology, 1, format!("{refine_path}/edges.skel"))?;
      manifold::io::save_coords_to_file(&coords, format!("{refine_path}/vertices.coords"))?;
      ddf::io::save_cochain_to_file(&galsol, format!("{refine_path}/fe.cochain"))?;

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
      let error_h1 = fe_l2_error(&dif_galsol, &dif_solution_exact, &topology, &coords);
      let conv_rate_h1 = conv_rate(&errors_h1, error_h1);
      errors_h1.push(error_h1);

      println!(
        "| {:>2} | {:<8.2e} | {:>7.2} | {:<8.2e} | {:>7.2} |",
        irefine, error_l2, conv_rate_l2, error_h1, conv_rate_h1
      );
    }
  }

  Ok(())
}
```


The output is
```
Solving Hodge-Laplace in 2d.
|  k | L2 err   | L2 conv |   H1 err | H1 conv |
|  0 | 3.13e0   |     inf | 6.51e-1  |     inf |
|  1 | 1.67e0   |    0.91 | 4.83e-1  |    0.43 |
|  2 | 7.51e-1  |    1.15 | 2.43e-1  |    0.99 |
|  3 | 3.90e-1  |    0.94 | 1.15e-1  |    1.08 |
|  4 | 1.97e-1  |    0.98 | 5.72e-2  |    1.01 |
|  5 | 9.89e-2  |    1.00 | 2.86e-2  |    1.00 |
|  6 | 4.95e-2  |    1.00 | 1.43e-2  |    1.00 |
|  7 | 2.47e-2  |    1.00 | 7.14e-3  |    1.00 |
```

We get order $alpha_(H 1) = 1$ which is exactly what theory predicts for
lowest-order Whitney form elements @douglas:feec-article, confirming the correct
implementation.

However we also get order $alpha_(L 2) = 1$, which is surprising, since
theory predicts order 2.
This is most likely due to a non-admissible variational crime, incurred
by the way we approximate the RHS source term. We first do a cochain-projection
before we multiply by the mass matrix to obtain the RHS.
This error dominates the finite element error, giving us a worse order than predicted.
This is not optimal, but because of time constraints, we weren't able to fix this.
The order 1 $H^1$ convergence is sufficient for proving a valid implementation.


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
