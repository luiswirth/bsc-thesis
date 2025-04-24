#import "../setup-math.typ": *
#import "../layout.typ": *
#import "../setup.typ": *

= Exterior Algebra

Exterior algebra is to exterior calculus, what vector algebra is to vector
calculus @frankel:diffgeo.\
In vector calculus we have vector fields $v$, which are functions
$v: p in Omega |-> v_p$ over the manifold $Omega$ that at each point
$p$ have a constant vector $v_p in T_p M$ as value.\
In exterior calculus we have differential forms $omega$, which are functions
$omega: p in Omega |-> omega_p$ over the manifold $Omega$ that at each point $p$
have a constant *multiform* $omega_p in wedgespace (T^*_p M)$ as value @frankel:diffgeo, @douglas:feec-book.

If one were to implement something related to vector calculus
it is of course crucial to be able to represent vectors in the program.
This is usually the job of a basic linear algebra library such as Eigen in `C++`
and nalgebra in Rust @crate:nalgebra.\
Since we want to implement FEEC @douglas:feec-article, which uses exterior calculus,
it is crucial, that we are able to represent multiforms in our program.
For this there aren't any established libraries. So we do this ourselves
and develop a small module.

== Exterior Algebra of Multiforms

In general an exterior algebra $wedgespace (V)$ is a construction over any linear
space $V$ @frankel:diffgeo. In this section we want to quickly look at the specific linear
space we are dealing with when modelling multiforms as element of an
exterior algebra. But our implementation would work for any finite-dimensional
real linear space $V$ with a given basis.

In our particular case we have the exterior algebra of alternating multilinear
forms $wedgespace (T^*_p M)$ @frankel:diffgeo. Here the linear space $V$ is the cotangent space
$T^*_p M$ of the manifold $M$ at a point $p in M$. It's the dual space $(T_p
M)^*$ of the tangent space $T_p M$.
The elements of the cotangent space are covectors $a in T^*_p M$, which are
linear functionals $a: T_p M -> RR$ on the tangent space.
The tangent space $T_p M$ has the standard basis ${diff/(diff x^1)}_(i=1)^n$
induced by some chart map $phi: p in U subset.eq M |-> (x_1,dots,x_n)$. This
gives rise to a dual basis ${dif x^i}_(i=1)^n$ of the cotangent space, defined
by $dif x^i (diff/(diff x^j)) = delta^i_j$ @frankel:diffgeo.

There is a related space, called the space of multivectors $wedgespace (T_p M)$,
which is the exterior algebra over the tangent space, instead of the cotangent space.
The space of multivectors and multiforms are dual to each other @frankel:diffgeo.
$
  wedgespace (T^*_p M) =^~ (wedgespace (T_p M))^*
$
The space of multivectors only plays a minor role in exterior calculus, since it
is not metric independent. We just wanted to quickly mentioned it here.

It is common practice to call the elements of any exterior algebra
multivectors, irregardless what the underlying linear space $V$ is.
This is confusing when working with multiforms, which are distinct from multivectors.
To avoid confusion, we therefore just call the elements of the exterior algebra
exterior elements or multielements, just like we say linear space instead of
vector space.

== The Numerical Exterior Algebra $wedgespace (RR^n)$

When working with vectors from a finite-dimensional real linear space $V$, then
we can always represent them computationally, by choosing a basis
${e_i}_(i=1)^n subset.eq V$.
This constructs an isomorphism $V =^~ RR^n$, where $n = dim V$.
This allows us to work with elements $avec(v) in RR^n$, which have real
values $v_i in RR$ as components, which are the basis coefficients.
These real numbers are what we can work with on computers and allow
us to do numerical linear algebra @hiptmair:numcse.
This means that when working with any finite-dimensional real linear space $V$
on a computer we always just use the linear space $RR^n$.

The same idea can be used to computationally work with exterior algebras @crane:ddg.
By choosing a basis of $V$, we also get an isomorphism on the exterior algebra
$wedgespace (V) =^~ wedgespace (RR^n)$.
Therefore our implementation will be we directly on $wedgespace (RR^n)$.

For our space of multiforms, we will be using the standard cotangent basis
${dif x^i}_(i=1)^n$.

== Representing Exterior Elements

An exterior algebra is a graded algebra @frankel:diffgeo.
$
  wedgespace (RR^n) = wedgespace^0 (RR^n) plus.circle.big dots.c plus.circle.big wedgespace^n (RR^n)
$
Each element $v in wedgespace (RR^n)$
has some particular exterior grade $k in {1,dots,n}$ and therefore lives in
a particular exterior power $v in wedgespace^k (RR^n)$.
We make use of this fact in our implementation, by splitting the representation
between these various grades.
```rust
pub type ExteriorGrade = usize;
```

For representing an element in a particular exterior power
$wedgespace^k (RR^n)$, we use the fact that, it itself is a linear space in it's
own right.
Due to the combinatorics of the anti-symmetric exterior algebra, we have $dim
wedgespace^k (RR^n) = binom(n,k)$ @frankel:diffgeo.
This means that by choosing a basis ${e_I}$ of this exterior power, we can just
use a list of $binom(n,k)$ coefficients to represent an exterior element, by
using the isomorphism $wedgespace^k (RR^n) =^~ RR^binom(n,k)$.
```rust
/// An element of an exterior algebra.
#[derive(Debug, Clone)]
pub struct ExteriorElement {
  coeffs: na::DVector<f64>,
  dim: Dim,
  grade: ExteriorGrade,
}
```
This struct represents an element `self` $in wedgespace^k (RR^n)$ with
`self.dim` $= n$, `self.grade` $= k$ and `self.coeffs.len()` $= binom(n,k)$.

This exterior basis ${e_I}_(I in cal(I)^n_k)$ is different from the basis
${e_i}_(i=1)^n$ of the original linear space $V$, but is best subsequently
constructed from it.
We do this by creating elementary multielements from the
exterior product of basis elements @frankel:diffgeo.
$
  e_I = wedge.big_(j=1)^k e_I_j = e_i_1 wedge dots.c wedge e_i_k
$
Here $I = (i_1,dots,i_k)$ is a multiindex, in particular a $k$-index, telling
us which basis elements to wedge.

Because of the anti-symmetry of the exterior product, there are certain conditions
on the multiindices $I$ for ${e_I}$ to constitute a meaningful basis @frankel:diffgeo.
First $I$ must not contain any duplicate indices, because otherwise $e_I = 0$
and second there must not be any permutations of the same index in the
basis set, otherwise we have linear dependence of the two elements.
We therefore only consider strictly increasing multiindices $I in cal(I)^n_k$
and denote their set by
$cal(I)^n_k = {(i_1,dots,i_k) in NN^k mid(|) 1 <= i_1 < dots.c < i_k <= n}$.
This is a good convention for supporting arbitrary dimensions @douglas:feec-book.

The basis also needs to be ordered, such that we can know which coefficient
in `self.coeffs` corresponds to which basis. A natural choice here is
a lexicographical ordering @crane:ddg.

Taking in all of this together we for example have as exterior basis for
$wedge.big^2 (RR^3)$ the elements $e_1 wedge e_2, e_1 wedge e_3, e_2 wedge e_3$.

== Representing Exterior Terms

It is helpful to represent these exterior basis wedges in our program.
```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExteriorTerm {
  indices: Vec<usize>,
  dim: Dim,
}
impl ExteriorTerm {
  pub fn dim(&self) -> Dim { self.dim }
  pub fn grade(&self) -> ExteriorGrade { self.indices.len() }
}
```
This struct allows for any multiindex, even if they are not strictly increasing.
But we are of course able to check whether this is the case or not and
then to convert it into a increasing representation plus the permutation sign.
We call this representation, canonical.
```rust
pub fn is_basis(&self) -> bool {
  self.is_canonical()
}
pub fn is_canonical(&self) -> bool {
  let Some((sign, canonical)) = self.clone().canonicalized() else {
    return false;
  };
  sign == Sign::Pos && canonical == *self
}
pub fn canonicalized(mut self) -> Option<(Sign, Self)> {
  let sign = sort_signed(&mut self.indices);
  let len = self.indices.len();
  self.indices.dedup();
  if self.indices.len() != len {
    return None;
  }
  Some((sign, self))
}
```

In the case of a strictly increasing term, we can also determine the lexicographical
rank of it in the set of all increasing terms. And the other way constructing
them from lexicographical rank.
```rust
pub fn lex_rank(&self) -> usize {
  assert!(self.is_canonical(), "Must be canonical.");
  let n = self.dim();
  let k = self.indices.len();

  let mut rank = 0;
  for (i, &index) in self.indices.iter().enumerate() {
    let start = if i == 0 { 0 } else { self.indices[i - 1] + 1 };
    for s in start..index {
      rank += binomial(n - s - 1, k - i - 1);
    }
  }
  rank
}

pub fn from_lex_rank(dim: Dim, grade: ExteriorGrade, mut rank: usize) -> Self {
  let mut indices = Vec::with_capacity(grade);
  let mut start = 0;
  for i in 0..grade {
    let remaining = grade - i;
    for x in start..=(dim - remaining) {
      let c = binomial(dim - x - 1, remaining - 1);
      if rank < c {
        indices.push(x);
        start = x + 1;
        break;
      } else {
        rank -= c;
      }
    }
  }
  Self::new(indices, dim)
}
```

Now that we have this we can implement a useful iterator on our `ExteriorElement`
struct that allows us to iterate through the basis expansion consisting
of both the coefficient and the exterior basis element.
```rust
pub fn basis_iter(&self) -> impl Iterator<Item = (f64, ExteriorTerm)> + '_ {
  let dim = self.dim;
  let grade = self.grade;
  self
    .coeffs
    .iter()
    .copied()
    .enumerate()
    .map(move |(i, coeff)| {
      let basis = ExteriorTerm::from_lex_rank(dim, grade, i);
      (coeff, basis)
    })
}
```
We then implemented the addition and scalar multiplication of exterior elements
by just applying the operation to the coefficients.

== Exterior Product

The most obvious operation on a `ExteriorElement` is of course the exterior product @frankel:diffgeo.
For this we rely on the exterior product of two `ExteriorTerm`s,
which is just a concatenation of the two multiindices.
```rust
impl ExteriorTerm {
  pub fn wedge(mut self, mut other: Self) -> Self {
    self.indices.append(&mut other.indices);
    self
  }
}
```

For the `ExteriorElement` we just iterate over the all combinations of
basis expansion and canonicalize the wedges of the individual terms.
```rust
impl ExteriorElement {
  pub fn wedge(&self, other: &Self) -> Self {
    assert_eq!(self.dim, other.dim);
    let dim = self.dim;

    let new_grade = self.grade + other.grade;
    if new_grade > dim {
      return Self::zero(dim, 0);
    }

    let new_basis_size = binomial(dim, new_grade);
    let mut new_coeffs = na::DVector::zeros(new_basis_size);

    for (self_coeff, self_basis) in self.basis_iter() {
      for (other_coeff, other_basis) in other.basis_iter() {
        let self_basis = self_basis.clone();

        let coeff_prod = self_coeff * other_coeff;
        if self_basis == other_basis || coeff_prod == 0.0 {
          continue;
        }
        if let Some((sign, merged_basis)) = self_basis.wedge(other_basis).canonicalized() {
          let merged_basis = merged_basis.lex_rank();
          new_coeffs[merged_basis] += sign.as_f64() * coeff_prod;
        }
      }
    }

    Self::new(new_coeffs, dim, new_grade)
  }
}
```

And we also implement a big wedge operator, that takes an iterator of factors.
```rust
pub fn wedge_big(factors: impl IntoIterator<Item = Self>) -> Option<Self> {
  let mut factors = factors.into_iter();
  let first = factors.next()?;
  let prod = factors.fold(first, |acc, factor| acc.wedge(&factor));
  Some(prod)
}
```

== Inner product on Exterior Elements

For the weak formulations of our PDEs @hiptmair:numpde we rely on Hilbert spaces that require
an $L^2$-inner product on differential forms @frankel:diffgeo.
This is derived directly from the point-wise inner product on multiforms.
Which itself is derived from the inner product on the tangent space,
which comes from the Riemannian metric at the point @frankel:diffgeo.

This derivation from the inner product on the tangent space $g_p$
to the inner product on the exterior fiber $wedge.big^k T^*_p M$, shall
be computed.

In general given an inner product on the vector space $V$, we can
derive an inner product on the exterior power $wedgespace^k (V)$ @frankel:diffgeo, @douglas:feec-book.
The rule is the following:
$
  inner(e_I, e_J) = det [inner(dif x_I_i, dif x_I_j)]_(i,j)^k
$

Computationally we represent inner products as Gramian matrices on some basis.
This means that we compute an extended Gramian matrix as the inner product on
multielements from the Gramian matrix of single elements using the determinant.
```rust
impl RiemannianMetricExt for RiemannianMetric {
  fn multi_form_gramian(&self, k: ExteriorGrade) -> na::DMatrix<f64> {
    let n = self.dim();
    let bases: Vec<_> = exterior_bases(n, k).collect();
    let covector_gramian = self.covector_gramian();

    let mut multi_form_gramian = na::DMatrix::zeros(bases.len(), bases.len());
    let mut multi_basis_mat = na::DMatrix::zeros(k, k);

    for icomb in 0..bases.len() {
      let combi = &bases[icomb];
      for jcomb in icomb..bases.len() {
        let combj = &bases[jcomb];

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

We are already at the end of the implementation of the exterior algebra.
There exist many operations that could be implemented as well, such as the
Hodge star operator @frankel:diffgeo, @douglas:feec-book, based on an inner product, but it's just not necessary
for the rest of the library to have such functionality, therefore we omit it
here.
