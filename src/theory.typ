#import "math.typ": *
#import "layout.typ": *

= Theory

// Notations needed:
// Space of all alternating l-multilinear forms on some vector space.
// Tangent space at point x at Omega.
// Space of all Differential l-forms on Omega.
// Space of all integrable l-forms on Omega.
// Space of all C^k l-forms on Omega.

#let Lin = $"Lin"$
#let Alt = $"Alt"$

== Exterior Algebra
=== Multilinear Forms
=== Alternating Forms
== Exterior Calculus
=== Differential Forms

In the most general sense, $Omega$ may be a (piecewise) smooth oriented and
bounded $n$-dimensional Riemannian manifold, $n in NN$, with a piecewise smooth
boundary.
A differential form $omega in wedgespace^l (Omega)$ of order $l$, $0 <= l <= n$, is a mapping from
$Omega$ into the $binom(n,l)$ -dimensional space $Alt^l (T_Omega (xv))$ of
alternating $l$–multilinear forms on the $n$-dimensional tangent space $T_Omega
(xv)$ at $Omega$ in $xv in Omega$.
@hiptmair-whitney

A fundamental concept in the theory of differential forms is the integral of
a $p$-form over a piecewise smooth $p$-dimensional oriented manifold. Through
integration a differential form assigns a value to each suitable manifold,
modeling the physical procedure of measuring a field.
@hiptmair-whitney

From alternating $l$-multilinear forms differential $l$-forms inherit the
exterior product \
$wedge: wedgespace^l (Omega) times wedgespace^k (Omega) -> wedgespace^(l+k) (Omega), 0 <= l, k, l + k <= n$,
defined in a pointwise sense.
Moreover, remember that the trace $t_Sigma omega$ of an $l$-form $omega in
wedgespace^l (Omega), 0 <= l < n$, onto some piecewise smooth ($n − 1$)-
dimensional submanifold $Sigma subset.eq clos(Omega)$ yields an $l$-form on $Sigma$.
It can be introduced by restricting $omega (xv) in Alt^l (T_Omega
(xv)), x in Sigma$, to the tangent space of $Sigma$ in $xv$.
The trace commutes with the exterior product and exterior differentiation, i.e.
$dif t_Sigma omega = t_Sigma dif omega$ for $omega in wedgespace^l_1 (Omega)$.
@hiptmair-whitney


Another crucial device is the exterior derivative $dif$, a linear
mapping from differentiable $l$-forms into ($l + 1$)-forms.
Stokes’ theorem makes it possible to define the exterior derivative
$dif omega in wedgespace^(l+1) (Omega)$ of $omega in wedgespace^l (Omega)$.
#theorembox[
  *Theorem (Stokes):*
  $ integral_Omega dif omega = integral_(diff Omega) omega $
  for all $omega in wedgespace^l_1 (Omega)$
]
A fundamental fact about exterior differentiation is that $dif(dif omega) = 0$
for any sufficiently smooth differential form $omega$.

Under some restrictions on the topology of $Omega$ the converse is
also true, which is called the exact sequence property:
\
#theorembox[
  *Theorem (Poincaré's lemma):*
  For a contractible domain $Omega subset.eq RR^n$ every
  $omega in wedgespace^l_1 (Omega), l >= 1$, with $dif omega = 0$ is the exterior
  derivative of an ($l − 1$)–form over $Omega$.
]
\
A second main result about the exterior derivative is the following formula
#theorembox[
  *Theorem (Integration by parts):*
  $
    integral_Omega dif omega wedge eta
    + (-1)^l integral_Omega omega wedge dif eta
    = integral_(diff Omega) omega wedge eta
  $
  for $omega in wedgespace^l (Omega), eta in wedgespace^k (Omega), 0 <= l, k < n − 1, l + k = n − 1$.
]
Here, the boundary $diff Omega$ is endowed with the induced orientation.
Finally, we recall the pullback $Omega |-> Phi^*omega$ under a change of
variables described by a diffeomorphism $Phi$. This transformation commutes
with both the exterior product and the exterior derivative, and it leaves the
integral invariant.


Remember that given a basis
$dif x_0, dots, dif x_n$ of the dual space of $T_Omega (xv)$ the set of all
exterior products of these
furnishes a basis for the space of alternating l-multilinear forms on
TΩ(x). Thus any ω ∈ Dl(Ω) has a representation
$
  omega = sum_(i_1, dots, i_l) phi_(i_1,dots,i_l) dif x_i_1 wedge dots.c wedge dif x_i_l
$
where the indices run through all combinations admissible according
to (6) and the ϕi1,... ,il : Ω �→ R are coefficient functions.
Therefore, we call a differential form polynomial of degree k,
k ∈ N0, if all its coefficient functions in (7) are polynomials of degree k.


We can define proxies to convert between vector fields and differential forms.
Sharp #sharp to move from differential form to vector field.
Flat #flat to move from vector field to differential form.

=== Exterior Derivative
=== Integration
=== Stockes' Theorem

== Finite Element Differential Forms

Lagragian (vertex), Raviart-Thomas (edge), Nédélec (face)

=== Complete Polynomial Spaces of Differential Forms
=== Whitney Forms

Whitney 0-form $cal(W) x_i$ corresponding to 0-simplex $x_i$ is the barycentric function $cal(W) x_i = lambda_i$.

The whitney $p$-form corresponding to the $p$-simplex $[x_0, dots, x_p]$ is
$
  cal(W) [x_0, dots, x_p] =
  p! sum_(i=0)^p (-1)^i lambda_i
  (dif lambda_0 wedge dots.c wedge hat(dif lambda_i) wedge dots.c wedge dif lambda_p)
$

As example in a triangle $K = [x_0,x_1,x_2]$ the whitney forms are
$
  cal(W) [x_0] = lambda_0
  wide
  cal(W) [x_1] = lambda_1
  wide
  cal(W) [x_2] = lambda_2
  \
  cal(W) [x_0,x_1] = lambda_0 dif lambda_1 - lambda_1 dif lambda_0
  wide
  cal(W) [x_0,x_2] = lambda_0 dif lambda_2 - lambda_2 dif lambda_0
  wide
  cal(W) [x_1,x_2] = lambda_1 dif lambda_2 - lambda_2 dif lambda_1
  \
  cal(W) [x_0,x_1,x_2] = 2 (lambda_0 (dif lambda_1 wedge dif lambda_2) - lambda_1 (dif lambda_0 wedge dif lambda_2) + lambda_2 (dif lambda_0 wedge dif lambda_1))
$

Whitney forms are affine invariant. \
Let $K_1 = [x_0, dots x_n]$ and $K_2 = [y_0, dots y_n]$ and $phi: K_1 -> K_2$ affine map, such that $phi(x_i) = y_i$,
then
$W[x_0, dots, x_n] = phi^* (W[y_0, dots y_n])$

== Various

Exterior derivative: $dif_l: dom(d_l) subset.eq L^2 wedgebig^l (Omega) -> L^2 wedgebig^(l+1)$

$L^2$-adjoint: $delta_l := dif _(l-1)^* = (-1)^l star_(l-1)^(-1) compose dif_(N-l) compose star_l$

