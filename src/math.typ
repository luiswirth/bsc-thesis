#let avec(a) = math.upright(math.bold(a))
#let amat(a) = math.upright(math.bold(a))

#let vvec(a) = math.accent(math.bold(a), math.arrow)
#let nvec(a) = math.accent(avec(a), math.hat)

#let xv = $avec(x)$
#let ii = $dotless.i$

#let linf(a) = math.sans(a)
#let bilf(a) = math.sans(a)

#let grad = $avec("grad")$

#let inner(a, b) = $lr(angle.l #a, #b angle.r)$

#let conj(u) = math.overline(u)
#let transp = math.tack.b
#let hert = math.upright(math.sans("H"))

#let clos(a) = math.overline(a)
#let restr(a) = $lr(#a|)$
#let openint(a,b) = $lr(\] #a, #b \[)$

#let argmin = math.op("arg min", limits: true)
#let argmax = math.op("arg max", limits: true)

#let mesh = $cal(M)$
#let nodes = $cal(N)$

#let Sdisc = $cal(S)^(-1)_0$
#let S0p = $cal(S)^0_p$
#let S10 = $cal(S)_1^0$
#let S100 = $cal(S)_(1,0)^0$

#let S0p0 = $cal(S)^0_(p,0)$

#let S0pM = $S0p(mesh)$
#let S0pMt = $S0p(mesh')$
#let S10M = $S10(mesh)$
#let S100M = $S100(mesh)$
#let S0p0M = $S0p0(mesh)$


#let math-template(doc) = [
  #set math.mat(delim: "[")
  #set math.vec(delim: "[")
  #set math.cancel(stroke: red)

  // make equation cites only display the number
  #show ref: it => {
    let eq = math.equation
    let el = it.element
    if el != none and el.func() == eq {
      numbering(
        el.numbering,
        ..counter(eq).at(el.location())
      )
    } else {
      it
    }
  }

  #doc
]
