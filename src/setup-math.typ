#let math-template(doc) = {
  show math.equation: set text(font: "New Computer Modern Math")
  
  set math.mat(delim: "[")
  set math.vec(delim: "[")

  // Make equation referencing only display the number.
  show ref: it => {
    let el = it.element
    if el != none and el.func() == math.equation {
      // Override equation references.
      link(el.location(),numbering(
        el.numbering,
        ..counter(math.equation).at(el.location())
      ))
    } else {
      // Other references as usual.
      it
    }
  }

  doc
}

#let avec(a) = math.upright(math.bold(a))
#let vvec(a) = math.accent(math.bold(a), math.arrow)
#let nvec(a) = math.accent(avec(a), math.hat)

#let amat(a) = math.upright(math.bold(a))

#let xv = $avec(x)$
#let ii = $dotless.i$


#let angled(a) = math.lr($angle.l #a angle.r$)

#let inner(a, b) = angled($#a, #b$)
#let innerlines(a, b) = angled(math.vec(delim: none, a, b))

#let conj(u) = math.overline(u)
#let transp = math.tack.b
#let hert = math.upright(math.sans("H"))

#let clos(a) = math.overline(a)
#let openint(a,b) = $lr(\] #a, #b \[)$

#let argmin = math.op("arg min", limits: true)
#let argmax = math.op("arg max", limits: true)

#let mesh = $cal(M)$


#let wedge = math.and
#let wedgespace = math.scripts(math.inline(wedge.big))
#let hodge = math.class("unary", math.star)
#let sharp = "♯"
#let flat = "♭"

#let dif = math.class("unary", math.upright($d$))

#let grad = $avec("grad")$
#let curl = $avec("curl")$
#let div = $"div"$


#let Hvec = $avec(H)$
#let H0 = $limits(H)^circle.stroked.small$



#let restr(a) = $lr(#a|)$
#let trace = $"Tr"$

#let lin = $"Lin"$
#let alt = $"Alt"$

#let vol = "vol"

#let dom = "dom"
