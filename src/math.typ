#let avec(a) = math.upright(math.bold(a))
#let amat(a) = math.upright(math.bold(a))

#let vvec(a) = math.accent(math.bold(a), math.arrow)
#let nvec(a) = math.accent(avec(a), math.hat)

#let xv = $avec(x)$
#let ii = $dotless.i$

#let linf(a) = math.sans(a)
#let bilf(a) = math.sans(a)

#let div = $"div"$
#let grad = $avec("curl")$
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

#let wedge = $and$
#let wedgebig = $and.big$
#let wedgespace = $Lambda$
#let sharp = "♯"
#let flat = "♭"

#let dom = "dom"

#let math-template(doc) = [
  #show math.equation: set text(font: "New Computer Modern Math")
  #show math.equation: set text(font: "Fira Math Book")
  
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
