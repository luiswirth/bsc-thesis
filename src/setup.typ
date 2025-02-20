//#import "@preview/cetz:0.2.1"

#import "setup-math.typ": *
#import "setup-code.typ": *

//#let fgcolor = white
//#let bgcolor = black
#let fgcolor = black
#let bgcolor = white

#let thesis-template(doc) = {
  show: math-template
  show: code-template
  
  set page(fill: bgcolor)
  set text(fill: fgcolor)

  set page(paper: "a4")
  set page(margin: 1cm)

  set text(font: "New Computer Modern Sans")
  set text(size: 12pt)

  //set par(justify: true)
 
  doc
}


#let weblink(..args) = text(
  fill: blue,
  link(..args)
)

#let monospaced(content) = text(font: "Fira Code", content)

#let preface-style(doc) = {
  set page(numbering: "I")

  set heading(numbering: none)

  show heading: it => {
    if it.level == 1 {
      //pagebreak(weak: true)
      v(70pt)
      text(size: 25pt)[#it.body]
      v(40pt)
    } else {
      let size = 20pt - 4pt * (it.level - 1)
      set text(size, weight: "bold")
      v(size, weak: true)
      //counter(heading).display()
      h(size, weak: true)
      it.body
      v(size, weak: true)
    }
  }

  doc
}

#let body-style(doc) = {
  set page(numbering: "1")
  counter(page).update(1)

  set heading(numbering: "1.1.1")

  show heading: it => {
    if it.level == 1 {
      pagebreak(weak: true)
      v(60pt)
      text(size: 18pt)[Chapter #counter(heading).display()]
      v(0pt)
      text(size: 25pt)[#it.body]
      v(30pt)
    } else {
      let size = 20pt - 4pt * (it.level - 1)
      set text(size, weight: "bold")
      v(size, weak: true)
      counter(heading).display()
      h(size, weak: true)
      it.body
      v(size, weak: true)
    }
  }

  doc
}

#let appendix-style(doc) = {
  set page(numbering: "1")

  set heading(numbering: "A.1.1")
  counter(heading).update(0)

  show heading: it => {
    if it.level == 1 {
      pagebreak(weak: true)
      v(60pt)
      text(size: 18pt)[Appendix #counter(heading).display()]
      v(0pt)
      text(size: 25pt)[#it.body]
      v(30pt)
    } else {
      let size = 20pt - 4pt * (it.level - 1)
      set text(size, weight: "bold")
      v(size, weak: true)
      counter(heading).display()
      h(size, weak: true)
      it.body
      v(size, weak: true)
    }
  }

  doc
}

#let postface-style(doc) = {
  set page(numbering: "i")
  counter(page).update(1)

  set heading(numbering: none)

  show heading: it => {
    if it.level == 1 {
      pagebreak(weak: true)
      text(size: 25pt)[#it.body]
      v(50pt, weak: true)
    } else {
      let size = 20pt - 4pt * (it.level - 1)
      set text(size, weight: "bold")
      v(size, weak: true)
      counter(heading).display()
      h(size, weak: true)
      it.body
      v(size, weak: true)
    }
  }

  doc
}
