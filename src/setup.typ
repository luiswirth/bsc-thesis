#import "@preview/cetz:0.2.1"

#import "math.typ": *
#import "layout.typ": *

#let monospaced(content) = text(font: "DejaVu Sans Mono", content)

#let weblink(url, content) = monospaced(link(url, content))

#let general-style(doc) = {
  show: math-template

  set page(paper: "a4")
  set text(size: 10pt)
  set text(font: "New Computer Modern")

  //#set page(margin: 1cm)
  //#set par(justify: true)

  //#set page(fill: black)
  //#set text(white)
 
  doc
}

#let preface-style(doc) = {
  set page(numbering: "I")

  set heading(numbering: none)

  show heading: it => {
    set text(font: "New Computer Modern")
    if it.level == 1 {
      //pagebreak()
      v(70pt)
      text(size: 25pt)[#it.body]
      v(40pt)
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

#let body-style(doc) = {
  set page(numbering: "1")
  counter(page).update(1)

  set heading(numbering: "1.1.1")

  show heading: it => {
    set text(font: "New Computer Modern")
    if it.level == 1 {
      pagebreak()
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
    set text(font: "New Computer Modern")
    if it.level == 1 {
      pagebreak()
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
    set text(font: "New Computer Modern")
    if it.level == 1 {
      //pagebreak()
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
