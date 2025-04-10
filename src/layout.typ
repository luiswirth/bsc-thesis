#import "setup.typ": *

#let finishline = line(stroke: red + 0.3pt, length: 100%)

#let titlebox(color: red.darken(40%), width: 100%, title, content) = block(
  fill: color,
  stroke: fgcolor,
  inset: 0.3em,
  radius: 0.3em,
  breakable: false,
  [
    #block(inset: 0.2em)[#pad(x: 1.0em)[*#title*]]
    #block(
      fill: black.lighten(10%),
      inset: 1.2em,
      above: 0.3em,
      width: width,
      [
        #content
      ]
    )
  ]
)

#let theorembox(content) = block(
  stroke: 0.5pt,
  inset: 5pt,
  breakable: false,
  content
)

#let circletext(content) = box(
  baseline: 0.2em,
  stroke: fgcolor,
  inset: 0.2em,
  radius: 0.3em,
  content,
)

