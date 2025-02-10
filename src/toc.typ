#show outline.entry.where(
  level: 1
): it => {
  v(15pt, weak: true)
  strong(it.body)
  h(1fr)
  strong(it.page)
}

#show outline: set heading(outlined: true)

#outline(
  title: "Table of Contents",
  indent: auto,
  fill: line(length: 100%, stroke: 0.2pt + white),
)
