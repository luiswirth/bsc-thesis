#!/usr/bin/env sh

typst watch src/main.typ out/thesis.pdf --root $(pwd) &
zathura out/thesis.pdf
