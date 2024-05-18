#!/usr/bin/env sh

typst watch src/main.typ out/thesis.pdf --root $(pwd) &

while :
do
  zathura out/thesis.pdf
done
