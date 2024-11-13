#!/bin/bash

export JUPYTER_BOOK_BUILD=true
jupyter-book clean --html .
jupyter-book build .
