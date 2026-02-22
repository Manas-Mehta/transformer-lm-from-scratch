# NYU Building LLM Reasoners Assignment 2: Systems

This assignment is adapted from Stanford CS336 ([original repository](https://github.com/stanford-cs336/)). All credit for its
development goes to the Stanford course staff. This README and all of the following code are adapted from theirs.

For a full description of the assignment, see the assignment handout at 
[a2.pdf](https://gregdurrett.github.io/courses/sp2026/a2.pdf)

## Setup

This directory is organized as follows:

- [`./a1-basics`](./a1-basics): directory containing a module
  `a1_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. If you want to use your own 
  implementation, you can replace this directory with your own implementation.
- [`./student`](./student): This folder is basically empty! This is the
  module where you will implement your optimized Transformer language model. 
  Feel free to take whatever code you need from assignment 1 (in `student`) and copy it 
  over as a starting point. In addition, you will implement distributed training and
  optimization in this module.
- [`./examples`](./examples): This folder contains benchmarking examples used in lecture,
  mostly adapted from Percy Liang's lecutre. It also contains a lightly modified version of the weighted
  sum implementation from the homework assignment. Feel free to use these as starting points
  for your benchmarking.

Visually, it should look something like:

``` sh
.
├── a1-basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── student  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

If you would like to use your own implementation of assignment 1, replace the `a1-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

0. We use `uv` to manage dependencies. You can verify that the code from the `a1-basics`
package is accessible by running:

```sh
$ uv run python
Using CPython 3.12.10
Creating virtual environment at: /path/to/uv/env/dir
      Built student @ file:///path/to/systems/dir
      Built a1-basics @ file:///path/to/basics/dir
Installed 85 packages in 711ms
Python 3.12.10 (main, Apr  9 2025, 04:03:51) [Clang 20.1.0 ] on linux
...
>>> import a1_basics
>>> 
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

