# GaussianSmoothingHomotopy

An implementation of the method proposed by Pan et al. in *Generalized Gaussian smoothing homotopy method for solving nonlinear optimal control problems* [doi: 10.1016/j.actaastro.2024.12.051](https://doi.org/10.1016/j.actaastro.2024.12.051)

Note only example 1 from the paper is working and produces the expected results.

## Notes on setting up 

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> GaussianSmoothingHomotopy

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "GaussianSmoothingHomotopy"
```
which auto-activate the project and enable local path handling from DrWatson.
