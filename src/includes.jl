
using StaticArrays
using FillArrays
using SpecialFunctions
using DifferentialEquations
using FastGaussQuadrature

include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "final_state_convolution.jl"))
include(joinpath(@__DIR__, "costate_convolutions.jl"))
