
global const sqrt2      = sqrt(2)
global const sqrt2pi    = sqrt(2*π)
global const sqrtpi     = sqrt(π)

abstract type AbstractQuadrature end
struct GaussLegendre <: AbstractQuadrature end
struct GaussHermite <: AbstractQuadrature end

get_weights_and_nodes(N, ::GaussLegendre) = gausslegendre(N)
get_weights_and_nodes(N, ::GaussHermite)  = gausshermite(N)

function perturb_z(
    z::SVector{n, Tz},
    i::Int,
    σ_i::Tσ,
    n_j::Float64,
    ::GaussLegendre,
) where {n, Tz, Tσ}
    pval = 3.0*σ_i*n_j
    pert = SVector{n, Tσ}(OneElement(pval, i, n))
    return z + pert
end

function perturb_z(
    z::SVector{n, Tz},
    i::Int,
    σ_i::Tσ,
    n_j::Float64,
    ::GaussHermite,
) where {n, Tz, Tσ}
    pval = sqrt2*σ_i*n_j
    pert = SVector{n, Tσ}(OneElement(pval, i, n))
    return z + pert
end

function get_quadrature_sum_scale_factor(
    σ_i::Tσ,
    n_j::Float64,
    ::GaussLegendre,
) where {Tσ}
    return exp(-(3.0*σ_i)^2*n_j^2 / (2.0 * σ_i^2))
end
function get_quadrature_sum_scale_factor(
    σ_i::Tσ,
    n_j::Float64,
    ::GaussHermite,
) where {Tσ}
    return 1.0
end

function get_quadrature_scale_factor(::GaussLegendre)
    return 3.0 /  sqrt2pi
end
function get_quadrature_scale_factor(::GaussHermite)
    return 1.0 / sqrtpi
end
