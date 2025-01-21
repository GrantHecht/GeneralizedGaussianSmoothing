
using DrWatson
@quickactivate "GaussianSmoothingHomotopy"

using StaticArrays
using SpecialFunctions
using DifferentialEquations
using NonlinearSolve
using FastGaussQuadrature
using LinearAlgebra

using Infiltrator
using GLMakie

include(srcdir("includes.jl"))

function min_fuel_landing_ode(u, p, t)
    # Get params
    κ     = p[1]
    T_min = p[2]
    T_max = p[3]
    v_ex  = p[4]

    # useful terms
    r       = norm(u[SA[1,2,3]])
    inv_r3  = 1.0 / r^3
    inv_r5  = inv_r3 / r^2
    nλv     = norm(u[SA[11,12,13]])
    Γ       = u[7] - 0.1 < 0.0 ? 0.0 : 1.0

    # Optimal thrust direction
    α  = u[SA[11,12,13]] / nλv

    # Optimal throttling (with gaussian kernel-based smoothing)
    S = nλv / u[7] - (1 + u[14]) / v_ex
    T = if κ < 1.0
        0.5*(T_max - T_min)*erf(S / (sqrt(2)*(1 - κ))) + 0.5*(T_max + T_min)
    else
        S >= 0.0 ? T_max : T_min
    end

    # state dynamics
    dr = u[SA[4,5,6]]
    dv = (T/u[7])*α - u[SA[1,2,3]] * inv_r3
    dm = -(T / v_ex)*Γ

    # costate dynamics
    dλr = -(3.0*dot(u[SA[11,12,13]], u[SA[1,2,3]])*inv_r5)*u[SA[1,2,3]] +
        inv_r3*u[SA[11,12,13]]
    dλv = -u[SA[8,9,10]]
    dλm = (T*dot(u[SA[11,12,13]], α) / u[7]^2)*Γ

    return SA[
        dr[1], dr[2], dr[3],
        dv[1], dv[2], dv[3],
        dm,
        dλr[1], dλr[2], dλr[3],
        dλv[1], dλv[2], dλv[3],
        dλm,
    ]
end

function shooting_fun!(
    F, y, κ,
    x0,
    xf,
    ti, ri,
    T_min, T_max, v_ex;
    N       = 10,
    solver  = Vern9(),
    reltol  = 1e-14,
    abstol  = 1e-14,
)
    # Compute smoothed states
    Xhats = Xhat_tf(
        SA[ti, y[end]], SA[y[1], y[2], y[3], y[4], y[5], y[6], y[7]], κ,
        x0, @SVector(fill(2.0 / 3.0, 7)),
        min_fuel_landing_ode, (κ, T_min, T_max, v_ex);
        solver  = solver,
        reltol  = reltol,
        abstol  = abstol,
    )

    # transversality conditions
    S = norm(x_tf[SA[11,12,13]]) / x_tf[7] - (1 + x_tf[14]) / v_ex
    T = if κ < 1.0
        0.5*(T_max - T_min)*erf(S / (sqrt(2)*(1 - κ))) + 0.5*(T_max + T_min)
    else
        S >= 0.0 ? T_max : T_min
    end

    tc1 = dot(x_tf[SA[8,9,10]], x_tf[SA[4,5,6]]) -
        dot(x_tf[SA[11,12,13]], x_tf[SA[1,2,3]]) / norm(x_tf[SA[1,2,3]])^3 +
        T*norm(x_tf[SA[11,12,13]]) / x_tf[7] - 1.0
    tc2 = x_tf[14]


    F[1] = x_ti[1] - ri[1]
    F[2] = x_ti[2] - ri[2]
    F[3] = x_ti[3] - ri[3]
    F[4] = x_tf[1] - rf[1]
    F[5] = x_tf[2] - rf[2]
    F[6] = x_tf[3] - rf[3]
    F[7] = x_tf[4] - vf[1]
    F[8] = x_tf[5] - vf[2]
    F[9] = x_tf[6] - vf[3]
    F[10] = tc1
    F[11] = tc2

    return nothing
end

# units
DU = 1.7374e6
VU = 1677.6733889526888
TU = 1035.6008573782028
MU = 874.4
g0 = 1.62

r0 = SA[1753000.0, 0.0, 0.0] / DU
v0 = SA[400.0,800.0,1100.0] / VU
m0 = 1.0

rf = SA[1669376.733,338459.298,345355.8] / DU
vf = SA[0.0, 0.0, 0.0]

ti = 600.0 / TU
ri = SA[1672000.0,336000.0,340000.0] / DU

T_min = 880 * TU^2 / (MU*DU)
T_max = 2200 * TU^2 / (MU*DU)
v_ex = 3090.15 / VU

# Test shooting function
F = zeros(11)
y = [-2.802,1.148,0.072,-0.649,0.209,-0.215,-1.125,-5.699,1.500,-0.696,0.879]
# shooting_fun!(
#     F, y, 0.0,
#     r0, v0, m0,
#     rf, vf,
#     ti, ri,
#     T_min, T_max, v_ex;
#     solver  = Vern9(),
#     reltol  = 1e-14,
#     abstol  = 1e-14,
# )

# tf_sol = 0.638
# λ0_sol = [-1.205,0.337,-0.375,-0.265,-0.035,-0.322,-0.490]
# α = [-4.197,3.115,-5.441]

# ode_sol1 = solve(
#     ODEProblem{false}(
#         min_fuel_landing_ode,
#         SA[
#             r0[1], r0[2], r0[3],
#             v0[1], v0[2], v0[3],
#             m0,
#             λ0_sol...
#         ],
#         (0.0, ti),
#         (0.0, T_min, T_max, v_ex),
#     ),
#     Vern9();
#     reltol          = 1e-14,
#     abstol          = 1e-14,
# )

# pert = SA[
#     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#     α[1], α[2], α[3], 0.0, 0.0, 0.0, 0.0,
# ]
# ode_sol2 = solve(
#     ODEProblem{false}(
#         min_fuel_landing_ode,
#         ode_sol1.u[end] + pert,
#         (0.0, tf_sol - ti),
#         (1.0, T_min, T_max, v_ex),
#     ),
#     Vern9();
#     reltol          = 1e-14,
#     abstol          = 1e-14,
# )

# fig = Figure()
# ax = Axis3(fig[1, 1]; aspect = :data)

# lines!(ax, ode_sol1[1,:], ode_sol1[2,:], ode_sol1[3,:], color = :blue)
# lines!(ax, ode_sol2[1,:], ode_sol2[2,:], ode_sol2[3,:], color = :red)

# scatter!(ax,
#     [r0[1],ri[1],rf[1]],
#     [r0[2],ri[2],rf[2]],
#     [r0[3],ri[3],rf[3]],
# )
# fig
