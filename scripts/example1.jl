using DrWatson
@quickactivate "GaussianSmoothingHomotopy"

using StaticArrays
using SpecialFunctions
using DifferentialEquations
using NonlinearSolve
using FastGaussQuadrature

using Infiltrator
using GLMakie

include(srcdir("includes.jl"))

# Define state and costate dynamics
function nlsys_energy_optimal_eoms(u, p, t)
    x, y, λx, λy = u
    T   = 2*λy
    dx  = 4*y
    dy  = 4*T + 8*sin(0.8*x)
    dλx = -6.4*λy*cos(0.8*x)
    dλy = -4*λx
    return SA[dx,dy,dλx,dλy]
end

function shooting_fun!(
    F, z, κ;
    t_f     = 1.0,
    x_0     = SA[0.0, -10.0],
    N       = 10,
    solver  = Vern9(),
    reltol  = 1e-14,
    abstol  = 1e-14,
)
    x_tf = Xhat_tf(
        t_f, SA[z[1], z[2]], κ, x_0,
        SA[2.0 / 3.0, 2.0 / 3.0],
        nlsys_energy_optimal_eoms;
        quad    = GaussHermite(),
        N       = N,
        solver  = solver,
        reltol  = reltol,
        abstol  = abstol,
    )

    F[1] = x_tf[1] - 30.0
    F[2] = x_tf[2] + 10.0

    return nothing
end

function main()
    # Generate random guess for costates
    z_guess = rand(2)

    # Solve the problem with increasing κ
    κs      = range(0.0, 1.0; length = 6)
    z_sol   = copy(z_guess)
    for κ in κs
        println("Solving for κ = $κ")
        prob = NonlinearProblem{true}(
            shooting_fun!, z_sol, κ;
        )
        sol = solve(
            prob,
            TrustRegion(;
                #autodiff = AutoFiniteDiff(),
            );
            show_trace = Val(false),
            trace_level = TraceAll(),
        )

        if SciMLBase.successful_retcode(sol.retcode)
            z_sol .= sol.u
        else
            println("Failed to solve for κ = $κ")
            break
        end
    end
    return z_sol
end

z_sol = main()

# Plot solution
ode_sol = solve(
    ODEProblem{false}(
        nlsys_energy_optimal_eoms,
        SA[0.0, -10.0, z_sol[1], z_sol[2]],
        (0.0, 1.0),
    ),
    Vern9();
    reltol          = 1e-14,
    abstol          = 1e-14,
)

fig = Figure()
ax  = Axis(fig[1, 1])

lines!(ax, ode_sol.t, ode_sol[1, :], color = :blue)
lines!(ax, ode_sol.t, ode_sol[2, :], color = :red)
fig
