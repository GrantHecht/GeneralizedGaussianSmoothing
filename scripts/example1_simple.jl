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

global const sqrt2 = sqrt(2)

# Define state and costate dynamics
function nlsys_energy_optimal_eoms(u, p, t)
    x, y, λx, λy = u
    T = 2*λy
    dx = 4*y
    dy = 4*T + 8*sin(0.8*x)
    dλx = -6.4*λy*cos(0.8*x)
    dλy = -4*λx
    return SA[dx,dy,dλx,dλy]
end

function gaussian_time_domain_convolution_eoms(u, p, t)
    # Get parameters
    t_f = p[1]
    σ_0 = p[2]

    # Compute standard dynamics
    du_ns = nlsys_energy_optimal_eoms(u[SA[1,2,3,4]], p, t)

    # Compute Gaussian convolution integrand
    sf  = erf((t_f - t) / (sqrt2 * σ_0))
    du_gs = sf*du_ns

    return [du_ns; du_gs]
end

function Xhat_tf_0(
    t_f, z, κ, x_0, σ_0max;
    solver = Vern9(),
    reltol = 1e-14,
    abstol = 1e-14,
)
    # Compute σ_0 from κ
    σ_0 = σ_0max * (1.0 - κ)

    # Form the initial state/costate vector
    X = [x_0; z]

    # Form the initial smoothed state/costate vector
    sf = erf(t_f / (sqrt2 * σ_0))
    Xhat = sf * X

    # Perform integration
    XXhat_tf = solve(
        ODEProblem{false, SciMLBase.FullSpecialize}(
            gaussian_time_domain_convolution_eoms,
            [X; Xhat],
            (0.0, t_f),
            (t_f, σ_0),
        ),
        solver;
        reltol          = reltol,
        abstol          = abstol,
        save_everystep  = false,
        save_start      = false
    )[end]
    return XXhat_tf[SA[5,6,7,8]]
end

function Xhat_tf(
    t_f, z, κ, x_0;
    N       = 200,
    σ_0max  = t_f / 3.0,
    σ_1max  = 2.0 / 3.0,
    σ_2max  = 2.0 / 3.0,
    solver  = Vern9(),
    reltol  = 1e-14,
    abstol  = 1e-14,
)
    if κ < 1.0
        # Compute σs
        σ_1 = σ_1max*(1.0 - κ)
        σ_2 = σ_2max*(1.0 - κ)

        # Compute Δs
        Δ_1 = 3.0*σ_1
        Δ_2 = 3.0*σ_2

        # Compute quadrature nodes and weights
        ns, ws = gausslegendre(N)
        #ns, ws = gausshermite(N)

        # Compute Xhat_tf_2
        Xhat_tf_2 = SA[0.0, 0.0, 0.0, 0.0]
        for idx_2 in 1:N
            # Compute Xhat_tf_1 for the current idx_2
            Xhat_tf_1 = SA[0.0, 0.0, 0.0, 0.0]
            for idx_1 in 1:N
                # Perturbed costates
                zp = SA[z[1] + Δ_1*ns[idx_1], z[2] + Δ_2*ns[idx_2]]
                #zp = SA[
                #    z[1] + sqrt2*σ_1*ns[idx_1],
                #    z[2] + sqrt2*σ_2*ns[idx_2],
                #]

                # Compute Xhat_tf_0
                Xhat_0 = Xhat_tf_0(
                    t_f, zp, κ, x_0, σ_0max;
                    solver = solver,
                    reltol = reltol,
                    abstol = abstol,
                )

                # Update Xhat_tf_1 for quadrature
                sf = exp(-Δ_1^2*ns[idx_1]^2 / (2.0*σ_1^2))
                Xhat_tf_1 = Xhat_tf_1 + sf*ws[idx_1]*Xhat_0
                #Xhat_tf_1 = Xhat_tf_1 + ws[idx_1]*Xhat_0
            end

            # Finish approximation of convolution for Xhat_tf_1
            Xhat_tf_1 = (Δ_1 / (sqrt(2.0*π)*σ_1))*Xhat_tf_1
            #Xhat_tf_1 = (1.0 / sqrt(π))*Xhat_tf_1

            # Update Xhat_tf_2 for quadrature
            sf = exp(-Δ_2^2*ns[idx_2]^2 / (2.0*σ_2^2))
            Xhat_tf_2 = Xhat_tf_2 + sf*ws[idx_2]*Xhat_tf_1
            #Xhat_tf_2 = Xhat_tf_2 + ws[idx_2]*Xhat_tf_1
        end

        # Finish approximation of convolution for Xhat_tf_2
        Xhat_tf_2 = (Δ_2 / (sqrt(2.0*π)*σ_2))*Xhat_tf_2
        #Xhat_tf_2 = (1.0 / sqrt(π))*Xhat_tf_2

        return Xhat_tf_2
    else
        return solve(
            ODEProblem(
                nlsys_energy_optimal_eoms,
                [x_0; z],
                (0.0, t_f),
                (0.0, 0.0),
            ),
            solver;
            reltol          = reltol,
            abstol          = abstol,
            save_everystep  = false,
            save_start      = false
        )[end]
    end
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
    # x_tf = Xhat_tf(
    #     t_f, SA[z[1], z[2]], κ, x_0;
    #     N       = N,
    #     solver  = solver,
    #     reltol  = reltol,
    #     abstol  = abstol,
    # )
    x_tf = Xhat_tf(
        t_f, SA[z[1], z[2]], κ, x_0,
        t_f / 3.0, SA[2.0 / 3.0, 2.0 / 3.0],
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
            display(z_sol)
        else
            println("Failed to solve for κ = $κ")
            break
        end
    end
    return z_sol
end

main()

# N = 20
# ns, ws = gausslegendre(N)

# κ = 0.1
# σ = (1.0 / 3.0)*(1.0 - κ)

# ts = (ns .+ 1.0) / 2.0

# sol = solve(
#     ODEProblem{false}(
#         nlsys_energy_optimal_eoms,
#         SA[0.0, -10.0, 13.686, 15.319],
#         (0.0, 1.0)
#     ),
#     Vern9();
#     reltol = 1e-14,
#     abstol = 1e-14,
#     saveat = ts,
# )

# Xquad = SA[0.0, 0.0, 0.0, 0.0]
# for i in 1:N
#     sf = exp(-(ts[i] - 1.0)^2/ (2.0*σ^2))
#     global Xquad = Xquad + sf*ws[i]*sol.u[i]
# end
# Xquad = (1.0/(sqrt(2*π)*σ))*Xquad

# Xint = Xhat_tf_0(
#     1.0, SA[13.686, 15.319], κ,
#     SA[0.0, -10.0], 1.0/3.0
# )

# Xint - Xquad
