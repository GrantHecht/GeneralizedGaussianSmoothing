
function gaussian_final_state_convolution_eoms(
    u::SVector{n4, T}, p, t, f::Function,
) where {n4, T}
    # Get half n2
    n2 = div(n4, 2)

    # Get parameters
    t_f     = p[1]
    σ_0     = p[2]
    f_ps    = p[3]

    # Compute standard dynamics
    du_ns = f(u[StaticArrays.SUnitRange(1,n2)], f_ps, t)

    # Compute Gaussian convolution integrand
    sf = erf((t_f - t) / (sqrt2 * σ_0))
    du_gs = sf*du_ns
    return [du_ns; du_gs]
end

function Xhat_tf_0(
    t_f, z::SVector{n, Tz}, κ,
    x_0::SVector{n, Tx},
    f::Function, f_params;
    σ_0max = t_f / 3.0,
    solver = Vern9(),
    reltol = 1e-14,
    abstol = 1e-14,
) where {n, Tz, Tx}
    # Compute σ_0 from κ
    σ_0 = σ_0max * (1.0 - κ)

    # Form full initial state/costate vector
    X = vcat(x_0, z)

    # Form the initial convolution state/costate vector
    sf = erf(t_f / (sqrt2 * σ_0))
    Xhat = sf * X

    # Form eoms
    leom = let f = f
        (u, p, t) -> gaussian_final_state_convolution_eoms(u, p, t, f)
    end

    # Perform integration
    XXhat_tf = solve(
        ODEProblem{false, SciMLBase.FullSpecialize}(
            leom,
            vcat(X, Xhat),
            (0.0, t_f),
            (t_f, σ_0, f_params),
        ),
        solver;
        reltol          = reltol,
        abstol          = abstol,
        save_everystep  = false,
        save_start      = false
    )[end]
    return XXhat_tf[StaticArrays.SUnitRange(2*n + 1, 4*n)]
end

function Xhat_tf_0(
    ts::SVector{m, Tt}, z::SVector{n, Tz}, κ,
    x_0::SVector{n, Tx},
    f::Function, f_params;
    σ_0maxs = map(j -> j == 1 ? ts[j] / 3.0 : (ts[j] - ts[j - 1]) / 3.0, 1:m),
    solver  = Vern9(),
    reltol  = 1e-14,
    abstol  = 1e-14,
) where {n, m, Tt, Tz, Tx}
    # Handle types
    TX = promote_type(Tt, Tz, Tx)

    # Form eoms
    leom = let f = f
        (u, p, t) -> gaussian_final_state_convolution_eoms(u, p, t, f)
    end

    ts_states = Vector{SVector{4*n, TX}}(undef, m)
    for j in eachindex(ts)
        X_t0, t_fj = if j == 1
            # Get first final time
            t_fj = ts[j]

            # Form full initial state/costate vector
            X = vcat(x_0, z)

            (X, t_fj)
        else
            # Get next final time
            t_fj = ts[j] - ts[j - 1]

            # Get initial state/costate vector
            X = ts_states[j - 1][SOneTo(2*n)]

            (X, t_fj)
        end

        # Compute σ_0 from κ
        σ_0 = σ_0maxs[j] * (1.0 - κ)

        # Form the initial convolution state/costate vector
        sf = erf(t_fj / (sqrt2 * σ_0))
        Xhat_t0 = sf * X_t0

        # Form full initial vector
        XXhat_t0 = vcat(X_t0, Xhat_t0)

        # Perform integration
        ts_states[j] = solve(
            ODEProblem{false, SciMLBase.FullSpecialize}(
                leom,
                XXhat_t0,
                (0.0, t_fj),
                (t_fj, σ_0, f_params),
            ),
            solver;
            reltol          = reltol,
            abstol          = abstol,
            save_everystep  = false,
            save_start      = false
        )[end]
    end
    return StaticArrays.sacollect(
        SMatrix{2*n, m},
        ts_states[j][k] for k in 2*n + 1:4*n, j in 1:m
    )
end
