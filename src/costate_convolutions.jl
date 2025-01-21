function Xhat_tf_no_convolution(
    t_f::Tt, z::SVector{n, Tz},
    x_0::SVector{n, Tx},
    f::Function, f_params;
    solver                   = Vern9(),
    reltol                   = 1e-14,
    abstol                   = 1e-14,
) where {n, Tt, Tz, Tx}
    return solve(
        ODEProblem{false, SciMLBase.FullSpecialize}(
            f,
            vcat(x_0, z),
            (0.0, t_f),
            f_params,
        ),
        solver;
        reltol          = reltol,
        abstol          = abstol,
        save_everystep  = false,
        save_start      = false,
    )[end]
end

function Xhat_tf_no_convolution(
    ts::SVector{m, Tt}, z::SVector{n, Tz},
    x_0::SVector{n, Tx},
    f::Function, f_params;
    solver                   = Vern9(),
    reltol                   = 1e-14,
    abstol                   = 1e-14,
) where {n, m, Tt, Tz, Tx}
    TX = promote_type(Tt, Tz, Tx)
    ts_states = Vector{SVector{n, TX}}(undef, m)
    for j in eachindex(ts)
        X0, tj = if j == 1
            (vcat(x_0, z), ts[j])
        else
            (ts_states[j - 1], ts[j] - ts[j - 1])
        end

        ts_states[j] =  solve(
            ODEProblem{false, SciMLBase.FullSpecialize}(
                f,
                X0,
                (0.0, tj),
                f_params,
            ),
            solver;
            reltol          = reltol,
            abstol          = abstol,
            save_everystep  = false,
            save_start      = false,
        )[end]
    end
    return StaticArrays.sacollect(
        SMatrix{2*n, m},
        ts_states[j][k] for k in 1:2*n, j in 1:m
    )
end

function Xhat_tf(
    ts, z::SVector{n, Tz}, κ,
    x_0::SVector{n, Tx},
    σs_max::SVector{n, Float64},
    f::Function, f_params = nothing;
    quad::AbstractQuadrature = GaussHermite(),
    N                        = 50,
    solver                   = Vern9(),
    reltol                   = 1e-14,
    abstol                   = 1e-14,
) where {n, Tz, Tx}
    # Get weights and nodes for quadrature
    ns, ws = get_weights_and_nodes(N, quad)

    if κ >= 1.0
        return Xhat_tf_no_convolution(
            ts, z, x_0,
            f, f_params;
            solver = solver,
            reltol = reltol,
            abstol = abstol,
        )
    else
        # Call recursive function to compute Xhat_tf
        return Xhat_tf(
            n, ts, z, κ,
            x_0, σs_max,
            ns, ws, quad,
            f, f_params;
            solver = solver,
            reltol = reltol,
            abstol = abstol,
        )
    end
end

function Xhat_tf(
    i::Int,
    t_f::Tt, z::SVector{n, Tz}, κ::Tκ,
    x_0::SVector{n, Tx},
    σs_max::SVector{n, Float64},
    ns::Vector{Float64}, ws::Vector{Float64},
    quad::AbstractQuadrature,
    f::Function, f_params;
    solver                   = Vern9(),
    reltol                   = 1e-14,
    abstol                   = 1e-14,
) where {n, Tt, Tz, Tκ, Tx}
    TX = promote_type(Tt, Tz, Tκ, Tx)
    if i == 0
        return Xhat_tf_0(
            t_f, z, κ, x_0,
            f, f_params;
            solver = solver,
            reltol = reltol,
            abstol = abstol,
        )
    else
        # Compute σ_i
        σ_i = σs_max[i]*(1.0 - κ)

        # Compute Xhat_tf
        Xhat_tf_i = @SVector zeros(TX, 2*n)
        for j in eachindex(ns)
            # Perturbed z
            zp = perturb_z(z, i, σ_i, ns[j], quad)

            # Compute Xhat_tf_im1
            Xhat_tf_im1 = Xhat_tf(
                i-1, t_f, zp, κ,
                x_0, σs_max,
                ns, ws, quad,
                f, f_params;
                solver = solver,
                reltol = reltol,
                abstol = abstol,
            )

            # Update Xhat_tf for quadrature
            sf = get_quadrature_sum_scale_factor(σ_i, ns[j], quad)
            Xhat_tf_i = Xhat_tf_i + sf*ws[j]*Xhat_tf_im1
        end
        # Finish computation of quadrature
        sf = get_quadrature_scale_factor(quad)
        return sf*Xhat_tf_i
    end
end

function Xhat_tf(
    i::Int,
    ts::SVector{m, Tt}, z::SVector{n, Tz}, κ::Tκ,
    x_0::SVector{n, Tx},
    σs_max::SVector{n, Float64},
    ns::Vector{Float64}, ws::Vector{Float64},
    quad::AbstractQuadrature,
    f::Function, f_params;
    solver                   = Vern9(),
    reltol                   = 1e-14,
    abstol                   = 1e-14,
) where {n, m, Tt, Tz, Tκ, Tx}
    TX = promote_type(Tt, Tz, Tκ, Tx)
    if i == 0
        return Xhat_tf_0(
            ts, z, κ, x_0,
            f, f_params;
            solver = solver,
            reltol = reltol,
            abstol = abstol,
        )
    else
        # Compute σ_i
        σ_i = σs_max[i]*(1.0 - κ)

        # Compute Xhat_tf
        Xhat_tf_i = @SMatrix zeros(TX, 2*n, m)
        for j in eachindex(ns)
            # Perturbed z
            zp = perturb_z(z, i, σ_i, ns[j], quad)

            # Compute Xhat_tf_im1
            Xhat_tf_im1 = Xhat_tf(
                i-1, ts, zp, κ,
                x_0, σs_max,
                ns, ws, quad,
                f, f_params;
                solver = solver,
                reltol = reltol,
                abstol = abstol,
            )

            # Update Xhat_tf for quadrature
            sf = get_quadrature_sum_scale_factor(σ_i, ns[j], quad)
            Xhat_tf_i = Xhat_tf_i + sf*ws[j]*Xhat_tf_im1
        end
        # Finish computation of quadrature
        sf = get_quadrature_scale_factor(quad)
        return sf*Xhat_tf_i
    end
end
