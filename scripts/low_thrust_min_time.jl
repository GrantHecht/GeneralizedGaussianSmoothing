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

function crtbp_min_time_eom(x, p, t)
    # Grab states
    r_x = x[1]; r_y = x[2]; r_z = x[3]
    v_x = x[4]; v_y = x[5]; v_z = x[6]
    m   = x[7]

    # Grab costates (don't use λm)
    lambda_r_x = x[8];  lambda_r_y = x[9];  lambda_r_z = x[10]
    lambda_v_x = x[11]; lambda_v_y = x[12]; lambda_v_z = x[13]

    # Grab parameters
    mu = p[1]
    T_max = p[2]
    c = p[3]

    @fastmath begin
        t2 = mu+r_x;
        t3 = lambda_v_x*lambda_v_x;
        t4 = lambda_v_y*lambda_v_y;
        t5 = lambda_v_z*lambda_v_z;
        t8 = r_y*r_y;
        t9 = r_z*r_z;
        t10 = 1.0/m;
        t11 = mu-1.0;
        t12 = t2-1.0;
        t13 = t2*t2;
        t17 = t3+t4+t5;
        t14 = t12*t12;
        t18 = t8+t9+t13;
        t20 = 1.0/sqrt(t17);
        t19 = t8+t9+t14;

        #t21 = 1.0/pow(t18,3.0/2.0);
        #t22 = 1.0/pow(t18,5.0/2.0);

        tt1 = sqrt(t18)
        tt13 = tt1*tt1*tt1
        tt15 = tt13*tt1*tt1
        t21 = 1.0/tt13
        t22 = 1.0/tt15

        #t23 = 1.0/pow(t19,3.0/2.0);
        #t24 = 1.0/pow(t19,5.0/2.0);

        tt2 = sqrt(t19)
        tt23 = tt2*tt2*tt2
        tt25 = tt23*tt2*tt2
        t23 = 1.0/tt23
        t24 = 1.0/tt25

        t26 = t11*t21;
        t29 = r_y*r_z*t11*t22*3.0;
        t25 = mu*t23;
        t28 = mu*r_y*r_z*t24*3.0;
        t27 = -t25;

        return SA[
            v_x,
            v_y,
            v_z,
            r_x+v_y*2.0+t2*t26+t12*t27-T_max*lambda_v_x*t10*t20,
            r_y-v_x*2.0+r_y*t26+r_y*t27-T_max*lambda_v_y*t10*t20,
            r_z*t26+r_z*t27-T_max*lambda_v_z*t10*t20,
            -T_max/c,
            -lambda_v_y*(mu*r_y*t12*t24*3.0-r_y*t2*t11*t22*3.0)-lambda_v_z*(mu*r_z*t12*t24*3.0-r_z*t2*t11*t22*3.0)-lambda_v_x*(t26+t27+mu*t14*t24*3.0-t11*t13*t22*3.0+1.0),
            -lambda_v_z*(t28-t29)-lambda_v_x*(mu*r_y*t12*t24*3.0-r_y*t2*t11*t22*3.0)-lambda_v_y*(t26+t27+mu*t8*t24*3.0-t8*t11*t22*3.0+1.0),
            -lambda_v_y*(t28-t29)-lambda_v_x*(mu*r_z*t12*t24*3.0-r_z*t2*t11*t22*3.0)+lambda_v_z*(t25-t26-mu*t9*t24*3.0+t9*t11*t22*3.0),
            -lambda_r_x+lambda_v_y*2.0,
            -lambda_r_y-lambda_v_x*2.0,
            -lambda_r_z,
            -(T_max*(t10*t10))/t20,
        ]
    end
end

function shooting_fun!(
    F, y, κ, x0, xf, ode_ps;
    N      = 10,
    solver = Vern9(),
    reltol = 1e-14,
    abstol = 1e-14,
)
    λ0 = SA[y[1], y[2], y[3], y[4], y[5], y[6], y[7]]
    x_tf = Xhat_tf(
        y[end], λ0, κ, x0,
        @SVector(fill(2.0 / 3.0, 7)),
        crtbp_min_time_eom, ode_ps;
        quad   = GaussHermite(),
        N      = N,
        solver = solver,
        reltol = reltol,
        abstol = abstol,
    )

    for i = 1:6
        F[i] = x_tf[i] - xf[i]
    end
    F[7] = x_tf[14]
    F[8] = dot(
        λ0,
        crtbp_min_time_eom(vcat(x0, λ0), ode_ps, 0.0)[SA[1,2,3,4,5,6,7]]) + 1.0
    return nothing
end

# Spacecraft Parameters
T_max_si    = 0.3e-3           # kg*km/s^2
I_sp_si     = 3000.0            # s
g_0_si      = 9.81e-3           # km/s^2
c_si        = I_sp_si * g_0_si  # km/s
m0_si       = 1500.0            # kg

# CRTBP Parameters
G           = 6.673e-20
m_1         = 5.9742e24         # kg
m_2         = 7.3483e22         # kg
r_12        = 384400.0          # km
mu          = m_2 / (m_1 + m_2) # n.d.

# Define units
DU = r_12
TU = 1.0 / sqrt((G*(m_1 + m_2)) / DU^3)
MU = m0_si

# Initial state (States in units defined below, i.e., see LU, TU, MU)
r0          = SA[-0.0194885115, -0.0160334798, 0.0]
v0          = SA[8.9188819237, -4.0817936888, 0.0]
m0          = 1.0
x0          = SA[r0[1], r0[2], r0[3], v0[1], v0[2], v0[3], m0]

# Final state
rxf = 0.8233851820
rzf = -0.02213408367497622
vyf = 0.1340880359734055
xf  = SA[rxf, 0.0, rzf, 0.0, vyf, 0.0]

# Scale paramters
T_max_nd  = T_max_si * TU^2 / (MU * DU)
I_sp_nd   = I_sp_si / TU
c_nd      = c_si * TU / DU

λ0  = @SVector(rand(7))
tf  = 200.0*rand()*86400.0/TU
tf  = 140.0*86400.0/TU

# F = zeros(8)
# shooting_fun!(
#     F, SA[λ0..., tf], 0.0, x0, xf, (mu, T_max_nd, c_nd);
#     N = 3,
# )

fun = let x0 = x0, xf = xf, ode_ps = (mu, T_max_nd, c_nd)
    (F, y, p) -> shooting_fun!(F, y, 0.0, x0, xf, ode_ps; N = 3)
end

prob = NonlinearProblem{true}(fun, [λ0..., tf])
sol = solve(prob, TrustRegion(); show_trace = Val(true), trace_level = TraceAll())
