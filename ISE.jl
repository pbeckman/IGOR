### ARGUMENTS:
### dimension, quadrature reltol, output.csv

using Cubature
using DataFrames
using CSV
using PyPlot
using IterTools

include("./trigger_approx.jl")
import trigger_approx: trig_dists, nn_trig_dists
include("./kernels.jl")
import kernels: SE, d_SE, dd_SE, matern, d_matern, dd_matern, NS_matern
include("./gaussian_process.jl")
import gaussian_process: GP, posterior, update_covariance
include("./latin_hypercube.jl")
import latin_hypercube: LH
include("./maximum_likelihood.jl")
import maximum_likelihood: MLE
include("./plotting.jl")
import plotting: plot_posterior

# open file for JuMP to write to
file = open("output.txt", "w")

### DEFINE TEST FUNCTION

# dimension of the space
p = parse(Int, ARGS[1])

if p == 1
    ### use piecewise polynomial

    # number of triggers
    k = 2
    g1(x) = x[1]-1.5
    ∇g1(x) = [1]
    g2(x) = x[1]-2.6 # x^3-8 #
    ∇g2(x) = [1] # 3x^2 #

    trigs = [g1, g2]
    trig_grads = [∇g1, ∇g2]

    bounds = [-5 5]

    # 1D nonlinear function with linear triggers
    function f(x)
        if g1(x) < 0
            return 0.1*(x[1]-1.5)^2 # -3 # 0.2x[1]^2
        elseif g2(x) < 0
            return 2(x[1]-1.5)^3 # 6x[1] # x[1]^3
        else
            return 2*1.1^3 # 3 # -x[1] + 10
        end
    end

    function ∇f(x)
        if g1(x) < 0
            return [0.2(x[1]-1.5)] # [0.4x[1]]
        elseif g2(x) < 0
            return [6(x[1]-1.5)^2] # [3x[1]^2]
        else
            return [0] # [-1]
        end
    end
else
    ### use radial function applicable in higher dimensions

    # number of triggers
    k = 2
    g1(x) = sum([x[i]^2 for i=1:p]) - 1.0
    ∇g1(x) = 2x
    g2(x) = sum([x[i]^2 for i=1:p]) - 2.0
    ∇g2(x) = 2x

    trigs = [g1, g2]
    trig_grads = [∇g1, ∇g2]

    bounds = vcat([[-2.5 2.5] for i=1:p]...)

    # 2D nonlinear function with nonlinear triggers
    function f(x)
        if g1(x) < 0 # inside
            return 3sum([x[i]^2 for i=1:p]) - 3
        elseif g2(x) < 0 # ring
            return -2sum([x[i]^2 for i=1:p]) + 2
        else # outside
            return sum([x[i]^2 for i=1:p]) - 4
        end
    end

    function ∇f(x)
        if g1(x) < 0
            return 6x
        elseif g2(x) < 0
            return -4x
        else
            return 2x
        end
    end
end

### ERROR ESTIMATES

function ISE(f, gp, bounds)
    if p == 1
        return hquadrature(
            x -> (f([x]) - posterior(gp, reshape([x], (1, 1)))[1][1])^2,
            bounds[1],
            bounds[2],
            reltol=parse(Float64, ARGS[2])
            )
    else
        return hcubature(
            x -> (f(x) - posterior(gp, reshape(x, (1, p)))[1][1])^2,
            bounds[:, 1],
            bounds[:, 2],
            reltol=parse(Float64, ARGS[2])
            )
    end
end

function test(n)
    ### COLLECT DATA
    if p == 1
        xs = reshape(linspace(bounds[1], bounds[2], n), (n, p))
    else
        # number of observations per dimension
        s = round(Int, n^(1/p))
        # generate iterator for p-tuples
        xs = product([linspace(bounds[i, 1], bounds[i, 2], s) for i=1:p]...)
        # collect and stack into matrix
        xs = vcat([collect(x)' for x in xs]...)
    end

    fs = zeros(n, 1)
    ∇fs = zeros(n, p)
    gs = zeros(n, k)
    ∇gs = zeros(n, p, k)
    ts = zeros(n, p)
    ∇ts = zeros(n, p)
    for i=1:n
        fs[i,1] = f(xs[i,:])
        ∇fs[i,:] = ∇f(xs[i,:])
        for j=1:k
            gs[i,j] = trigs[j](xs[i,:])
            ∇gs[i,:,j] = trig_grads[j](xs[i,:])
        end
    end
    for i=1:n
        t, ∇t = trig_dists(xs[i,:], gs[i,:], ∇gs[i,:,:])
        ts[i,:] = t
        ∇ts[i,:] = ∇t
    end

    ## BUILD GAUSSIAN PROCESSES

    ## hyperparameters
    # magnitude of matern
    α_m = 1
    # length scale of matern
    l_m = if (p==1) 3 else 2 end
    # differentiability of matern
    v = 2.5
    # maximum noise
    max_noise = 1e10
    # noise width
    β = 0.007

    # hyperparameter bounds
    hp_bounds = [0.1 10.0; 0.1 10.0]
    # noise parameter bounds
    noise_bounds = [1.0 1e8; 1e-8 1]

    # GP no derivatives
    gp_noder = GP(
        xs=xs,
        ys=fs,
        ds=∇fs,
        ts=ts,
        kernel=matern,
        hyperparameters=[α_m, l_m],
        args=[v],
        use_ds=false
        )

    # GP derivatives
    gp_der = GP(
        xs=xs,
        ys=fs,
        ds=∇fs,
        ts=ts,
        kernel=matern,
        d_kernel=d_matern,
        dd_kernel=dd_matern,
        hyperparameters=[α_m, l_m],
        args=[v],
        use_ds=true,
        use_AD=false
        )

    # GP noisy derivatives
    gp_noisy = GP(
        xs=xs,
        ys=fs,
        ds=∇fs,
        ts=ts,
        kernel=matern,
        d_kernel=d_matern,
        dd_kernel=dd_matern,
        hyperparameters=[α_m, l_m],
        args=[v],
        noise_params=[max_noise, β],
        use_ds=true,
        use_AD=false,
        use_noise=true
        )

    MLE(gp_noder, hp_bounds, gp_der.hyperparameters, file, target="hyperparameters")
    MLE(gp_der, hp_bounds, gp_der.hyperparameters, file, target="hyperparameters")
    MLE(gp_noisy, vcat(hp_bounds, noise_bounds), vcat(gp_noisy.hyperparameters, gp_noisy.noise_params), file, target="all")
    println("noise parameters: ", gp_noisy.noise_params)

    ### COMPUTE POSTERIOR
    update_covariance(gp_noder)
    update_covariance(gp_der)
    update_covariance(gp_noisy)

    plot_posterior(gp_noder, bounds, if (p==1) 300 else 20 end, f=f)
    show()
    plot_posterior(gp_der, bounds, if (p==1) 300 else 20 end, f=f)
    show()
    plot_posterior(gp_noisy, bounds, if (p==1) 300 else 20 end, f=f)
    show()

    ise_noder, err_noder = ISE(f, gp_noder, bounds)
    ise_der, err_der = ISE(f, gp_der, bounds)
    ise_noisy, err_noisy = ISE(f, gp_noisy, bounds)

    println("n: ", n, ", noder: ", ise_noder, ", der: ", ise_der, ", noisy: ", ise_noisy)
    println("  err - noder: ", err_noder, ", der: ", err_der, ", noisy: ", err_noisy)

    return [ise_noder, ise_der, ise_noisy]'
end

### RUN TESTS
ns = [n^p for n=2:10]

data = zeros(3, length(ns))

for i=1:length(ns)
    data[:, i] = test(ns[i])
end

println(data)

if length(ARGS) > 2
    CSV.write(ARGS[3], DataFrame(data))
end
