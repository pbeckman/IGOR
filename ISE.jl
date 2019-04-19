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

function plot_posterior(f, gp, bounds)
    if p == 1
        # make grid on which to compute posterior
        n_grid = 300
        g = reshape(linspace(bounds[1], bounds[2], n_grid), (n_grid, 1))

        # compute posterior mean and covariance of GP
        mean, cov = posterior(gp, g)
        std_dev = [sqrt(max(0, cov[i, i])) for i=1:size(cov,1)]

        # plot true function
        plot(g, [f(x) for x in g], color="black")

        # plot GP
        plot(g, mean, color="red")
        fill_between(vec(g), vec(mean - 2*std_dev), vec(mean + 2*std_dev), color="#dddddd")

        # plot data
        scatter(gp.xs, gp.ys, color="black")
    elseif p == 2
        # make grid on which to compute posterior
        n_grid = 20
        g = vcat([[i j] for i=bounds[1, 1]:(bounds[1, 2] - bounds[1, 1])/n_grid:bounds[1, 2], j=bounds[2, 1]:(bounds[2, 2] - bounds[2, 1])/n_grid:bounds[2, 2]]...)

        # compute posterior mean and covariance of GP
        mean, cov = posterior(gp, g)

        # plot true function
        surf(g[:,1], g[:,2], [f(g[i,:]) for i=1:size(g,1)], color="blue")

        # plot GP
        surf(g[:,1], g[:,2], mean, color="red")

        # plot data
        scatter3D(gp.xs[:,1], gp.xs[:,2], gp.ys, color="black")
    end

    show()
end

function ISE(f, gp, bounds)
    if p == 1
        return hquadrature(
            x -> (f([x]) - posterior(gp, reshape([x], (1, 1)))[1][1])^2,
            bounds[1],
            bounds[2]
            )
    else
        return hcubature(
            x -> (f(x) - posterior(gp, reshape(x, (1, p)))[1][1])^2,
            bounds[:, 1],
            bounds[:, 2],
            reltol=parse(Float64, ARGS[3])
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
        # println("x: ", xs[i,:], " t: ", t, " ∇t: ", ∇t)
        ts[i,:] = t
        ∇ts[i,:] = ∇t
    end

    # println("xs: ",  xs)
    # println("fs: ",  fs)
    # println("∇fs: ", ∇fs)
    # println("gs: ",  gs)
    # println("∇gs: ")
    # display(∇gs)
    # println("\nts: ", ts)

    ## BUILD GAUSSIAN PROCESSES

    ## hyperparameters
    # magnitude of matern
    α_m = 1
    # length scale of matern
    l_m = if (p==1) 3 else 2 end
    # differentiability of matern
    v = 2.5

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
        use_AD=(p==1)
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
        use_ds=true,
        use_AD=(p==1),
        noise_width=parse(Float64, ARGS[2])
        )

    ### COMPUTE POSTERIOR
    update_covariance(gp_noder)
    update_covariance(gp_der)
    update_covariance(gp_noisy)

    # plot_posterior(f, gp_noder, bounds)
    # plot_posterior(f, gp_der, bounds)
    # plot_posterior(f, gp_noisy, bounds)

    ise_noder, err_noder = ISE(f, gp_noder, bounds)
    ise_der, err_der = ISE(f, gp_der, bounds)
    ise_noisy, err_noisy = ISE(f, gp_noisy, bounds)

    println("n: ", n, ", noder: ", ise_noder, ", der: ", ise_der, ", noisy: ", ise_noisy)
    println("  err - noder: ", err_noder, ", der: ", err_der, ", noisy: ", err_noisy)

    return [ise_noder, ise_der, ise_noisy]'
end

### RUN TESTS
if p == 1
    ns = 2:15
else
    ns = [n^p for n=2:10]
end

data = zeros(3, length(ns))

for i=1:length(ns)
    data[:, i] = test(ns[i])
end

println(data)

CSV.write(ARGS[4], DataFrame(data))
