using PyPlot

include("./trigger_approx.jl")
import trigger_approx: trig_dists, nn_trig_dists
include("./kernels.jl")
import kernels: SE, matern, gibbs_matern, stein_matern
include("./gaussian_process.jl")
import gaussian_process: GP, posterior, update_covariance
include("./latin_hypercube.jl")
import latin_hypercube: LH
include("./expected_improvement.jl")
include("./maximum_likelihood.jl")
include("./experiment_design.jl")
import experiment_design: var_DOE

method = ARGS[1]

### DEFINE TEST FUNCTION

# 1D nonlinear function with linear triggers
g1(x) = x[1]-1.5
∇g1(x) = [1]
g2(x) = x[1]-2.6 # x^3-8 #
∇g2(x) = [1] # 3x^2 #

trigs = [g1, g2]
trig_grads = [∇g1, ∇g2]

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

### COLLECT DATA
# xs, function values and gradients, trigger values and gradients

# number of samples to take of f
n = 5

bounds = [-5 5]

# dimension of space
p = size(bounds, 1)
# number of triggers
k = 2

# xs = LH(bounds, n)
xs = reshape(linspace(bounds[1], bounds[2], n), (n, p))
# xs = reshape([-5, 0.2, 5], (n, p))
fs = zeros(n, 1)
∇fs = zeros(n, p)
gs = zeros(n, k)
∇gs = zeros(n, p, k)
ts = zeros(n, 1)
∇ts = zeros(n, p)
for i=1:n
    fs[i,1] = f(xs[i,:])
    ∇fs[i,:] = ∇f(xs[i,:])
    for j=1:k
        gs[i,j] = trigs[j](xs[i][1])
        ∇gs[i,:,j] = trig_grads[j](xs[i][1])
    end
end
for i=1:n
    t, ∇t = trig_dists([xs[i]], gs[i,:], ∇gs[i,:,:])
    println("x: ", xs[i], " t: ", t, " ∇t: ", ∇t)
    ts[i,1] = t[1]
    ∇ts[i,:] = ∇t
end

println("LH xs: ",  xs)
println("LH fs: ",  fs)
println("LH ∇fs: ", ∇fs)
println("LH gs: ",  gs)
println("LH ∇gs: ")
display(∇gs)
println("\nLH ts: ", ts)

## BUILD GAUSSIAN PROCESSES

## hyperparameters
# magnitude of matern
α_m = 1
# length scale of matern
l_m = 3

if method == "noder"
    # GP using matern kernel
    gp = GP(
        xs=xs,
        ys=fs,
        ds=∇fs,
        ts=ts,
        kernel=matern,
        hyperparameters=[α_m, l_m],
        args=[2.5],
        use_ds=false
        )
elseif method == "der"
    # GP using matern kernel
    gp = GP(
        xs=xs,
        ys=fs,
        ds=∇fs,
        ts=ts,
        kernel=matern,
        hyperparameters=[α_m, l_m],
        args=[2.5],
        use_ds=true
        )
elseif method == "noisy"
    # GP using matern kernel
    gp = GP(
        xs=xs,
        ys=fs,
        ds=∇fs,
        ts=ts,
        kernel=matern,
        hyperparameters=[α_m, l_m],
        args=[2.5],
        use_ds=true,
        noise_params=[1e10, 0.004],
        use_noise=true
        )
elseif method in ["length", "smoothness"]
    # log GP for log trigger distance
    gp_t = GP(
        xs=xs,
        ys=log.(ts),
        ds=vcat([
            ∇ts[i,:] / ts[i,1] for i=1:n
        ]...),
        kernel=matern,
        hyperparameters=[1, 3],
        args=[2.5],
        use_ds=false
        )

    # update trigger distance GP covariance
    update_covariance(gp_t)

    # use GP posterior mean as a surrogate for length scale
    t(x) = exp.(posterior(gp_t, reshape(x, 1, 1))[1][1])

    if method == "length"
        # GP using nonstationary lengthscale matern kernel
        gp = GP(
            xs=xs,
            ys=fs,
            ds=∇fs,
            kernel=gibbs_matern,
            hyperparameters=[2, 0.7, 3, 2.0],
            args=[2.5, t],
            use_ds=true
            )
    elseif method == "smoothness"
        # GP using matern kernel
        gp = GP(
            xs=xs,
            ys=fs,
            ds=∇fs,
            ts=ts,
            kernel=stein_matern,
            hyperparameters=[α_m, l_m, 4.0],
            args=[t],
            use_ds=false
            )
    end
elseif method == "mixture"
    # proximity to derivative in support of rough GP
    ϵ = 1.0
    δ = 3.0
    smooth_inds = [i for i=1:n if min(ts[i,:]...) > ϵ]
    rough_inds = [i for i=1:n if min(ts[i,:]...) <= ϵ]
    # mixing parameter
    function m(x)
        r = min([norm(x - xs[i,:]) for i in rough_inds]...)
        l = trunc(Int, p/2) + 2
        return max(1 - r/δ, 0)^(l+1) * ((l+1) * r/δ + 1)
    end

    # smooth GP for mixture
    gp_smooth = GP(
        xs=xs[smooth_inds, :],
        ys=fs[smooth_inds],
        ds=∇fs[smooth_inds, :],
        kernel=matern,
        hyperparameters=[α_m, l_m],
        args=[2.5],
        use_ds=true
        )

    # rough GP for mixture
    gp_rough = GP(
        xs=xs,
        ys=fs,
        kernel=matern,
        hyperparameters=[α_m, l_m],
        args=[0.5],
        use_ds=false
        )
end

### COMPUTE POSTERIOR

# make grid on which to compute posterior
n_grid = 300
g = reshape(linspace(bounds[1], bounds[2], n_grid), (n_grid, 1))

# plot true function
plot(g, [f(x) for x in g], color="black")#, label="true function")

if method == "mixture"
    # update covariance matrix K
    update_covariance(gp_smooth)
    update_covariance(gp_rough)
    display(gp_smooth.K)
    display(gp_rough.K)

    # compute posterior mean and covariance of GPs
    mean_s, cov_s = posterior(gp_smooth, g)
    mean_r, cov_r = posterior(gp_rough, g)
    mean = [m(g[i]) * mean_r[i] + (1 - m(g[i])) * mean_s[i] for i=1:n_grid]
    std_dev = [m(g[i]) * sqrt(max(0, cov_r[i, i])) + (1 - m(g[i])) * sqrt(max(0, cov_s[i, i])) for i=1:n_grid]

    # plot means and standard deviation
    fill_between(vec(g), vec(mean - 2*std_dev), vec(mean + 2*std_dev), color="#dddddd")
    plot(g, mean_s, color="purple", alpha=0.3, label="smooth mean")
    plot(g, mean_r, color="orange", alpha=0.6, label="nonsmooth mean")
    plot(g, mean, color="red", label="mixture mean")
else
    # update covariance matrix K
    update_covariance(gp)
    display(gp.K)

    # compute posterior mean and covariance of data GP
    mean, cov = posterior(gp, g)
    std_dev = [sqrt(max(0, cov[i, i])) for i=1:n_grid]

    # plot mean and standard deviation
    plot(g, mean, color="red", label="mean")
    fill_between(vec(g), vec(mean - 2*std_dev), vec(mean + 2*std_dev), color="#dddddd")

    if method in ["length", "smoothness"]
        # plot trigger distance
        plot(g, [t([x]) for x in g], color="blue", alpha=0.5, label="length scale")
    end
end

# plot data
scatter(xs, fs, color="black")

legend(loc="upper right", fontsize="x-large")

if length(ARGS) > 1
    savefig(ARGS[2], dpi=500)
end
show()
