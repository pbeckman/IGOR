using PyPlot
using ForwardDiff

include("./kernels.jl")
include("./gaussian_process.jl")
include("./maximum_likelihood.jl")
import kernels: matern, NS_SE
import gaussian_process: GP, posterior, update_covariance
import maximum_likelihood: MLE

file = open("output.txt", "w")

### DEFINE TEST FUNCTION

# piecewise linear continuous function
function f(x)
    if x <= -2
        return -2*x - 10
    elseif x <= -1
        return 5*x + 4
    elseif x <= 3
        return -1*x - 2
    else
        return -5
    end
end

function d_f(x)
    if x <= -2
        return -2
    elseif x <= -1
        return 5
    elseif x <= 3
        return -1
    else
        return 0
    end
end

# discontinuities
disc = [-2, -1, 3]

### COLLECT DATA
# xs, function values, derivative values, and trigger distances

# number of regular grid samples to take of f
n_samples = 4

xs = reshape(linspace(-5,5.5,n_samples), (n_samples,1))
ys = [f(x) for x in xs]
ds = [d_f(x) for x in xs]
ts = [min([abs(x-d) for d in disc]...) for x in xs]

### BUILD GAUSSIAN PROCESSES

# whether or not to use derivative data
use_ds = true

# GP for log trigger distance
gp_t = GP(
    xs=xs,
    ys=log.(ts),
    kernel=matern,
    hyperparameters=[1, 2],
    args=[2.5]
    )

# update trigger distance GP covariance
update_covariance(gp_t, use_ds=false)

## hyperparameters
# magnitude of SE
α_SE = 1
# length scale of SE
l_SE = 1
# minimum length scale of SE
min_l_SE = 0.1
# magnitude of matern
α_m = 1
# length scale of matern
l_m = 3
# use GP posterior mean as a surrogate for length scale
t(x) = exp.(posterior(gp_t, reshape(x, 1, 1) ; use_ds=false)[1][1]) + 0.5
# differentiability of matern
v = 2.5

# construct product kernel of matern 5/2 and variable length SE
k(x, y, hps, args) = NS_SE(x, y, hps[1:3], [args[1]]) * matern(x, y, hps[4:end], [args[2]])

# GP using product kernel
gp = GP(
    xs=xs,
    ys=ys,
    ds=ds,
    kernel=k,
    hyperparameters=[α_SE, l_SE, min_l_SE, α_m, l_m],
    args=[t, v]
    )

### ESTIMATE HYPERPARAMETERS

# use a single MLE Ipopt run to estimate hyperparameters
# generally you should use multiple starts to avoid local minima
MLE(gp, [0.1 10.0; 0.0 20.0; 0.0 2.0; 0.1 10.0; 0.0 20.0], [1.0, 15.0, 0.2, 1.0, 5.0], file, use_ds=use_ds)

println("hyperparameters: ", gp.hyperparameters)

### COMPUTE POSTERIOR

# update data GP covariance with new hyperparameters
update_covariance(gp, use_ds=use_ds)

display(gp.K)

# make grid on which to compute posterior
n_grid = 200
g = reshape(linspace(-5,5.5,n_grid), (n_grid,1))

# compute posterior mean and covariance of data GP
mean, cov = posterior(gp, g, use_ds=use_ds)
std_dev = [sqrt(max(0, cov[i,i])) for i=1:size(cov,1)]

### PLOT

# plot trigger distance
plot(g, [t([x]) for x in g], color="blue", label="Matern 5/2 GP for trigger distance")
# plot trigger data
scatter(xs, [t([x]) for x in xs], color="blue")

# plot mean
plot(g, mean, color="red", label="variable length SE * Matern 5/2 GP for data")
# plot true function
plot(g, [f(x) for x in g], color="black")
# plot discontinuities
for d in disc
    plot([d,d], [-50,50], color="black", linestyle="--")
end
# plot variance
fill_between(vec(g), vec(mean - 2*std_dev), vec(mean + 2*std_dev), color="#dddddd")
# plot data
scatter(xs, ys, color="black")

xlim([-5, 5.5])
ylim([-10, 10])
legend(loc="upper right")

show()
