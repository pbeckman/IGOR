using PyPlot
using ForwardDiff

include("./kernels.jl")
include("./gaussian_process.jl")
include("./maximum_likelihood.jl")
import kernels: matern, d_matern, dd_matern, d_hp_matern,
                wendland2, d_wendland2, dd_wendland2,
                prod, d_prod, dd_prod, d_hp_prod,
                VL_SE
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
    else
        return -1*x - 2
    end
end

function d_f(x)
    if x <= -2
        return -2
    elseif x <= -1
        return 5
    else
        return -1
    end
end

# discontinuities
disc = [-2, -1]

### COLLECT DATA
# xs, function values, derivative values, and trigger distances

n_samples = 4
use_ds = true

xs = reshape(linspace(-5,5,n_samples), (n_samples,1))
ys = [f(x) for x in xs]
ds = [d_f(x) for x in xs]
ts = [min([abs(x-d) for d in disc]...) for x in xs]

### BUILD GAUSSIAN PROCESSES

# GP for log trigger distance
gp_t = GP(
    xs=xs,
    ys=log.(ts),
    kernel=matern,
    hyperparameters=[2],
    args=[2.5]
    )

# update trigger distance GP covariance
update_covariance(gp_t, use_ds=false)

## hyperparameters
# length scale of SE
l_SE = 2
# length scale of matern
l_m = 5
# use GP posterior mean as a surrogate for length scale
t(x) = exp(posterior(gp_t, reshape(x, 1, 1) ; use_ds=false)[1][1])
# differentiability of matern
v = 2.5

# construct product kernel
k(x, y, hp, args) = VL_SE(x, y, [hp[1]], [args[1]]) * matern(x, y, [hp[2]], [args[2]])

# variable length SE
gp = GP(
    xs=xs,
    ys=ys,
    ds=ds,
    ts=ts,
    kernel=k,
    hyperparameters=[l_SE, l_m],
    args=[t, v]
    )

### ESTIMATE HYPERPARAMETERS

# MLE(gp, [0 1000], [5], file, use_ds=use_ds)
# MLE(gp, [0 1000; 0 1000], [1, 1.1], file, use_ds=use_ds)
println("hyperparameters: ", gp.hyperparameters)

### COMPUTE POSTERIOR

# update data GP covariance with new hyperparameters
update_covariance(gp, use_ds=use_ds)

display(gp.K)

# make grid on which to compute posterior
n_grid = 200
g = reshape(linspace(-5,5,n_grid), (n_grid,1))

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

xlim([-5,5])
ylim([-10,10])
legend(loc="upper right")

show()
