using PyPlot

include("./kernels.jl")
include("./gaussian_process.jl")
include("./maximum_likelihood.jl")
import kernels: matern, d_matern, dd_matern, d_hp_matern,
                wendland2, d_wendland2, dd_wendland2,
                prod, d_prod, dd_prod, d_hp_prod
import gaussian_process: GP, posterior, update_covariance
import maximum_likelihood: MLE

file = open("output.txt", "w")

### DEFINE TEST FUNCTION

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

xs = reshape(linspace(-5,5,n_samples), (n_samples,1)) # [-4 -0.2 0.5 4]' # [1 2 3.9 4.1 6 9]' # [-0.9 1.9 3.9 4.2 6.2 8.2]'
ys = [f(x) for x in xs]
ds = [d_f(x) for x in xs]
ts = [min([abs(x-d) for d in disc]...) for x in xs]

### BUILD GAUSSIAN PROCESSES

# GP for trigger distance
gp_t = GP(
    xs,
    ts,
    nothing,
    nothing,
    matern, d_matern, dd_matern, d_hp_matern,
    [2],
    [2.5],
    nothing,
    nothing
    )

# update trigger distance GP covariance
update_covariance(gp_t, use_ds=false)

# average distance to trigger
function r(x,y)
    mean, cov = posterior(gp_t, reshape([x, y], 2, 1), use_ds=false)
    t1, t2 = mean
    return (t1 + t2)/2
end

# GP for data
# product of matern and wendland2
gp = GP(
    xs,
    ys,
    ds,
    ts,
    prod(matern, wendland2, 1, 1),
    d_prod(matern, d_matern, wendland2, d_wendland2, 1, 1),
    dd_prod(matern, d_matern, dd_matern, wendland2, d_wendland2, dd_wendland2, 1, 1),
    nothing, # d_hp_prod(d_hp_matern, d_hp_matern, 1, 1),
    [3, 8], # matern length scale, wendland2 radius factor
    [2.5, r], # matern differentiability, wendland2 radius surrogate
    nothing,
    nothing
    )

### ESTIMATE HYPERPARAMETERS

# MLE(gp, [0 1000], [5], file, use_ds=use_ds)
# MLE(gp, [0 1000; 0 1000], [1, 1.1], file, use_ds=use_ds)
println(gp.hyperparameters)

### COMPUTE POSTERIOR

# update data GP covariance with new hyperparameters
update_covariance(gp, use_ds=use_ds)

n_grid = 500
g = reshape(linspace(-5,5,n_grid), (n_grid,1))

# compute posterior mean and covariance of trigger distance GP
mean_t, cov_t = posterior(gp_t, g, use_ds=false)

# compute posterior mean and covariance of data GP
mean, cov = posterior(gp, g, use_ds=use_ds)
std_dev = [sqrt(max(0, cov[i,i])) for i=1:size(cov,1)]

### PLOT

# plot trigger distance
plot(g, mean_t, color="blue", label="trigger distance")
# plot trigger data
scatter(xs, ts, color="blue")

# plot mean
plot(g, mean, color="red", label="SE")
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
# title("alpha = 10")
# legend(loc="upper left")

show()
