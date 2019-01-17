using PyPlot

include("./gaussian_process.jl")
import gaussian_process: GP, matern, d_matern, dd_matern, SE, d_SE, dd_SE, add_data, update_covariance, cov_mat, posterior, wendland2, hat


# function trigger(x1, x2, hyperparameters)
#     l, k, v = hyperparameters
#     r = norm(x1[1] - x2[1])
#     if x1[1] == x2[1]
#         return (1 + 5^0.5*r/l + 5*r^2/(3*l^2)) * exp(-5^0.5*r/l)
#     else
#         return (1 + 5^0.5*r/l + 5*r^2/(3*l^2)) * exp(-5^0.5*r/l) # * (x1[2] + x2[2]) # exp(-1/(k*(x1[2] + x2[2])))
#     end
# end

# function l(l, x1, x2)
#     if x1[1] != x2[1]
#         l *= exp(-1/(k*(x1[2] + x2[2])))
#     end
#     return l
# end

# function trigger(x1, x2, hyperparameters)
#     l, k, v = hyperparameters
#     r = norm(x1[1] - x2[1])
#     l = l_x(l, x1, x2)
#     # println("x: ")
#     return (1 + 5^0.5*r/l + 5*r^2/(3*l^2)) * exp(-5^0.5*r/l)
# end

function dd(x, xs, ts)
    # distance to discontinuity
    min([norm(x - xs[i]) + ts[i] for i=1:length(xs)]...)
end

function alpha(d, k)
    return exp(-1.0/(k * d))
end

function l(d, l_base, k)
    return l_base * exp(-1.0/(k * d))
end

function trigger(x1, x2, hyperparameters)
    xs, dists, l_base, k = hyperparameters

    l1 = l(dd(x1, xs, dists), l_base, k)
    l2 = l(dd(x2, xs, dists), l_base, k)

    return (l1 * l2)^0.5 * (0.5*l1^2 + 0.5*l2^2)^-0.5 * exp(-2 * norm(x1 - x2)^2 / (l1^2 + l2^2))
end

function d_trigger(x1, x2, d, hyperparameters)
    xs, dists, l_base, k = hyperparameters

    l1 = l(dd(x1, xs, dists), l_base, k)
    l2 = l(dd(x2, xs, dists), l_base, k)

    return - 2 * (l1 * l2)^0.5 * (0.5*l1^2 + 0.5*l2^2)^-1.5 * (x1[d] - x2[d]) * exp(- norm(x1 - x2)^2 / (0.5*l1^2 + 0.5*l2^2))
end

function dd_trigger(x1, x2, d1, d2, hyperparameters)
    xs, dists, l_base, k = hyperparameters

    l1 = l(dd(x1, xs, dists), l_base, k)
    l2 = l(dd(x2, xs, dists), l_base, k)

    return 2 * (l1 * l2)^0.5 * (0.5*l1^2 + 0.5*l2^2)^-1.5 * ((d1 == d2) - 2 * (0.5*l1^2 + 0.5*l2^2)^-0.5 * (x1[d1] - x2[d1])*(x1[d2] - x2[d2])) * exp(-2 * norm(x1 - x2)^2 / (l1^2 + l2^2))
end

# a = -1
# b = 1
#
# function f(x)
#     if x <= 0
#         return a*x - 3
#     else
#         return b*x - 3
#     end
# end
#
# function d_f(x)
#     if x <= 0
#         return a
#     else
#         return b
#     end
# end

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


n_samples = 5
xs = reshape(linspace(-5,5,n_samples), (n_samples,1)) # [-4 -0.2 0.5 4]' # [1 2 3.9 4.1 6 9]' # [-0.9 1.9 3.9 4.2 6.2 8.2]'
disc = [-2, -1]
# distance to trigger at observations
ts = [min([abs(x-d) for d in disc]...) for x in xs]
println(ts)
ys = [f(x) for x in xs]
ds = [d_f(x) for x in xs]

l_base = 2
k = 100

gp1 = GP(
    xs,
    ys,
    ds,
    ts,
    # SE, d_SE, dd_SE,
    # [l_base],
    # trigger, d_trigger, dd_trigger,
    # [xs, dists, l_base, k],
    matern, nothing, nothing, nothing,
    [l_base],
    [0.5],
    nothing,
    nothing
    )

# gp2 = GP(
#     xs,
#     ys,
#     ds,
#     ts,
#     SE, d_SE, dd_SE,
#     [l_base],
#     nothing,
#     nothing
#     )

gp2 = GP(
    xs,
    ys,
    ds,
    ts,
    matern, d_matern, dd_matern, nothing,
    [l_base],
    [2.5],
    nothing,
    nothing
    )

# gp2 = GP(
#     xs,
#     ys,
#     ds,
#     ts,
#     trigger, d_trigger, dd_trigger,
#     [xs, dists, l_base, k],
#     nothing,
#     nothing
#     )

use_ds = true

update_covariance(gp1, use_ds=false)
update_covariance(gp2, use_ds=use_ds)

# display(gp2.K)
# println("\n")
# display(gp2.K_inv)
# println("\n")

n_grid = 500
g = reshape(linspace(-5,5,n_grid), (n_grid,1))

mean1, cov1 = posterior(gp1, g, use_ds=false, taper_func=wendland2)
mean2, cov2 = posterior(gp2, g, use_ds=use_ds, taper_func=wendland2)

epsilons = [dd(x, xs, ts) for x in g]
# 0 for exp, 1 for SE/matern 2.5
alphas = [1 for e in epsilons] # [alpha(e, 0.1) for e in epsilons] #
ls = [l(e, l_base, k) for e in epsilons]

mean = [(1 - alphas[i]) * mean1[i] + alphas[i] * mean2[i] for i=1:length(alphas)]
std_dev = [sqrt(max(0, (1 - alphas[i]) * cov1[i,i] + alphas[i] * cov2[i,i])) for i=1:size(cov1,1)] # [sqrt(max(0, cov[i,i])) for i=1:size(cov,1)]

# plot(g, mean1, color="blue", alpha=0.5, label="exponential")
# plot(g, mean2, color="green", alpha=0.5, label="variable length SE")
plot(g, mean, color="red", label="SE")
# plot(g, alphas, color="blue")
# plot(g, ls, color="green")
plot(g, [f(x) for x in g], color="black")
for d in disc
    plot([d,d], [-50,50], color="black", linestyle="--")
end
fill_between(vec(g), vec(mean - 2*std_dev), vec(mean + 2*std_dev), color="#dddddd")
scatter(xs, ys, color="black")
xlim([-5,5])
ylim([-10,10])

# title("alpha = 10")
# legend(loc="upper left")

show()
