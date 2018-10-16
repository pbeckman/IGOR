include("./latin_hypercube.jl")
include("./expected_improvement.jl")
import latin_hypercube: LH
import expected_improvement: kernel, K, multistart_optimize

using PyPlot

dim = 1
n_init = 1
n_add = 10

n_starts = 10*dim

bounds = hcat(zeros(dim,1), ones(dim,1))

f(x) = norm(x - 0.2*ones(dim))^2

xs = LH(bounds, n_init)
ys = [f(xs[i,:]) for i = 1:n_init]

alpha = 5.0
beta = 0
cost(x) = 0

for i = 1:n_add
    next_x = multistart_optimize(xs, ys, bounds, n_starts ; obj="EI", alpha=alpha, beta=beta, cost=cost)

    println("NEXT X: ", next_x)

    xs = vcat(xs, next_x')
    ys = vcat(ys, rand(dim)) # f(next_x))

    if dim == 1
        m = size(xs,1)
        K_inv = inv(K(xs, xs))

        g = linspace(0,1,300)
        means = [sum([kernel(xs[i,:], x) for i = 1:m][i] * (K_inv * ys)[i] for i = 1:m) for x in g]
        std_devs = [sqrt(max(0,v)) for v in diag(K(g, g) - K(g, xs) * K_inv * K(xs, g))]

        plot(g, means, color="red")
        fill_between(g, means - 2*std_devs, means + 2*std_devs, color="#dddddd")
        scatter(xs, ys, color="black")

        show()
    elseif dim == 2
        m = size(xs,1)
        K_inv = inv(K(xs, xs))

        n_g = 20
        g = reshape([[i, j] for i=0:1.0/n_g:1, j=0:1.0/n_g:1], (n_g + 1)^2)
        means = [sum([kernel(xs[i,:], x) for i = 1:m][i] * (K_inv * ys)[i] for i = 1:m) for x in g]
        x = [p[1] for p in g]
        y = [p[2] for p in g]
        surf(x, y, means, color="red")
        scatter3D(xs[:,1], xs[:,2], ys, color="black")

        show()
    end
end

println("MIN FUNC VAL: ", min(ys...))
