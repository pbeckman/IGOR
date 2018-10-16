include("./gaussian_process.jl")
include("./maximum_likelihood.jl")
import gaussian_process: GP, matern, d_matern
import maximum_likelihood: MLE

xs = [1 1.1 3 3.5]'
ys = [4 3 200    1]'

gp = GP(xs, ys, matern, d_matern, nothing, nothing, nothing)

l = MLE(gp)

println("\nOPTIMAL l: ", l)
