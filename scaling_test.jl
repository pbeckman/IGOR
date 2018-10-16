include("./gaussian_process.jl")
include("./latin_hypercube.jl")
include("./expected_improvement.jl")
include("./maximum_likelihood.jl")
include("./experiment_design.jl")
import experiment_design: EI_DOE, LH_DOE

file = open("output.txt", "w")

for dim in [1, 2, 5, 10, 100]
    for n_tot in [5, 10, 50, 100, 500, 1000]
        if n_tot <= 10 * dim
            println("dim: ", dim, ", samples: ", n_tot)
            n_starts = 10 * dim
            bounds = hcat(zeros(dim,1), ones(dim,1))
            f(x) = norm(x)^2
            n_init = max(2, div(n_tot, 4))
            n_add  = n_tot - n_init

            n_trials = 10

            EI_vals = zeros(n_trials)

            for i = 1:n_trials
                # println("running EI DOE")
                EI_x, EI_vals[i] = EI_DOE(bounds, n_init, n_add, n_starts, f, 0.1, 0, x->0, file)
                # println(EI_vals[i])
            end

            LH_vals = zeros(10000)
            for i = 1:10000
                # println("running LH DOE")
                LH_x, LH_vals[i] = LH_DOE(bounds, n_tot, f)
            end

            println("AVERAGE: \n\tEI = ", mean(EI_vals),  "\n\tLH = ", mean(LH_vals), "\n===================\n")
        end
    end
end

close(file)
