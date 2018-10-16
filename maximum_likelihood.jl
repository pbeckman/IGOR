module maximum_likelihood

    using JuMP
    using Ipopt
    using PyPlot

    include("./gaussian_process.jl")
    import gaussian_process: update_covariance

    export MLE

    function MLE(gp, file)
        SO = STDOUT

        m, n = size(gp.xs)

        function logL(hyperparameters...)
            gp.hyperparameters = collect(hyperparameters)[1:1]

            update_covariance(gp)

            return -0.5 * sum(sum(gp.ys[i] * gp.K_inv[i,j] * gp.ys[j] for i = 1:m) for j = 1:m) -
                    0.5 * log(abs(det(gp.K))) -
                    0.5 * n * log(2*π)
        end

        function d_logL(g, hyperparameters...)
            gp.hyperparameters = collect(hyperparameters)[1:1]

            update_covariance(gp)

            alpha = gp.K_inv * gp.ys

            d_K = zeros(m, m)

            for i = 1:m
                for j = 1:m
                    d_K[i,j] = gp.d_kernel(gp.xs[i,:], gp.xs[j,:], gp.hyperparameters)
                end
            end

            g[1] = 0.5 * trace((alpha * alpha' - gp.K_inv) * d_K)
            g[2] = 0
        end

        # g = linspace(1e-08,10,300)
        # plot(g, [logL(x,0) for x in g])
        # show()

        # println("making model")
        model = Model(solver = IpoptSolver())

        @variable(model, l >= 1e-08)
        @variable(model, fake)

        # println("building objective")
        JuMP.register(model, :likelihood, 2, logL, d_logL)

        JuMP.setNLobjective(model, :Max, Expr(:call, :likelihood, l, fake))

        # print(model)

        redirect_stdout(file)
        status = solve(model)
        redirect_stdout(SO)

        gp.hyperparameters = [getvalue(l)]
    end

end
