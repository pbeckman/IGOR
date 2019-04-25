module experiment_design

    using ForwardDiff
    using JuMP
    using Ipopt
    using PyPlot

    include("./latin_hypercube.jl")
    import latin_hypercube: LH
    include("./gaussian_process.jl")
    import gaussian_process: GP, add_data, update_covariance, posterior, gp_var
    include("./expected_improvement.jl")
    import expected_improvement: multistart_optimize
    include("./maximum_likelihood.jl")
    import maximum_likelihood: MLE
    include("./trigger_approx.jl")
    import trigger_approx: trig_dists

    export var_DOE, EI_DOE, LH_DOE

    function var_DOE(gp, bounds, n_add, n_starts, f, ∇f, trigs, trig_grads, file ; use_ds=true)
        p = size(bounds, 1)
        k = length(trigs)
        SO = STDOUT

        for i=1:n_add
            function var(x...)
                x = collect(x)
                v = gp_var(gp, x)
                return -v
            end

            model = Model(solver = IpoptSolver(max_iter=5000))

            variables = Array{Any}(p)
            for j = 1:p
                variables[j] = @variable(model, lowerbound = bounds[j, 1], upperbound = bounds[j, 2])
            end

            # println("building objective")
            JuMP.register(model, :var, p, var, autodiff=true)

            JuMP.setNLobjective(model, :Min, Expr(:call, :var, variables...))

            redirect_stdout(file)
            next_x = multistart_optimize(model, variables, bounds, n_starts)
            next_x = reshape(next_x, (1, p))
            redirect_stdout(SO)

            println("next x:     ", next_x)
            println("next f(x):  ", f(next_x))
            println("next ∇f(x): ", ∇f(next_x))

            gs = zeros(1, k)
            ∇gs = zeros(1, p, k)

            for j=1:k
                gs[1, j] = trigs[j](next_x)
                ∇gs[1, :, j] = trig_grads[j](next_x)
            end

            t, ∇t = trig_dists(next_x, gs[1, :], ∇gs[1, :, :])
            println("next t(x): ", t)

            add_data(
                gp,
                next_x,
                f(next_x),
                ds=∇f(next_x),
                ts=t
                )

            update_covariance(gp, c=240.0)

            display(gp.K)

            if i % 1 == 0
                # n_grid = 20
                # g = vcat([[i j] for i=-5.0:10.0/n_grid:5.0, j=-5.0:10.0/n_grid:5.0]...)
                # mean, cov = posterior(gp, g, use_ds=use_ds)
                # surf(g[:,1], g[:,2], mean, color="red")
                # surf(g[:,1], g[:,2], [f(g[i,:]) for i=1:size(g,1)], color="blue")
                # scatter3D(gp.xs[:,1], gp.xs[:,2], gp.ys, color="black")
                #
                # show()

                for j=1:p
                    n_grid = 100
                    g = zeros(n_grid, p)
                    g[:,j] = reshape(linspace(bounds[j,1], bounds[j,2], n_grid), (n_grid,1))

                    # println(g)
                    # println(gp.xs)
                    # println(gp.ys)
                    # println(gp.ds)

                    # compute posterior mean and covariance of data GP
                    mean, cov = posterior(gp, g)
                    std_dev = [sqrt(max(0, cov[i, i])) for i=1:size(cov,1)]

                    # plot mean
                    plot(g[:,j], mean, color="red")
                    # plot true function
                    plot(g[:,j], [f(g[i,j]) for i=1:n_grid], color="black")
                    # plot variance
                    fill_between(g[:,j], vec(mean - 2*std_dev), vec(mean + 2*std_dev), color="#dddddd")
                    # plot data
                    scatter(gp.xs[:,j], gp.ys, color="black")

                    show()
                end
            end
        end
    end


    # function EI_DOE(bounds, n_init, n_add, n_starts, f, alpha, beta, cost, file ; hyperparameters=nothing, epsilon=0)
    #     SO = STDOUT
    #
    #     xs = LH(bounds, n_init)
    #     ys = [f(xs[i,:]) for i = 1:n_init]
    #
    #     gp = GP(xs, ys, matern, d_matern, nothing, nothing, nothing)
    #
    #     if hyperparameters == nothing
    #         MLE(gp, file, epsilon=epsilon, bounds=bounds)
    #     else
    #         gp.hyperparameters = hyperparameters
    #     end
    #
    #     update_covariance(gp)
    #
    #     println("LENGTH SCALE l: ", gp.hyperparameters[1])
    #
    #     for i = 1:size(xs,1)
    #         println("LHS X     : ", xs[i,:])
    #         println("  LHS f(X): ", f(xs[i,:]))
    #     end
    #
    #     # sequentially add sample points and store their function values
    #     for i = 1:(n_add-1)
    #         # find max of EI
    #         redirect_stdout(file)
    #         try
    #             next_x = multistart_optimize(gp, bounds, n_starts ; obj="EI", alpha=alpha, beta=beta, cost=cost)
    #         catch e
    #             if isa(e, DomainError)
    #                 redirect_stdout(SO)
    #                 println("COVARIANCE MATRIX NEARLY SINGULAR")
    #                 break
    #             else
    #                 rethrow(e)
    #             end
    #         end
    #         redirect_stdout(SO)
    #
    #         println("NEXT X     : ", next_x)
    #         println("  NEXT f(X): ", f(next_x))
    #
    #         # if min([norm(next_x' - xs[i,:]) for i = 1:size(xs,1)]...) < 0.1
    #         #     break
    #         # end
    #
    #         add_data(gp, next_x', f(next_x))
    #         update_covariance(gp)
    #     end
    #
    #     # find min of mean
    #     redirect_stdout(file)
    #     next_x = multistart_optimize(gp, bounds, n_starts ; obj="mean")
    #     redirect_stdout(SO)
    #
    #     println("FINAL X   : ", next_x)
    #     println("FINAL f(X): ", f(next_x))
    #
    #     add_data(gp, next_x', f(next_x))
    #
    #     min_val, min_ind = findmin(gp.ys)
    #
    #     println("BEST f(X): ", min_val, "\n")
    #
    #     return gp.xs[min_ind,:], min_val
    # end


    function LH_DOE(bounds, n_tot, f)
        xs = LH(bounds, n_tot)
        ys = [f(xs[i,:]) for i = 1:n_tot]

        min_val, min_ind = findmin(ys)

        return xs[min_ind,:], min_val
    end

end
