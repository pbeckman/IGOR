module maximum_likelihood

    using JuMP
    using Ipopt
    using PyPlot
    using Iterators

    include("./gaussian_process.jl")
    import gaussian_process: update_covariance, gp_var, stack

    export MLE

    function MLE(gp, bounds, starts, file ; use_ds=false) # ; epsilon=0, bounds=nothing)
        SO = STDOUT

        # number of observations, dimension
        m, n = size(gp.xs)

        # number of hyperparameters
        n_hp = length(gp.hyperparameters)

        function logL(hyperparameters...)
            gp.hyperparameters = collect(hyperparameters)#[1:1]

            println("hyperparameters: ", gp.hyperparameters)

            update_covariance(gp, use_ds=use_ds)

            return (
                -0.5 * stack(gp.ys, gp.ds)' * (gp.K \ stack(gp.ys, gp.ds)) -
                 0.5 * log(abs(det(gp.K)))
                    )[1]

            # return -0.5 * sum(sum(gp.ys[i] * gp.K_inv[i,j] * gp.ys[j] for i = 1:m) for j = 1:m) -
            #         0.5 * log(abs(det(gp.K)))
        end

        function d_logL(g, hyperparameters...)
            gp.hyperparameters = collect(hyperparameters)#[1:1]

            update_covariance(gp, use_ds=use_ds)

            if use_ds
                alpha = gp.K \ stack(gp.ys, gp.ds)
                d_Ks = [zeros(2m, 2m) for k=1:n_hp]
            else
                alpha = gp.K \ gp.ys
                d_Ks = [zeros(m, m) for k=1:n_hp]
            end



            for i = 1:m
                for j = 1:m
                    # derivative of kernel in each hyperparameter
                    d_hps = gp.d_hp_kernel(gp.xs[i,:], gp.xs[j,:], gp.hyperparameters, gp.args)
                    for k = 1:n_hp
                        d_Ks[k][i,j] = d_hps[k]
                    end
                end
            end

            for k = 1:n_hp
                g[k] = 0.5 * trace((alpha * alpha' - gp.K_inv) * d_Ks[k])
            end
        end

        # println("making model")
        model = Model(solver = IpoptSolver(max_iter=1000))

        vars = Array{Any}(n_hp)
        for k = 1:n_hp
            vars[k] = @variable(model, lowerbound = bounds[k,1], upperbound = bounds[k,2], start = starts[k])
        end

        # @variable(model, l >= 1e-16, start=100.0)
        # @variable(model, fake)

        # if epsilon > 0
        #     function var_x(x,l)
        #         gp.hyperparameters = [l]
        #         println("x : ", x)
        #         return gp_var(gp, x)
        #     end
        #
        #     # vertices = []
        #     global constraint_funcs = []
        #     for x in product([bounds[i,:] for i = 1:n]...)
        #         push!(constraint_funcs, partial(var_x, collect(x)))
        #     end
        #
        #     for i = 1:length(constraint_funcs)
        #         s = Symbol("cons"*string(i))
        #         JuMP.register(model, s, 1, constraint_funcs[i], autodiff=true)
        #         @eval @NLconstraint($model, $(Expr(:call, s, l)) <= 1-$epsilon)
        #     end
        # end

        # println("building objective")
        JuMP.register(model, :likelihood, n_hp, logL, d_logL)

        JuMP.setNLobjective(model, :Max, Expr(:call, :likelihood, vars...))

        # print(model)

        redirect_stdout(file)
        status = solve(model)
        redirect_stdout(SO)

        if status != :Optimal && status != :UserLimit
            println("MLE FAILED")
            throw(ErrorException("MLE FAILED"))
        end

        gp.hyperparameters = [getvalue(vars[k]) for k = 1:n_hp]
    end

    function partial(f,a...)
        ((b...) -> f(a...,b...))
    end

end
