module experiment_design

    include("./latin_hypercube.jl")
    include("./gaussian_process.jl")
    include("./expected_improvement.jl")
    include("./maximum_likelihood.jl")
    import latin_hypercube: LH
    import gaussian_process: GP, matern, d_matern, add_data, update_covariance
    import expected_improvement: multistart_optimize
    import maximum_likelihood: MLE

    export EI_DOE, LH_DOE

    function EI_DOE(bounds, n_init, n_add, n_starts, f, alpha, beta, cost, file ; hyperparameters=nothing, epsilon=0)
        SO = STDOUT

        xs = LH(bounds, n_init)
        ys = [f(xs[i,:]) for i = 1:n_init]

        gp = GP(xs, ys, matern, d_matern, nothing, nothing, nothing)

        if hyperparameters == nothing
            MLE(gp, file, epsilon=epsilon, bounds=bounds)
        else
            gp.hyperparameters = hyperparameters
        end

        update_covariance(gp)

        println("LENGTH SCALE l: ", gp.hyperparameters[1])

        for i = 1:size(xs,1)
            println("LHS X     : ", xs[i,:])
            println("  LHS f(X): ", f(xs[i,:]))
        end

        # refine bounds to bounding box of sampled points
        # if n_init > 1
        #     new_bounds = hcat([[min(xs[:,i]...), max(xs[:,i]...)] for i = 1:size(xs,2)]...)'
        #     for i = 1:size(xs,2)
        #         if new_bounds[i,2]-new_bounds[i,1] < 0.2*(bounds[i,2]-bounds[i,1])
        #             new_bounds[i,:] = bounds[i,:]
        #         end
        #     end
        # end
        # println(new_bounds)

        # sequentially add sample points and store their function values
        for i = 1:(n_add-1)
            # find max of EI
            redirect_stdout(file)
            try
                next_x = multistart_optimize(gp, bounds, n_starts ; obj="EI", alpha=alpha, beta=beta, cost=cost)
            catch e
                if isa(e, DomainError)
                    redirect_stdout(SO)
                    println("COVARIANCE MATRIX NEARLY SINGULAR")
                    break
                else
                    rethrow(e)
                end
            end
            redirect_stdout(SO)

            println("NEXT X     : ", next_x)
            println("  NEXT f(X): ", f(next_x))

            # if min([norm(next_x' - xs[i,:]) for i = 1:size(xs,1)]...) < 0.1
            #     break
            # end

            add_data(gp, next_x', f(next_x))
            update_covariance(gp)
        end

        # find min of mean
        redirect_stdout(file)
        next_x = multistart_optimize(gp, bounds, n_starts ; obj="mean")
        redirect_stdout(SO)

        println("FINAL X   : ", next_x)
        println("FINAL f(X): ", f(next_x))

        add_data(gp, next_x', f(next_x))

        min_val, min_ind = findmin(gp.ys)

        println("BEST f(X): ", min_val, "\n")

        return gp.xs[min_ind,:], min_val
    end


    function LH_DOE(bounds, n_tot, f)
        xs = LH(bounds, n_tot)
        ys = [f(xs[i,:]) for i = 1:n_tot]

        min_val, min_ind = findmin(ys)

        return xs[min_ind,:], min_val
    end

end
