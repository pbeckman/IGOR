module maximum_likelihood

    using JuMP
    using Ipopt
    using PyPlot
    using Iterators
    using ForwardDiff

    include("./gaussian_process.jl")
    import gaussian_process: cov_mat, update_covariance, stack

    export MLE

    function MLE(gp, bounds, starts, file ; target="all")
        SO = STDOUT

        # number of observations, dimension
        n, p = size(gp.xs)

        # number of parameters
        if target == "hyperparameters"
            n_params = length(gp.hyperparameters)
        elseif target == "noise"
            n_params = length(gp.noise_params)
        elseif target == "all"
            n_params = length(gp.hyperparameters) + length(gp.noise_params)
        end

        # log likelihood
        function logL(params...)
            assign_params(gp, params, target)

            update_covariance(gp)

            if gp.use_ds
                return (
                    -0.5 * stack(gp.ys, gp.ds)' * (gp.K \ stack(gp.ys, gp.ds)) -
                     0.5 * log(abs(det(gp.K)))
                        )[1]
            else
                return (
                    -0.5 * gp.ys' * (gp.K \ gp.ys) -
                     0.5 * log(abs(det(gp.K)))
                        )[1]
            end
        end

        # gradient of log likelihood
        function ∇logL(g, params...)
            assign_params(gp, params, target)

            update_covariance(gp)

            if gp.use_ds
                alpha = gp.K \ stack(gp.ys, gp.ds)
                d_Ks = zeros(n_params, n*(p+1), n*(p+1))
            else
                alpha = gp.K \ gp.ys
                d_Ks = zeros(n_params, n, n)
            end

            # Jacobian of transformation which takes a vector of parameters
            # to the vectorized covariance matrix
            D = ForwardDiff.jacobian(
                function (ps)
                    if target == "hyperparameters"
                        gp.hyperparameters = ps
                    elseif target == "noise"
                        gp.noise_params = ps
                    elseif target == "all"
                        gp.hyperparameters = ps[1:length(gp.hyperparameters)]
                        gp.noise_params = ps[length(gp.hyperparameters)+1:end]
                        # println("hp: ", gp.hyperparameters, ", noise: ", gp.noise_params)
                    end
                    return cov_mat(gp, gp.xs, gp.xs, is_K=true)
                end,
                if target == "hyperparameters"
                    gp.hyperparameters
                elseif target == "noise"
                    gp.noise_params
                elseif target == "all"
                    vcat(gp.hyperparameters, gp.noise_params)
                end
            )

            # reshape columns of the Jacobian into matrices containing the
            # derivative of each covariance in terms of a single hyperparameter
            if gp.use_ds
                for k = 1:n_params
                    d_Ks[k,:,:] = reshape(D[:,k], (n*(p+1), n*(p+1)))
                end
            else
                for k = 1:n_params
                    d_Ks[k,:,:] = reshape(D[:,k], (n,n))
                end
            end

            # construct gradient vector
            for k = 1:n_params
                g[k] = 0.5 * trace(alpha * alpha' * d_Ks[k,:,:] - gp.K \ d_Ks[k,:,:])
            end
        end

        # construct model
        model = Model(solver = IpoptSolver(max_iter=500))

        vars = Array{Any}(n_params)
        for k = 1:n_params
            vars[k] = @variable(model, lowerbound = bounds[k, 1], upperbound = bounds[k, 2], start = starts[k])
        end

        JuMP.register(model, :likelihood, n_params, logL, ∇logL)

        JuMP.setNLobjective(model, :Max, Expr(:call, :likelihood, vars...))

        # output optimization procedure printed statements to file
        redirect_stdout(file)
        status = solve(model)
        redirect_stdout(SO)

        if status != :Optimal && status != :UserLimit
            println("MLE FAILED")
            throw(ErrorException("MLE FAILED"))
        end

        # set hyperparameters or noise parameters to optimal MLE values
        if target == "hyperparameters"
            gp.hyperparameters = [getvalue(vars[k]) for k = 1:n_params]
        elseif target == "noise"
            gp.noise_params = [getvalue(vars[k]) for k = 1:n_params]
        elseif target == "all"
            gp.hyperparameters = [getvalue(vars[k]) for k = 1:length(gp.hyperparameters)]
            gp.noise_params = [getvalue(vars[k]) for k = (length(gp.hyperparameters)+1):n_params]
        end
    end

    function assign_params(gp, params, target)
        ps = collect(params)

        if target == "hyperparameters"
            gp.hyperparameters = ps
        elseif target == "noise"
            gp.noise_params = ps
        elseif target == "all"
            gp.hyperparameters = ps[1:length(gp.hyperparameters)]
            gp.noise_params = ps[length(gp.hyperparameters)+1:end]
        end
    end
end
