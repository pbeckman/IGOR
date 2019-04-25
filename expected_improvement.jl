module expected_improvement

    using JuMP
    using Ipopt
    using Distributions

    include("./latin_hypercube.jl")
    import latin_hypercube: LH
    include("./gaussian_process.jl")
    import gaussian_process: GP, gp_mean, gp_var

    export multistart_optimize

    function build_model(gp, bounds, x0 ; obj="EI", alpha=1, beta=0, cost=x->0)
        m, n = size(gp.xs)

        function mean(x...)
            x = collect(x)
            return gp_mean(gp, x)
        end

        function var(x...)
            x = collect(x)
            return gp_var(gp, x)
        end

        function EI(x...)
            s = sqrt(var(x...) + 1e-10)
            y_hat = mean(x...)
            c = cost(collect(x))
            return -1*(
                (alpha * s - y_hat - beta * c) * cdf(Normal(), alpha - (y_hat + beta * c)/s) + s * pdf(Normal(), alpha - (y_hat + beta * c)/s)
                )
        end

        # println("making model")
        model = Model(solver = IpoptSolver(max_iter=500))

        # println("building variables")
        vars = [@variable(model, start=x0[i]) for i = 1:n]
        # @constraintref constrs[1:n]
        for i = 1:n
          @constraint(model, bounds[i,1] <= vars[i] <= bounds[i,2])
        end

        # println("building objective")
        JuMP.register(model, :mean, n, mean, autodiff=true)
        JuMP.register(model, :var,  n, var,  autodiff=true)
        JuMP.register(model, :EI,   n, EI,   autodiff=true)

        if obj == "EI"
            JuMP.setNLobjective(model, :Min, Expr(:call, :EI, vars...))
        elseif obj == "mean"
            JuMP.setNLobjective(model, :Min, Expr(:call, :mean, vars...))
        else
            throw(error("invalid objective supplied to build_model"))
        end

        return (model, vars)
    end


    function multistart_optimize(model, vars, bounds, n_starts)
        p = size(bounds, 1)

        min_val = Inf
        min_x = nothing

        starts = LH(bounds, n_starts)

        for i = 1:n_starts
            for k = 1:p
                setvalue(vars[k], starts[i, k])
            end

            status = solve(model)

            # println("\nx0 = ", x0s[i,:])
            # println("Objective value: ", getobjectivevalue(model))
            # println("x = ", [getvalue(v) for v in vars])

            obj_val = getobjectivevalue(model)

            if obj_val < min_val
                min_val = obj_val
                min_x = [getvalue(v) for v in vars]
            end
        end

        return min_x
    end
end
