module kernels

    export SE, d_SE, dd_SE,
           matern, d_matern, dd_matern, d_hp_matern,
           wendland2, d_wendland2, dd_wendland2,
           prod, d_prod, dd_prod, d_hp_prod

    ### SQUARED EXPONENTIAL

    function SE(x, y, hyperparameters, args)
        l = hyperparameters[1]
        return exp(-1.0/(2.0 * l^2) * norm(x - y)^2)
    end

    function d_SE(x, y, d, hyperparameters, args)
        l = hyperparameters[1]
        return - 1.0/l^2 * (x[d] - y[d]) * exp(-1.0/(2.0 * l^2) * norm(x - y)^2)
    end

    function dd_SE(x, y, d1, d2, hyperparameters, args)
        l = hyperparameters[1]
        return 1.0/l^2 * ((d1 == d2) - 1/l^2 * (x[d1] - y[d1])*(x[d2] - y[d2])) * exp(-1.0/(2.0 * l^2) * norm(x - y)^2)
    end

    # function grad_SE(x, y, hyperparameters)
    #     l = hyperparameters[1]
    #     return - 1.0/l^2 * (x - y) * exp(-1.0/(2.0 * l^2) * norm(x - y)^2)
    # end
    #
    # function hess_SE(x, y, hyperparameters)
    #     l = hyperparameters[1]
    #     return  1.0/l^2 * (eye(size(x,1)) - 1.0/l^2 * (x-y)*(x-y)') * exp(-1.0/(2.0 * l^2) * norm(x - y)^2)
    # end

    ### MATERN

    function matern(x, y, hyperparameters, args)
        l = hyperparameters[1]
        v = args[1]
        r = norm(x - y)

        if v == 0.5
            return exp(- r / l)
        elseif v == 1.5
            return (1 + 3^0.5*r/l) * exp(-3^0.5*r/l)
        elseif v == 2.5
            return (1 + 5^0.5*r/l + 5*r^2/(3*l^2)) * exp(-5^0.5*r/l)
        else
            throw("differentiability parameter v must be 0.5, 1.5, or 2.5")
        end
    end

    function d_matern(x, y, d, hyperparameters, args)
        l = hyperparameters[1]
        v = args[1]
        r = norm(x - y)

        if v == 2.5
            return -5.0 * (x[d] - y[d]) / (3*l^2) * (1 + 5^0.5*r/l) * exp(-5^0.5*r/l)
        else
            throw("derivative differentiability parameter v must be 2.5")
        end
    end

    function dd_matern(x, y, d1, d2, hyperparameters, args)
        l = hyperparameters[1]
        v = args[1]
        r = norm(x - y)

        if v == 2.5
            if d1 == d2
                return 5.0/(3*l^4) * (l^2 + 5^0.5 * l * r - 5 * (x[d1] - y[d1])^2) * exp(-5^0.5*r/l)
            else
                return 25.0 * (x[d1] - y[d1]) * (x[d2] - y[d2]) / (3*l^4) * exp(-5^0.5*r/l)
            end
        else
            throw("derivative differentiability parameter v must be 2.5")
        end
    end

    function d_hp_matern(x, y, hyperparameters, args)
        l = hyperparameters[1]
        v = args[1]
        r = norm(x - y)

        # return 0 in the first component as the differentiability parameter is constant
        if v == 0.5
            return [r/l^2 * exp(- r / l)] # [0, r/l^2 * exp(- r / l)]
        elseif v == 1.5
            return [3*r^2/l^3 * exp(-3^0.5*r/l)] # [0, 3*r^2/l^3 * exp(-3^0.5*r/l)]
        elseif v == 2.5
            return [(5*r^2 * (l + 5^0.5*r))/(3*l^4) * exp(-5^0.5*r/l)] # [0, (5*r^2 * (l + 5^0.5*r))/(3*l^4) * exp(-5^0.5*r/l)]
        else
            throw("differentiability parameter v must be 0.5, 1.5, or 2.5")
        end
    end

    ### WENDLAND2

    function wendland2(x, y, hyperparameters, args)
        l_t = hyperparameters[1]
        s = args[1]
        h = norm(x - y)

        r = l_t * s(x,y)

        if h < r
            return (1 - h/r)^6 * (1 + 6*h/r + 35*h^2/(3*r^2))
        else
            return 0
        end
    end

    function d_wendland2(x, y, d, hyperparameters, args)
        l_t = hyperparameters[1]
        s = args[1]
        h = norm(x - y)

        r = l_t * s(x,y)

        if h < r
            return -56 * (x[d] - y[d]) * (r - h)^5 * (r + 5*h) / (3*r^8)
        else
            return 0
        end
    end

    function dd_wendland2(x, y, d1, d2, hyperparameters, args)
        l_t = hyperparameters[1]
        s = args[1]
        h = norm(x - y)

        r = l_t * s(x,y)

        if h < r
            if d1 == d2
                return 56 * (h - r)^4 * (r^2 + 4*r*h - 30*(x[d1] - y[d1])^2 - 5*h^2) / (3*r^8)
            else
                return 560 * (x[d1] - y[d1]) * (x[d2] - y[d2]) * (r-h)^4 / r^8
            end
        else
            return 0
        end
    end

    # function d_hp_wendland2(x, y, hyperparameters, args)
    #     l_t = hyperparameters[1]
    #     s = args[1]
    #     r = norm(x - y)
    #
    #     l = l_t * s(x,y)
    #
    #     # return 0 in the first component as the differentiability parameter is constant
    #     if v == 0.5
    #         return [0, r/l^2 * exp(- r / l)]
    #     elseif v == 1.5
    #         return [0, 3*r^2/l^3 * exp(-3^0.5*r/l)]
    #     elseif v == 2.5
    #         return [0, (5*r^2 * (l + 5^0.5*r))/(3*l^4) * exp(-5^0.5*r/l)]
    #     else
    #         throw("differentiability parameter v must be 0.5, 1.5, or 2.5")
    #     end
    # end

    ## PRODUCT KERNELS

    # use product rule to generate derivatives for product kernel
    # kernel 1, kernel 2, number of hyperparameters for kernel 1, number of args for kernel 1
    # the rest will be assigned to kernel 2
    function prod(k1, k2, n_hp1, n_args1)
        return (x, y, hp, args) -> k1(x, y, hp[1:n_hp1], args[1:n_args1]) * k2(x, y, hp[n_hp1+1:end], args[n_args1+1:end])
    end

    function d_prod(k1, d_k1, k2, d_k2, n_hp1, n_args1)
        return (x, y, d, hp, args) -> d_k1(x, y, d, hp[1:n_hp1], args[1:n_args1]) * k2(x, y, hp[n_hp1+1:end], args[n_args1+1:end]) +
                                      k1(x, y, hp[1:n_hp1], args[1:n_args1]) * d_k2(x, y, d, hp[n_hp1+1:end], args[n_args1+1:end])
    end

    function dd_prod(k1, d_k1, dd_k1, k2, d_k2, dd_k2, n_hp1, n_args1)
        return (x, y, d1, d2, hp, args) -> dd_k1(x, y, d1, d2, hp[1:n_hp1], args[1:n_args1]) * k2(x, y, hp[n_hp1+1:end], args[n_args1+1:end]) +
                                           d_k1(x, y, d1, hp[1:n_hp1], args[1:n_args1]) * d_k2(x, y, d2, hp[n_hp1+1:end], args[n_args1+1:end]) +
                                           d_k1(x, y, d2, hp[1:n_hp1], args[1:n_args1]) * d_k2(x, y, d1, hp[n_hp1+1:end], args[n_args1+1:end]) +
                                           k1(x, y, hp[1:n_hp1], args[1:n_args1]) * dd_k2(x, y, d1, d2, hp[n_hp1+1:end], args[n_args1+1:end])

    end

    function d_hp_prod(d_hp_k1, d_hp_k2, n_hp1, n_args1)
        return (x, y, hp, args) -> vcat(d_hp_k1(x, y, hp[1:n_hp1], args[1:n_args1]), d_hp_k2(x, y, hp[n_hp1+1:end], args[n_args1+1:end]))
    end

end
