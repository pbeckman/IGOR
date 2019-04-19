module kernels

    using SpecialFunctions

    export SE, NS_SE, matern, d_matern, dd_matern, wendland2

    ### SQUARED EXPONENTIAL

    function SE(x, y, hyperparameters, args)
        α, l = hyperparameters
        # squared norm of difference
        sn = sum((x[i] - y[i])^2 for i=1:length(x)) # norm(x-y)^2
        return α * exp(-1.0/(2.0 * l^2) * sn)
    end

    function d_SE(x, y, d, hyperparameters, args)
        α, l = hyperparameters
        sn = sum((x[i] - y[i])^2 for i=1:length(x))
        return - α * 1.0/l^2 * (x[d] - y[d]) * exp(-1.0/(2.0 * l^2) * sn)
    end

    function dd_SE(x, y, d1, d2, hyperparameters, args)
        α, l = hyperparameters
        sn = sum((x[i] - y[i])^2 for i=1:length(x))
        return α * 1.0/l^2 * ((d1 == d2) - 1/l^2 * (x[d1] - y[d1])*(x[d2] - y[d2])) * exp(-1.0/(2.0 * l^2) * sn)
    end

    ### MATERN

    function matern(x, y, hyperparameters, args)
        α, l = hyperparameters
        v = args[1]
        r = norm(x - y)

        if v == 0.5
            return α * exp(-r / l)
        elseif v == 1.5
            return α * (1 + 3^0.5*r/l) * exp(-3^0.5*r/l)
        elseif v == 2.5
            return α * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
        else
            return α * 1.0/(gamma(v) * 2^(v-1)) * (2.0*sqrt(v)*r/l)^v * besselk(v, 2.0*sqrt(v)*r/l)
        end
    end

    function d_matern(x, y, d, hyperparameters, args)
        α, l = hyperparameters
        v = args[1]
        r = norm(x - y)

        if v == 2.5
            return -α * 5.0 * (x[d] - y[d]) / (3*l^3) * (l + sqrt(5)*r) * exp(-sqrt(5)*r/l)
        else
            throw("derivative differentiability parameter v must be 2.5")
        end
    end

    function dd_matern(x, y, d1, d2, hyperparameters, args)
        α, l = hyperparameters
        v = args[1]
        r = norm(x - y)

        if v == 2.5
            if d1 == d2
                return α * 5.0/(3*l^4) * (l^2 + sqrt(5) * l * r - 5 * (x[d1] - y[d1])^2) * exp(-5^0.5*r/l)
            else
                return α * 25.0 * (x[d1] - y[d1]) * (x[d2] - y[d2]) / (3*l^4) * exp(-5^0.5*r/l)
            end
        else
            throw("derivative differentiability parameter v must be 2.5")
        end
    end

    ### 1D VARIABLE LENGTH SCALE MATERN

    function NS_matern(x, y, hyperparameters, args)
        α, l_min, l_max, β = hyperparameters
        v, t = args
        r = norm(x - y)

        l_f(x) = (l_max - l_min) * exp(-1 / (β * t(x))) + l_min

        c = sqrt(2) * l_f(x)^1/4 * l_f(y)^1/4 / (l_f(x) + l_f(y))^1/2
        l = (l_f(x) + l_f(y)) / 2

        if v == 0.5
            return α * c * exp(-r / l)
        elseif v == 1.5
            return α * c * (1 + 3^0.5*r/l) * exp(-3^0.5*r/l)
        elseif v == 2.5
            return α * c * (1 + 5^0.5*r/l + 5*r^2/(3*l^2)) * exp(-5^0.5*r/l)
        else
            return α * c * 1.0/(gamma(v) * 2^(v-1)) * (2.0*sqrt(v)*r/l)^v * besselk(v, 2.0*sqrt(v)*r/l)
        end
    end

    ### WENDLAND2

    function wendland2(x, y, hyperparameters, args)
        l_t = hyperparameters[1]
        t = args[1]
        h = norm(x - y)

        r = l_t * t(x, y)

        if h < r
            return (1 - h/r)^6 * (1 + 6*h/r + 35*h^2/(3*r^2))
        else
            return 0
        end
    end
end
