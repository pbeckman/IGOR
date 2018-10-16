module gaussian_process

    export GP, gp_mean, gp_var, add_data, update_covariance, matern, d_matern, cov_mat

    mutable struct GP
        xs
        ys
        kernel
        d_kernel
        hyperparameters
        K
        K_inv
    end

    function gp_mean(gp, x)
        m, n = size(gp.xs)
        k = [gp.kernel(gp.xs[i,:], x, gp.hyperparameters) for i = 1:m]
        return sum(k[i] * (gp.K_inv * gp.ys)[i] for i = 1:m)
    end

    function gp_var(gp, x)
        m, n = size(gp.xs)
        k = [gp.kernel(gp.xs[i,:], x, gp.hyperparameters) for i = 1:m]
        return gp.kernel(x, x, gp.hyperparameters) - sum(sum(k[i] * gp.K_inv[i,j] * k[j] for i = 1:m) for j = 1:m)
    end

    function add_data(gp, xs, ys)
        gp.xs = vcat(gp.xs, xs)
        gp.ys = vcat(gp.ys, ys)
    end

    function update_covariance(gp)
        gp.K = cov_mat(gp, gp.xs, gp.xs)
        gp.K_inv = inv(gp.K)
    end

    # squared exponential
    # a = 1e8
    # l = 0.5
    # kernel(x, y) = a * exp(-1.0/2.0*l^2 * norm(x - y)^2)
    # matern 5/2
    function matern(x, y, hyperparameters)
        l = hyperparameters[1]
        r = norm(x - y)
        return (1 + 5^0.5*r/l + 5*r^2/(3*l^2)) * exp(-5^0.5*r/l)
    end

    function d_matern(x, y, hyperparameters)
        l = hyperparameters[1]
        r = norm(x - y)
        return (5*r^2/(3*l^2) + 5^1.5*r^3/(3*l^4)) * exp(-sqrt(5)*r/l)
    end

    function cov_mat(gp, xs1, xs2)
        m1 = size(xs1,1)
        m2 = size(xs2,1)

        mat = zeros(m1, m2)

        for i = 1:m1
            for j = 1:m2
                mat[i,j] = gp.kernel(xs1[i,:], xs2[j,:], gp.hyperparameters)
            end
        end

        return mat
    end

end
