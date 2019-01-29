module gaussian_process

    using ForwardDiff

    export GP, gp_mean, gp_var, add_data, update_covariance,
           cov_mat, posterior, stack

    mutable struct GP
        xs
        ys
        ds
        ts
        kernel
        d_kernel
        dd_kernel
        d_hp_kernel
        hyperparameters
        args
        K
        K_inv
    end

    function add_data(gp, xs, ys ; ds=nothing, ts=nothing)
        gp.xs = vcat(gp.xs, xs)
        gp.ys = vcat(gp.ys, ys)
        if ds != nothing
            gp.ds = vcat(gp.ds, ds)
        end
        if ts != nothing
            gp.ts = vcat(gp.ts, ts)
        end
    end

    # function cov_mat(gp, xs1, xs2 ; use_ds=false)
    #     # form covariance matrix, using derivatives if desired
    #     n = size(xs1,2)
    #     m1 = size(xs1,1)
    #     m2 = size(xs2,1)
    #
    #     if use_ds
    #         K = zeros(typeof(xs1[1]), (n+1)*m1, (n+1)*m2)
    #     else
    #         K = zeros(typeof(xs1[1]), m1, m2)
    #     end
    #
    #     # fill function-function matrix block
    #     for i = 1:m1
    #         for j = 1:m2
    #             K[i,j] = gp.kernel(xs1[i,:], xs2[j,:], gp.hyperparameters, gp.args)
    #             # if use_ds
    #             #     s_j = m2 + n*(j-1)
    #             #     s_i = m1 + n*(i-1)
    #             #
    #             #     K[i, (s_j + 1):(s_j + n)] = gp.d_kernel(xs2[j,:], xs1[i,:], gp.hyperparameters)
    #             #     K[(s_i + 1):(s_i + n), j] = gp.d_kernel(xs1[i,:], xs2[j,:], gp.hyperparameters)
    #             #
    #             #     K[m1 + i, (s_j + 1):(s_j + n)] = gp.dd_kernel(xs2[j,:], xs1[i,:], gp.hyperparameters)
    #             #     K[(s_i + 1):(s_i + n), j] = gp.dd_kernel(xs1[i,:], xs2[j,:], gp.hyperparameters)
    #             # end
    #         end
    #     end
    #
    #     if use_ds
    #         # fill function-derivative matrix blocks
    #         for i = 1:m1
    #             for j = 1:m2
    #                 for d = 1:n
    #                     K[i, m2 + n*(j-1) + d] = gp.d_kernel(xs2[j,:], xs1[i,:], d, gp.hyperparameters, gp.args)
    #                     K[m1 + n*(i-1) + d, j] = gp.d_kernel(xs1[i,:], xs2[j,:], d, gp.hyperparameters, gp.args)
    #                 end
    #             end
    #         end
    #
    #         # fill derivative-derivative matrix block
    #         for i = 1:m1
    #             for d1 = 1:n
    #                 for j = 1:m2
    #                     for d2 = 1:n
    #                         K[m1 + n*(i-1) + d1, m2 + n*(j-1) + d2] = gp.dd_kernel(xs1[i,:], xs2[j,:], d1, d2, gp.hyperparameters, gp.args)
    #                     end
    #                 end
    #             end
    #         end
    #     end
    #
    #     return K
    # end

    function cov_mat(gp, xs1, xs2 ; use_ds=false)
        # form covariance matrix, using derivatives if desired
        n = size(xs1,2)
        m1 = size(xs1,1)
        m2 = size(xs2,1)

        if use_ds
            K = zeros(typeof(xs1[1]), (n+1)*m1, (n+1)*m2)
        else
            K = zeros(typeof(xs1[1]), m1, m2)
        end

        for i = 1:m1
            for j = 1:m2
                # fill function-function block with kernel values
                K[i,j] = gp.kernel(xs1[i,:], xs2[j,:], gp.hyperparameters, gp.args)

                if use_ds
                    # starting index of derivative sub-blocks given the dimension of the space
                    s_j = m2 + n*(j-1)
                    s_i = m1 + n*(i-1)

                    # fill function-derivative blocks with derivative kernel values
                    K[(s_i + 1):(s_i + n), j] = ForwardDiff.gradient(
                        x1 -> gp.kernel(x1, xs2[j,:], gp.hyperparameters, gp.args),
                        xs1[i,:]
                    )
                    K[i, (s_j + 1):(s_j + n)] = ForwardDiff.gradient(
                        x2 -> gp.kernel(xs1[i,:], x2, gp.hyperparameters, gp.args),
                        xs2[j,:]
                    )

                    # fill derivative-derivative block with second order derivative kernel values
                    # this is computed as J_z ∇_x k(x_i, z_j)
                    # it depends on the order or arguments to the kernel and is NOT symmetric for x≠z
                    K[(s_i + 1):(s_i + n), (s_j + 1):(s_j + n)] = ForwardDiff.jacobian(
                        x2 -> ForwardDiff.gradient(
                            x1 -> gp.kernel(x1, x2, gp.hyperparameters, gp.args),
                            xs1[i,:]
                        ),
                        xs2[j,:]
                    )
                end
            end
        end

        return K
    end

    function update_covariance(gp ; use_ds=false)
        gp.K = cov_mat(gp, gp.xs, gp.xs, use_ds=use_ds)
        # gp.K_inv = inv(gp.K)
    end

    function posterior(gp, xs ; use_ds=false)
        # compute posterior mean and variance using default linear solves
        m, n = size(xs)

        if use_ds
            ys = stack(gp.ys, gp.ds)
        else
            ys = gp.ys
        end

        # println(size(cov_mat(gp, xs, gp.xs, use_ds=use_ds)), " ", size(gp.K), " ", size(ys))

        K_test = cov_mat(gp, xs, xs, use_ds=use_ds)
        K_cross = cov_mat(gp, xs, gp.xs, use_ds=use_ds)
        K = gp.K

        # display(K_test)
        # display(K_cross)
        # display(K)
        # readline(STDIN)

        mean = K_cross * (K \ ys)
        cov  = K_test - K_cross * (K \ transpose(K_cross))

        return mean[1:m], cov[1:m,1:m]
    end

    function gp_mean(gp, x)
        # compute mean using K inverse
        # less stable but works with JuMP AD
        m, n = size(gp.xs)
        k = [gp.kernel(gp.xs[i,:], x, gp.hyperparameters, gp.args) for i = 1:m]
        return sum(k[i] * (gp.K_inv * gp.ys)[i] for i = 1:m)
    end

    function gp_var(gp, x)
        # compute var using K inverse
        # less stable but works with JuMP AD
        m, n = size(gp.xs)
        k = [gp.kernel(gp.xs[i,:], x, gp.hyperparameters, gp.args) for i = 1:m]
        return gp.kernel(x, x, gp.hyperparameters, gp.args) - sum(sum(k[i] * gp.K_inv[i,j] * k[j] for i = 1:m) for j = 1:m)
    end

    function stack(ys, ds)
        return vcat(ys, [ds[i,:]' for i=1:size(ds,1)]...)
    end

end
