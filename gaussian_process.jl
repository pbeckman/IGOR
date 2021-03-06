module gaussian_process

    using ForwardDiff

    export GP, gp_mean, gp_var, add_data, update_covariance,
           cov_mat, posterior, stack

    mutable struct GP_struct
        xs
        ys
        ds # derivative data
        ts # trigger distance data
        kernel
        d_kernel
        dd_kernel
        hyperparameters
        args
        noise_params
        K
        K_inv
        use_ds
        use_AD
        use_noise
    end

    function GP(; xs=nothing, ys=nothing, ds=nothing, ts=nothing,
                kernel=nothing, d_kernel=nothing, dd_kernel=nothing,
                hyperparameters=nothing, args=nothing, noise_params=nothing,
                K=nothing, K_inv=nothing,
                use_ds=false, use_AD=true, use_noise=false)
        # constructor function for GPs
        return GP_struct(xs, ys, ds, ts, kernel, d_kernel, dd_kernel, hyperparameters, args, noise_params, K, K_inv, use_ds, use_AD, use_noise)
    end

    function add_data(gp, xs, ys ; ds=nothing, ts=nothing)
        p = size(xs, 2)
        # add oberservations
        # reformat ys if it is scalar
        if isscalar(ys)
            ys = [ys]
        end
        # xs, ds, and ts must be 1xp arrays
        gp.xs = vcat(gp.xs, reshape(xs, (1, p)))
        gp.ys = vcat(gp.ys, reshape(ys, (1, 1)))
        if ds != nothing
            gp.ds = vcat(gp.ds, reshape(ds, (1, p)))
        end
        if ts != nothing
            gp.ts = vcat(gp.ts, reshape(ts, (1, p)))
        end
    end

    function cov_mat(gp, xs1, xs2 ; is_K=false)
        # form covariance matrix using AD for derivatives
        p = size(xs1, 2)
        n1 = size(xs1, 1)
        n2 = size(xs2, 1)

        # takes on ForwardDiff.Dual type if a dual is passed to the function
        T = Float64
        for obj in [xs1, xs2, gp.hyperparameters, gp.noise_params]
            if obj != nothing && !(typeof(vec(obj)[1]) in [Float64, Float32, Int64, Int32])
                T = typeof(vec(obj)[1])
            end
        end

        if gp.use_ds
            K = zeros(T, (p+1)*n1, (p+1)*n2)
        else
            K = zeros(T, n1, n2)
        end

        for i = 1:n1
            for j = 1:n2
                # fill function-function block with kernel values
                K[i,j] = gp.kernel(xs1[i,:], xs2[j,:], gp.hyperparameters, gp.args)

                if gp.use_ds
                    # starting index of derivative sub-blocks given the dimension of the space
                    s_j = n2 + p*(j-1)
                    s_i = n1 + p*(i-1)

                    if gp.use_AD
                        # fill function-derivative blocks with derivative kernel values
                        K[(s_i + 1):(s_i + p), j] = ForwardDiff.gradient(
                            x1 -> gp.kernel(x1, xs2[j,:], gp.hyperparameters, gp.args),
                            xs1[i,:]
                        )
                        K[i, (s_j + 1):(s_j + p)] = ForwardDiff.gradient(
                            x2 -> gp.kernel(xs1[i,:], x2, gp.hyperparameters, gp.args),
                            xs2[j,:]
                        )

                        # fill derivative-derivative block with second order derivative kernel values
                        # this is computed as J_z ∇_x k(x_i, z_j)
                        # it depends on the order or arguments to the kernel and is NOT symmetric for x≠z
                        K[(s_i + 1):(s_i + p), (s_j + 1):(s_j + p)] = ForwardDiff.jacobian(
                            x2 -> ForwardDiff.gradient(
                                x1 -> gp.kernel(x1, x2, gp.hyperparameters, gp.args),
                                xs1[i,:]
                            ),
                            xs2[j,:]
                        )
                    else
                        # fill function-derivative blocks with derivative kernel values
                        for d = 1:p
                            K[i, s_j + d] = gp.d_kernel(xs2[j,:], xs1[i,:], d, gp.hyperparameters, gp.args)
                            K[s_i + d, j] = gp.d_kernel(xs1[i,:], xs2[j,:], d, gp.hyperparameters, gp.args)
                        end
                        # fill derivative-derivative block with second order derivative kernel values
                        for d1 = 1:p
                            for d2 = 1:p
                                K[s_i + d1, s_j + d2] = gp.dd_kernel(xs1[i,:], xs2[j,:], d1, d2, gp.hyperparameters, gp.args)
                            end
                        end
                    end
                end
            end
        end

        # add derivative noise to lower right block
        if is_K && gp.use_noise && gp.use_ds
            n = n1
            max_noise, β = gp.noise_params
            Σ = zeros(T, n*p, n*p)
            for i=1:n
                for j=1:p
                    Σ[p*(i-1)+j, p*(i-1)+j] = max_noise * exp(-gp.ts[i, j] / β)
                end
            end
            K += [zeros(n, n) zeros(n, n*p); zeros(n*p, n) Σ]
        end

        return K
    end

    function update_covariance(gp)
        gp.K = cov_mat(gp, gp.xs, gp.xs ; is_K=true)

        # gp.K_inv = inv(gp.K)
    end

    function posterior(gp, xs)
        # compute posterior mean and variance using linear solves
        n, p = size(xs)

        if gp.use_ds
            ys = stack(gp.ys, gp.ds)
        else
            ys = gp.ys
        end

        K_test = cov_mat(gp, xs, xs)
        K_cross = cov_mat(gp, xs, gp.xs)
        K = gp.K

        # println(K_cross)
        # println(K)
        # println(ys)
        # println("====")

        mean = K_cross * (K \ ys)
        cov  = K_test - K_cross * (K \ transpose(K_cross))

        return mean[1:n], cov[1:n,1:n]
    end

    function gp_mean(gp, x)
        # compute mean using K inverse
        # less stable but works with JuMP AD
        n, p = size(gp.xs)

        if gp.use_ds
            ys = stack(gp.ys, gp.ds)
            n = n*(p+1)
        else
            ys = gp.ys
        end

        # k = [gp.kernel(gp.xs[i,:], x, gp.hyperparameters, gp.args) for i = 1:m]
        k = cov_mat(gp, gp.xs, x)[:,1]
        return sum(k[i] * (gp.K_inv * ys)[i] for i = 1:n)
    end

    function gp_var(gp, x)
        # compute var using K inverse
        # less stable but works with JuMP AD
        n, p = size(gp.xs)

        if gp.use_ds
            n = n*(p+1)
        end

        # k = [gp.kernel(gp.xs[i,:], x, gp.hyperparameters, gp.args) for i = 1:m]
        k = cov_mat(gp, gp.xs, x)[:,1]
        v = gp.kernel(x, x, gp.hyperparameters, gp.args) - sum(sum(k[i] * gp.K_inv[i,j] * k[j] for i = 1:n) for j = 1:n)
        return v
    end

    function stack(ys, ds)
        return vcat(ys, [ds[i, :] for i=1:size(ds, 1)]...)
    end

    function isscalar(x)
        return length(size(x)) == 0
    end
end
