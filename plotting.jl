module plotting
    using PyPlot

    include("./gaussian_process.jl")
    import gaussian_process: posterior

    export plot_posterior

    function plot_posterior(gp, bounds, n_grid ; f=nothing)
        # num oberservations, dimension
        n, p = size(gp.xs)

        if p == 1
            # make grid on which to compute posterior
            g = reshape(linspace(bounds[1], bounds[2], n_grid), (n_grid, 1))

            # compute posterior mean and covariance of GP
            mean, cov = posterior(gp, g)
            std_dev = [sqrt(max(0, cov[i, i])) for i=1:size(cov,1)]

            if f != nothing
                # plot true function
                plot(g, [f(x) for x in g], color="black")
            end

            # plot GP
            plot(g, mean, color="red")
            fill_between(vec(g), vec(mean - 2*std_dev), vec(mean + 2*std_dev), color="#dddddd")

            # plot data
            scatter(gp.xs, gp.ys, color="black")
        elseif p == 2
            # make grid on which to compute posterior
            g = vcat([[i j] for i=bounds[1, 1]:(bounds[1, 2] - bounds[1, 1])/n_grid:bounds[1, 2], j=bounds[2, 1]:(bounds[2, 2] - bounds[2, 1])/n_grid:bounds[2, 2]]...)

            # compute posterior mean and covariance of GP
            mean, cov = posterior(gp, g)

            if f != nothing
                # plot true function
                surf(g[:,1], g[:,2], [f(g[i,:]) for i=1:size(g,1)], color="blue")
            end

            # plot GP
            surf(g[:,1], g[:,2], mean, color="red")

            # plot data
            scatter3D(gp.xs[:,1], gp.xs[:,2], gp.ys, color="black")
        end
    end
end
