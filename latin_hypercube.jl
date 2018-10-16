module latin_hypercube

    function LH(bounds, n_points)
        dim = size(bounds,1)

        xs = []

        # place point randomly in grid sqaure
        grid = [[(bounds[i,2] - bounds[i,1]) * (j-rand())/n_points for j = 1:n_points] for i = 1:dim]
        n_left = n_points

        for i = 1:n_points
            ind = rand(1:n_left, dim)
            push!(xs, [grid[j][ind[j]] for j = 1:dim])
            for j = 1:dim
                deleteat!(grid[j], ind[j])
            end
            n_left -= 1
        end

        return hcat(xs...)'
    end
    export LH

end
