module trigger_approx

    export trig_dists, nn_trig_dists

    function linear_approx_trigger(x, x0, g, ∇g)
        p = length(x)
        # distance from x to trigger in each coordinate direction
        # based on linear approximation at x0
        v = [-∇g[i]^(-1) * (∇g⋅(x-x0) + g) for i=1:p]
        return reshape(v, (1, length(v)))
    end

    function trig_dists(x0, gs, ∇gs)
        p, k = size(∇gs)

        # matrix in which each row is the distance in each coordinate direction for a given trigger
        T = vcat([
            linear_approx_trigger(x0, x0, gs[j], ∇gs[:,j]) for j=1:k
            ]...)
        T[T .== Inf] = 1e10
        T[T .== -Inf] = 1e10
        t, inds = findmin(abs.(T), 1)
        # println("T: ", T)
        # println("t: ", t)
        # println("inds: ", inds)
        ∇t = [-sign(T[inds[j]]) for j=1:p]
        ∇t = reshape(∇t, (1, length(∇t)))
        # return trigger distance and derivative
        # i.e. distance and whether you are moving towards or away from a kink
        return t, ∇t
    end

    function nn_trig_dists(x, x0s, gs, ∇gs)
        n, p, k = size(∇gs)

        # index of nearest neighbor
        nn = findmin([norm(x-x0s[i,:]) for i=1:n])[2]
        # println([norm(x-x0s[i,:]) for i=1:n])
        # println(nn)

        # println(vcat([
        #     single_trig_dists(x, x0s[nn,:], gs[nn,j], ∇gs[nn,:,j]) for j=1:k
        #     ]...))
        # minimum distance in each coordinate direction to a trigger approximation
        return minimum(
            abs.(
                vcat([
                    linear_approx_trigger(x, x0s[nn,:], gs[nn,j], ∇gs[nn,:,j]) for j=1:k
                    ]...)
                ),
            1
        )
    end

end


⋅
