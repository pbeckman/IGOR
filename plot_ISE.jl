using PyPlot
using DataFrames
using CSV

function plot_average()
    data_noder = CSV.read(ARGS[1])
    data_der = CSV.read(ARGS[2])
    data_noisy = CSV.read(ARGS[3])

    ns, n_tests = size(data_noder)

    plot(2:(ns+1), [mean(convert(Array, data_noder[i, :])) for i=1:ns], color="red", label="No derivatives")
    plot(2:(ns+1), [mean(convert(Array, data_der[i, :])) for i=1:ns],   color="green", label="Derivatives")
    plot(2:(ns+1), [mean(convert(Array, data_noisy[i, :])) for i=1:ns], color="blue", label="Noisy derivatives")

    # for i=1:ns
    #     d_nd = convert(Array, data_noder[i, :])'
    #     d_d  = convert(Array, data_der[i, :])'
    #     d_n  = convert(Array, data_noisy[i, :])'
    #     errorbar(i+1, mean(d_nd), yerr=0.5*std(d_nd), color="red", capsize=5, alpha=0.5)
    #     errorbar(i+1, mean(d_d),  yerr=0.5*std(d_d),  color="green", capsize=5, alpha=0.5)
    #     errorbar(i+1, mean(d_n),  yerr=0.5*std(d_n),  color="blue", capsize=5, alpha=0.5)
    # end

    legend(loc="upper right")

    show()
end

function plot_direct(p, in_file, out_file)
    data = CSV.read(in_file)

    ns = size(data, 2)

    plot([n^p for n=2:(ns+1)], convert(Array, data[1, :])', color="red", label="No derivatives", linestyle="--", marker="o", markersize=3)
    plot([n^p for n=2:(ns+1)], convert(Array, data[2, :])', color="green", label="Derivatives", linestyle=":", marker="o", markersize=3)
    plot([n^p for n=2:(ns+1)], convert(Array, data[3, :])', color="blue", label="Noisy derivatives", marker="o", markersize=3, alpha=0.5)

    # title("Integrated squared error for 1D regular grid designs")
    legend(loc="upper right")
    xlabel("number of observations")
    ylabel("ISE")
    yscale("log")

    if out_file != nothing
        savefig(out_file, dpi=500)
    end
    show()
end

plot_direct(
    parse(Int, ARGS[1]),
    ARGS[2],
    if length(ARGS) > 2 ARGS[3] else nothing end
    )
