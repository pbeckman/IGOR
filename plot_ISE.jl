### ARGUMENTS:
### dimension, input.csv, output.png

using PyPlot
using DataFrames
using CSV

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
