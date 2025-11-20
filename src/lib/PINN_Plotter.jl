module PINNPlotter

using Plots

export PINN_Plotter

"""
    PINN_Plotter(infused_matrix, data, ode_solution, labels, data_steps, extrapolation_tseps, xlabel, ylabel, title, filename)

Creates a comparative visualization plot showing PINN predictions, original data, and standard ODE solutions.

This function generates a multi-series plot that compares three different trajectories for each equation in the system:
1. Original/noisy data points (displayed as scatter points)
2. PINN-infused predictions (displayed as continuous lines)
3. Standard ODE solution without neural network correction (displayed as continuous lines)

The plot is particularly useful for evaluating PINN performance in extrapolation scenarios, allowing visual comparison
between the neural network-enhanced predictions and both the original data and baseline ODE solutions.

# Arguments
- `infused_matrix::AbstractMatrix{Float64}`: Matrix containing PINN predictions with shape `(samples, equations)`.
- `data::AbstractMatrix{Float64}`: Matrix containing original/reference data points with shape `(samples, equations)`.
- `ode_solution::AbstractMatrix{Float64}`: Matrix containing standard ODE solution without NN correction with shape `(samples, equations)`.
- `labels::Vector{String}`: Vector of labels for the plot legend. Length must be a multiple of the number of equations (typically 3 labels per equation: data, PINN, ODE).
- `data_steps::AbstractVector{Float64}`: Time points corresponding to the original data samples.
- `extrapolation_tseps::AbstractVector{Float64}`: Time points for the PINN predictions and ODE solutions.
- `xlabel::String`: Label for the x-axis.
- `ylabel::String`: Label for the y-axis.
- `title::String`: Title for the plot.
- `filename::String`: The file path where the plot will be saved (e.g., "comparison_plot.png").
"""
function PINN_Plotter(
    infused_matrix::AbstractMatrix{Float64},
    data::AbstractMatrix{Float64},
    ode_solution::AbstractMatrix{Float64},
    labels::Vector{String},
    data_steps::AbstractVector{Float64},
    extrapolation_tseps::AbstractVector{Float64},
    xlabel::String,
    ylabel::String,
    title::String,
    filename::String
)

num_of_equations = size(infused_matrix, 2)

if length(labels) % num_of_equations != 0
    error("Number of labels must be a multiple of number of equations")
end

plot(data_steps, data[:, 1], label=labels[1], seriestype=:scatter, markersize=2, markerstrokewidth=0, legend=:topright)
plot!(extrapolation_tseps, infused_matrix[:, 1], label=labels[2])
plot!(extrapolation_tseps, ode_solution[:, 1], label=labels[3])

for i in 2:num_of_equations
    plot!(data_steps, data[:,i], label=labels[i * 3 - 2], seriestype=:scatter, markersize=2, markerstrokewidth=0)
    plot!(extrapolation_tseps, infused_matrix[:, i], label=labels[i * 3 - 1])
    plot!(extrapolation_tseps, ode_solution[:, i], label=labels[i * 3])
end

xlabel!(xlabel)
ylabel!(ylabel)
title!(title)
savefig("$filename")
println("Plot saved as $filename")

end

end # module PINNPlotter