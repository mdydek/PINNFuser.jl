using OrdinaryDiffEq
using Lux, Plots, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers
using ForwardDiff

rng = StableRNG(5958)

# Initial condition
u0 = [1.0, 0.0]  # y(0) = 1, dy/dt(0) = 0

# Training range
tspan = (0.0, 10.0)
num_of_samples = 500
tsteps = range(tspan[1], tspan[2], length=num_of_samples)

# Analytical solution
y_analytical(t) = exp(-t)

# Evaluate at collocation points
y_ground_truth = y_analytical.(tsteps)


NN = Lux.Chain(
    Lux.Dense(1, 10, elu),
    Lux.Dense(10, 10, elu),
    Lux.Dense(10, 10, elu),
    Lux.Dense(10, 1)
)

p, st = Lux.setup(rng, NN)
p = 0 * ComponentVector{Float64}(p)

function simple_PINN!(du, u, p, t)
    y = u[1]
    NN_output = NN([y], p, st)[1][1]

    # ODE: dy/dt = -y + NN_correction
    du[1] = -y + NN_output
end

prob_NN = ODEProblem(simple_PINN!, u0, tspan, p)


# --- Prediction function ---
function predict(p)
    temp_prob = remake(prob_NN, p=p)
    temp_sol = solve(temp_prob, Vern7(), dtmax=1e-2, saveat=tsteps, reltol=1e-6, abstol=1e-6)
    return temp_sol
end

# --- Data loss ---
function data_loss(pred, y_true)
    y_pred = pred[1, :]
    return mean(abs2, y_pred .- y_true)
end

# --- Physics loss ---
function physics_loss(pred, p)
    loss = 0.0
    for (i, t) in enumerate(tsteps)
        u = [pred[1, i]]
        du = similar(u)
        simple_PINN!(du, u, p, t)
        # Residual: dy/dt + y = du + y
        loss += abs2(du[1] + u[1])
    end
    return loss / length(tsteps)
end

# --- Total loss ---
function total_loss(p, λ=1.0)
    pred = predict(p)
    L_data = data_loss(pred, y_ground_truth)
    L_phys = physics_loss(pred, p)
    return L_data + λ * L_phys
end


losses = Float64[]

callback = function (p, l)
    push!(losses, l)
    println("Current loss after $(length(losses)) iterations: $(losses[end])")
    return false
end

println("Hello, PINN!")

adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, p) -> total_loss(x, 1.0), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

w1 = Optimization.solve(optprob, ADAM(0.01), callback=callback, maxiters=20)
optprob1 = Optimization.OptimizationProblem(optf, w1.u)
PINN_sol = Optimization.solve(optprob1, ADAM(0.001), callback=callback, maxiters=20)

prediction = predict(PINN_sol.u)

# --- Comparison plot: Analytical vs PINN ---
y_pinn = prediction[1, :] # PINN prediction
y_true = y_analytical.(tsteps) # Analytical solution

plot(tsteps, y_true, label="Analytical y(t) = exp(-t)", lw=3)
plot!(tsteps, y_pinn, label="PINN Prediction", lw=3, ls=:dash)
xlabel!("t")
ylabel!("y(t)")
title!("Comparison: Analytical vs PINN")
savefig("exponential_plots/exponential_training.png")
println("Plot saved as pinn_vs_analytical.png")

# --- Loss function plot ---
plot(1:length(losses), losses, label="Total Loss", lw=2, yscale=:log10, grid=true)
xlabel!("Iteration")
ylabel!("Loss")
title!("Training Loss Over Iterations")
savefig("exponential_plots/exponential_loss.png")
println("Training loss plot saved as pinn_training_loss.png")


# --- Extrapolation range ---
tspan_extrap = (0.0, 50.0)
num_extrap_points = 2000
tsteps_extrap = range(tspan_extrap[1], tspan_extrap[2], length=num_extrap_points)

# Solve PINN on extrapolation range
prob_extrap = remake(prob_NN, p=PINN_sol.u, tspan=tspan_extrap)
sol_extrap = solve(prob_extrap, Vern7(), dtmax=1e-2, saveat=tsteps_extrap, reltol=1e-6, abstol=1e-6)

y_pinn_extrap = sol_extrap[1, :]
y_true_extrap = y_analytical.(tsteps_extrap)

# --- Plot extrapolation ---
plot(tsteps_extrap, y_true_extrap, label="Analytical y(t) = exp(-t)", lw=3)
plot!(tsteps_extrap, y_pinn_extrap, label="PINN Prediction", lw=3, ls=:dash)
xlabel!("t")
ylabel!("y(t)")
title!("Extrapolation: Analytical vs PINN")
savefig("exponential_plots/exponential_extrapolation.png")
println("Extrapolation plot saved as pinn_extrapolation.png")