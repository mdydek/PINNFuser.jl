using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Optim, Measures, BenchmarkTools
using DelimitedFiles, ForwardDiff

include("simple_CVS_model.jl")
using .simpleModel

rng = StableRNG(5958)

# Training range
tspan = (0.0, 7.0)
num_of_samples = 300
tsteps = range(5.0, 7.0, length = num_of_samples)

loaded_data = readdlm("data/original_data.txt")
original_data = Array{Float64}(loaded_data)

u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0]
params = [0.3, 0.45, 0.006, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]

# Shi timing parameters
Eshift = 0.0
Eₘᵢₙ = 0.03
τₑₛ = 0.3
τₑₚ = 0.45
Eₘₐₓ = 1.5
Rmv = 0.006
τ = 1.0   # cycle (1.0s)

# Elu activation function
elu(x) = x >= 0 ? x : exp(x) - 1

# --- Neural Network ---
NN = Lux.Chain(Lux.Dense(8, 16, tanh), Lux.Dense(16, 16, tanh), Lux.Dense(16, 7))


p, st = Lux.setup(rng, NN)
p = 0.5 * ComponentVector{Float64}(p)

# --- ODE z PINN ---
function NIK_PINN!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
    τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params

    # Punkt 1: dodanie cycle_phase do wejścia NN
    cycle_phase = mod(t, τ) / τ
    aug_u = vcat(u, cycle_phase)

    # Korekta NN (skalowana, żeby nie destabilizować okresowości)
    NN_output = 0.05 .* NN(aug_u, p, st)[1]

    du[1] =
        (Qmv - Qav) * ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) +
        pLV / ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) *
        DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) +
        NN_output[1]
    du[2] = (Qav - Qs) / Csa + NN_output[2]
    du[3] = (Qs - Qmv) / Csv + NN_output[3]
    du[4] = Qmv - Qav + NN_output[4]
    du[5] = Valve(Zao, (du[1] - du[2]), u[1] - u[2]) + NN_output[5]
    du[6] = Valve(Rmv, (du[3] - du[1]), u[3] - u[1]) + NN_output[6]
    du[7] = (du[2] - du[3]) / Rs + NN_output[7]
end

prob_NN = ODEProblem(NIK_PINN!, u0, tspan, p)

function predict(p)
    temp_prob = remake(prob_NN, p = p)
    solve(temp_prob, Vern7(), dtmax = 1e-2, saveat = tsteps, reltol = 1e-7, abstol = 1e-4)
end

# --- Loss function ---
function loss(p; alpha = 0.1, beta = 0.01)
    pred = predict(p)
    pred_vals = hcat(
        ForwardDiff.value(pred[1, :]),
        ForwardDiff.value(pred[2, :]),
        ForwardDiff.value(pred[3, :]),
        ForwardDiff.value(pred[4, :]),
        ForwardDiff.value(pred[5, :]),
        ForwardDiff.value(pred[6, :]),
        ForwardDiff.value(pred[7, :]),
    )

    # Data loss (presja)
    loss_data = mean(abs2, pred_vals[:, 1:3] .- original_data[:, 1:3])

    # Physics loss (flows)
    dt = tsteps[2] - tsteps[1]
    dpred_dt = diff(pred_vals, dims = 1) ./ dt
    τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params
    phys_loss1 =
        mean(abs2, dpred_dt[:, 5] .- (pred_vals[2:end, 1] .- pred_vals[2:end, 2]) ./ Zao)
    phys_loss2 =
        mean(abs2, dpred_dt[:, 6] .- (pred_vals[2:end, 3] .- pred_vals[2:end, 1]) ./ Rmv)
    phys_loss3 =
        mean(abs2, dpred_dt[:, 7] .- (pred_vals[2:end, 2] .- pred_vals[2:end, 3]) ./ Rs)
    loss_physics = phys_loss1 + phys_loss2 + phys_loss3

    # Punkt 6: Periodic loss (wymuszenie okresowości co τ)
    Np = Int(round(τ / dt))  # liczba kroków odpowiadająca jednemu cyklowi
    idx = 1:(size(pred_vals, 1)-Np)
    periodic_loss = mean(abs2, pred_vals[idx, :] .- pred_vals[idx .+ Np, :])

    return loss_data + alpha * loss_physics + beta * periodic_loss
end

# --- Trening ---
losses = Float64[]
callback = function (p, l)
    push!(losses, l)
    println("Iter $(length(losses)) | Loss = $(losses[end])")
    return false
end

adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

w1 = Optimization.solve(optprob, ADAM(0.01), callback = callback, maxiters = 1000)

# Extrapolation
tspan2 = (0.0, 20.0)
num_of_samples = 3000
tsteps = range(0.0, 20.0, length = num_of_samples)
trained_NN = ODEProblem(NIK_PINN!, u0, tspan2, w1.u)
s = solve(trained_NN, Vern7(), dtmax = 1e-2, saveat = tsteps, reltol = 1e-7, abstol = 1e-4)

data_to_save = hcat(s[1, :], s[2, :], s[3, :], s[4, :], s[5, :], s[6, :], s[7, :])
writedlm("data/pinn_extrapolation_with_phase_periodic.txt", data_to_save)
println("Saved extrapolation with phase+periodic_loss.")
