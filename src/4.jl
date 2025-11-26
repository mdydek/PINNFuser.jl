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
τ = 1.0

# ADDED periodicity to NN outputs infusion
period = 1.0   # approximate cardiac cycle period (seconds)

function periodic_correction(nn_out, t)
    return nn_out * sin(2π * t / period)
end

NN = Lux.Chain(
    Lux.Dense(7, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh), #CHANGE
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 7),
)

p, st = Lux.setup(rng, NN)
p = 0 * ComponentVector{Float64}(p)

function NIK_PINN!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
    τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params

    NN_output = NN(u, p, st)[1]

    elastance = simpleModel.ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    delastance = simpleModel.DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)

    alfa = 0.1
    du[1] =
        ((Qmv - Qav) * elastance + pLV / elastance * delastance) *
        (1 + alfa * periodic_correction(NN_output[1], t))

    du[2] = ((Qav - Qs) / Csa) * (1 + alfa * periodic_correction(NN_output[2], t))     # Systemic arteries     
    du[3] = ((Qs - Qmv) / Csv) * (1 + alfa * periodic_correction(NN_output[3], t))       # Venous
    du[4] = (Qmv - Qav) * (1 + alfa * periodic_correction(NN_output[4], t))               # LV volume
    du[5] =
        (simpleModel.Valve(Zao, (du[1] - du[2]), u[1] - u[2])) *
        (1 + alfa * periodic_correction(NN_output[5], t))  # AV 
    du[6] =
        (simpleModel.Valve(Rmv, (du[3] - du[1]), u[3] - u[1])) *
        (1 + alfa * periodic_correction(NN_output[6], t))  # MV
    du[7] = ((du[2] - du[3]) / Rs) * (1 + alfa * periodic_correction(NN_output[7], t))   # Systemic flow
end

prob_NN = ODEProblem(NIK_PINN!, u0, tspan, p)

s = solve(prob_NN, Vern7(), dtmax = 1e-2, saveat = tsteps, reltol = 1e-7, abstol = 1e-4)

function predict(p)
    temp_prob = remake(prob_NN, p = p)
    temp_sol = solve(
        temp_prob,
        Vern7(),
        dtmax = 1e-2,
        saveat = tsteps,
        reltol = 1e-7,
        abstol = 1e-4,
    )
    return temp_sol
end

function split_pred(pred)
    # pLV, psa, psv, Vlv
    pressures = hcat(pred[1, :], pred[2, :], pred[3, :], pred[4, :])

    # Qav, Qmv, Qs
    flows = hcat(pred[5, :], pred[6, :], pred[7, :])

    return pressures, flows
end

function data_loss(pred, original)
    pressures_pred, _ = split_pred(pred)
    pressures_true = original[:, 1:4]  # columns 1-4
    lossplv = mean(abs2, (pressures_pred[:, 1] - pressures_true[:, 1]))
    losspsa = mean(abs2, (pressures_pred[:, 2] - pressures_true[:, 2]))
    losspv = mean(abs2, (pressures_pred[:, 3] - pressures_true[:, 3]))
    lossvlv = mean(abs2, (pressures_pred[:, 4] - pressures_true[:, 4]))
    return lossplv + losspsa + losspv + lossvlv
end

function physics_loss(pred, p)
    loss = 0.0
    for (i, t) in enumerate(tsteps)
        u = pred[:, i]
        du = similar(u)
        NIK_PINN!(du, u, p, t)
        τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params
        loss += mean(abs2, du[5] - simpleModel.Valve(Zao, (du[1] - du[2]), u[1] - u[2]))
        loss += mean(abs2, du[6] - simpleModel.Valve(Rmv, (du[3] - du[1]), u[3] - u[1]))
        loss += mean(abs2, du[7] - (du[2] - du[3]) / Rs)
    end
    return loss / length(tsteps)
end

function total_loss(p, λ = 5)
    pred = predict(p)
    L_data = data_loss(pred, original_data)
    L_phys = physics_loss(pred, p)
    return L_data + λ * L_phys
end

losses = Float64[]

callback = function (p, l)
    push!(losses, l)
    println("Current loss after $(length(losses)) iterations: $(losses[end])")
    return false
end

println("Hello, Physics-Informed PINN!")

adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, p) -> total_loss(x, 1.0), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

w1 = Optimization.solve(optprob, ADAM(0.01), callback = callback, maxiters = 100)
optprob1 = Optimization.OptimizationProblem(optf, w1.u)
PINN_sol = Optimization.solve(optprob1, ADAM(0.001), callback = callback, maxiters = 25)

prediction = predict(PINN_sol.u)

tspan2 = (0.0, 20.0)
num_of_samples = 3000
tsteps = range(0.0, 20.0, length = num_of_samples)

p_trained = PINN_sol.u
trained_NN = ODEProblem(NIK_PINN!, u0, tspan2, p_trained)
s = solve(trained_NN, Vern7(), dtmax = 1e-2, saveat = tsteps, reltol = 1e-7, abstol = 1e-4)

data_to_save = hcat(s[1, :], s[2, :], s[3, :], s[4, :], s[5, :], s[6, :], s[7, :])
writedlm("data/4_extrapolation.txt", data_to_save)
println("Extrapolation saved to data/physics_pinn_extrapolation_4.txt")
