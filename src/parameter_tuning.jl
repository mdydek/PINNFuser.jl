using OrdinaryDiffEq
using Lux, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers
using DelimitedFiles, Base.Threads
using SciMLSensitivity

include("simple_CVS_model.jl")
using .simpleModel

println("Liczba dostępnych wątków: ", Threads.nthreads())

rng = StableRNG(5958)
tspan = (0.0, 7.0)
num_of_samples = 300
tsteps = range(5.0, 7.0, length = num_of_samples)

loaded_data = readdlm("src/data/original_data.txt")
original_data = Array{Float64}(loaded_data)
original_data = original_data[1:num_of_samples, :]

u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0]

Eshift, τₑₛ, τₑₚ, τ = 0.0, 0.3, 0.45, 1.0
Zao, Csa, Csv = 0.033, 1.11, 1.13

NN = Lux.Chain(
    Lux.Dense(7, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 4),  # [Eₘₐₓ, Eₘᵢₙ, Rs, Rmv]
)
p, st = Lux.setup(rng, NN)
p = 0 * ComponentVector{Float64}(p)

function param_from_NN(u, p, st)
    y, _ = NN(u, p, st)
    Eₘₐₓ = 0.5 + abs(y[1]) * 2.0     # ~0.5–2.5
    Eₘᵢₙ = 0.01 + abs(y[2]) * 0.05   # ~0.01–0.06
    Rs = 5.0 + abs(y[3]) * 15.0    # ~5–20
    Rmv = 0.002 + abs(y[4]) * 0.01  # ~0.002–0.012
    return Eₘₐₓ, Eₘᵢₙ, Rs, Rmv
end

function model_ODE!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u

    Eₘₐₓ, Eₘᵢₙ, Rs, Rmv = param_from_NN(u, p, st)

    elastance = simpleModel.ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    delastance = simpleModel.DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)

    du[1] = ((Qmv - Qav) * elastance + pLV / elastance * delastance)
    du[2] = ((Qav - Qs) / Csa)
    du[3] = ((Qs - Qmv) / Csv)
    du[4] = (Qmv - Qav)
    du[5] = simpleModel.Valve(Zao, (du[1] - du[2]), pLV - psa)
    du[6] = simpleModel.Valve(Rmv, (du[3] - du[1]), psv - pLV)
    du[7] = (du[2] - du[3]) / Rs
end

function predict(p)
    prob = ODEProblem(model_ODE!, u0, tspan, p)
    sol = solve(prob, Vern7(), dtmax = 1e-2, saveat = tsteps, reltol = 1e-7, abstol = 1e-4)
    return sol
end

function total_loss(p)
    sol = predict(p)
    pred = Array(sol)
    pressures_pred = pred[1:4, :]'
    pressures_true = original_data[:, 1:4]
    return mean(abs2, pressures_pred .- pressures_true)
end

function make_callback()
    iter = Ref(0)
    return function (p, l)
        if iter[] % 10 == 0
            println("Iter $(iter[]): loss = $(round(l, sigdigits=6))")
        end
        iter[] += 1
        return false
    end
end

adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, p) -> total_loss(x), adtype)

optprob = Optimization.OptimizationProblem(optf, p)
cb = make_callback()

res = Optimization.solve(optprob, ADAM(0.001), callback = cb, maxiters = 300)
println("\nTrening zakończony: final loss = ", total_loss(res.u))

println("\nTrening zakończony: final loss = ", total_loss(res.u))

println("\nPrzykładowe parametry wygenerowane przez NN po treningu:")
for i = 1:5
    u_sample = original_data[i, 1:7]
    Eₘₐₓ, Eₘᵢₙ, Rs, Rmv = param_from_NN(u_sample, res.u, st)
    println(
        "Stan ",
        i,
        ": Emax=$(round(Eₘₐₓ, digits=3)), Emin=$(round(Eₘᵢₙ, digits=3)), Rs=$(round(Rs, digits=3)), Rmv=$(round(Rmv, digits=5))",
    )
end

tspan2 = (0.0, 60.0)
tsteps2 = range(0.0, 60.0, length = 9000)
trained_prob = ODEProblem(model_ODE!, u0, tspan2, res.u)
sol_trained = solve(
    trained_prob,
    Vern7(),
    dtmax = 1e-2,
    saveat = tsteps2,
    reltol = 1e-7,
    abstol = 1e-4,
)
data_to_save = Array(sol_trained)'
writedlm("src/data/NN_tuned_parameters_simulation.txt", data_to_save)

println("Zapisano dane z wytrenowanymi parametrami NN.")
