using OrdinaryDiffEq
using Lux, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers
using DelimitedFiles, Base.Threads

include("simple_CVS_model.jl")
using .simpleModel

println("Liczba dostępnych wątków: ", Threads.nthreads())

rng = StableRNG(5958)
tspan = (0.0, 7.0)
num_of_samples = 300
tsteps = range(5.0, 7.0, length = num_of_samples)

loaded_data = readdlm("src/data/original_data.txt")
original_data = Array{Float64}(loaded_data)

u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0]
params = [0.3, 0.45, 0.006, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]

Eshift, Eₘᵢₙ, τₑₛ, τₑₚ, Eₘₐₓ, Rmv, τ = 0.0, 0.03, 0.3, 0.45, 1.5, 0.006, 1.0

NN = Lux.Chain(
    Lux.Dense(7, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 7),
)
p, st = Lux.setup(rng, NN)
p = 0 * ComponentVector{Float64}(p)

function NIK_PINN!(du, u, p, t, α)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
    τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params

    du_phys = similar(u)
    elastance = simpleModel.ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    delastance = simpleModel.DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)

    du_phys[1] = ((Qmv - Qav) * elastance + pLV / elastance * delastance)
    du_phys[2] = ((Qav - Qs) / Csa)
    du_phys[3] = ((Qs - Qmv) / Csv)
    du_phys[4] = (Qmv - Qav)
    du_phys[5] = simpleModel.Valve(Zao, (du_phys[1] - du_phys[2]), u[1] - u[2])
    du_phys[6] = simpleModel.Valve(Rmv, (du_phys[3] - du_phys[1]), u[3] - u[1])
    du_phys[7] = (du_phys[2] - du_phys[3]) / Rs

    NN_output = NN(u, p, st)[1]

    for i = 1:7
        du[i] = du_phys[i] * (1 + α * tanh(NN_output[i]))
    end
end
function predict(p, α)
    prob = ODEProblem((du, u, p, t) -> NIK_PINN!(du, u, p, t, α), u0, tspan, p)
    sol = solve(prob, Vern7(), dtmax = 1e-2, saveat = tsteps, reltol = 1e-7, abstol = 1e-4)
    return sol
end

function split_pred(pred)
    pressures = hcat(pred[1, :], pred[2, :], pred[3, :], pred[4, :])
    flows = hcat(pred[5, :], pred[6, :], pred[7, :])
    return pressures, flows
end

function data_loss(pred, original)
    pressures_pred, _ = split_pred(pred)
    pressures_true = original[:, 1:4]
    return sum([mean(abs2, pressures_pred[:, i] .- pressures_true[:, i]) for i = 1:4])
end

function physics_loss(pred, p, α)
    loss = 0.0
    for (i, t) in enumerate(tsteps)
        u = pred[:, i]
        du_PINN = similar(u)
        du_phys = similar(u)
        NIK_PINN!(du_PINN, u, p, t, α)

        pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
        τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params
        elastance = simpleModel.ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
        delastance = simpleModel.DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
        du_phys[1] = ((Qmv - Qav) * elastance + pLV / elastance * delastance)
        du_phys[2] = ((Qav - Qs) / Csa)
        du_phys[3] = ((Qs - Qmv) / Csv)
        du_phys[4] = (Qmv - Qav)
        du_phys[5] = simpleModel.Valve(Zao, (du_phys[1] - du_phys[2]), u[1] - u[2])
        du_phys[6] = simpleModel.Valve(Rmv, (du_phys[3] - du_phys[1]), u[3] - u[1])
        du_phys[7] = (du_phys[2] - du_phys[3]) / Rs

        loss += sum(abs2, du_PINN .- du_phys)
    end
    return loss / (length(tsteps) * 7)
end

function total_loss(p, α; λ = 40)
    pred = predict(p, α)
    return data_loss(pred, original_data) + λ * physics_loss(pred, p, α)
end

function make_callback(α; print_every = 10)
    iter = Ref(0)
    return function (p, l)
        if iter[] % print_every == 0
            tid = Threads.threadid()
            println(
                "    [α=$(α), wątek $tid] iter=$(iter[])  |  loss=$(round(l, sigdigits=6))",
            )
        end
        iter[] += 1
        return false
    end
end

alphas = [0.1, 0.3, 0.8]
results = Vector{Tuple{Float64,Float64,ComponentVector{Float64}}}(undef, length(alphas))

@threads for i = 1:length(alphas)
    α = alphas[i]
    println("==> Start treningu dla α = $α (wątek $(threadid()))")

    cb = make_callback(α, print_every = 3)
    adtype = Optimization.AutoForwardDiff()
    optf = Optimization.OptimizationFunction((x, p) -> total_loss(x, α), adtype)
    optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

    w = Optimization.solve(optprob, ADAM(0.01), callback = cb, maxiters = 300)
    optprob2 = Optimization.OptimizationProblem(optf, w.u)
    final_sol = Optimization.solve(optprob2, ADAM(0.001), callback = cb, maxiters = 20)

    final_loss = total_loss(final_sol.u, α)
    results[i] = (α, final_loss, final_sol.u)

    println("==> α = $α zakończony, loss = $final_loss")
end

best_idx = argmin([r[2] for r in results])
best_alpha, best_loss, best_params = results[best_idx]
println("\nNajlepszy α = $best_alpha z loss = $best_loss")

PINN_sol_u = best_params

tspan2 = (0.0, 60.0)
tsteps2 = range(0.0, 60.0, length = 9000)
trained_NN =
    ODEProblem((du, u, p, t) -> NIK_PINN!(du, u, p, t, best_alpha), u0, tspan2, PINN_sol_u)
s = solve(trained_NN, Vern7(), dtmax = 1e-2, saveat = tsteps2, reltol = 1e-7, abstol = 1e-4)
data_to_save = hcat(s[1, :], s[2, :], s[3, :], s[4, :], s[5, :], s[6, :], s[7, :])
writedlm("src/data/5_full_physics.txt", data_to_save)

println("Zapisano dane dla najlepszej α = $best_alpha")
