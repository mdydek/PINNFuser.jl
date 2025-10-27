using OrdinaryDiffEq
using Lux, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers
using DelimitedFiles, Base.Threads

include("simple_CVS_model.jl")
using .simpleModel

println("Number of threads: ", Threads.nthreads())

rng = StableRNG(5958)
tspan_global = (0.0, 10.0)
u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0]
params = [0.3, 0.45, 0.006, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]
Eshift, Eₘᵢₙ, τₑₛ, τₑₚ, Eₘₐₓ, Rmv, τ = 0.0, 0.03, 0.3, 0.45, 1.5, 0.006, 1.0

loaded_data = readdlm("src/data/original_extrapolation.txt")
original_data = Array{Float64}(loaded_data)

NN = Lux.Chain(
    Lux.Dense(7, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 7)
)
p_init, st = Lux.setup(rng, NN)
p_init = 0 * ComponentVector{Float64}(p_init)

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
    for i in 1:7
        du[i] = du_phys[i] * (1 + α * tanh(NN_output[i]))
    end
end

function predict(p, α, tspan, tsteps)
    prob = ODEProblem((du, u, p, t) -> NIK_PINN!(du, u, p, t, α), u0, tspan, p)
    sol = solve(prob, Vern7(), dtmax=1e-2, saveat=tsteps, reltol=1e-7, abstol=1e-4)
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
    return sum([mean(abs2, pressures_pred[:, i] .- pressures_true[:, i]) for i in 1:4])
end

function physics_loss(pred, p, α)
    loss = 0.0
    for (i, t) in enumerate(pred.t)
        u = pred[:, i]
        du = similar(u)
        NIK_PINN!(du, u, p, t, α)
        τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params
        loss += mean(abs2, du[5] - simpleModel.Valve(Zao, (du[1]-du[2]), u[1]-u[2]))
        loss += mean(abs2, du[6] - simpleModel.Valve(Rmv, (du[3]-du[1]), u[3]-u[1]))
        loss += mean(abs2, du[7] - (du[2]-du[3]) / Rs)
    end
    return loss / length(pred.t)
end

function total_loss(p, α, tspan, tsteps; λ=5)
    pred = predict(p, α, tspan, tsteps)
    return data_loss(pred, original_data_part) + λ * physics_loss(pred, p, α)
end

function make_callback(α; print_every=10)
    iter = Ref(0)
    return function (p, l)
        if iter[] % print_every == 0
            tid = Threads.threadid()
            println("    [α=$(α), thread $tid] iter=$(iter[])  |  loss=$(round(l, sigdigits=6))")
        end
        iter[] += 1
        return false
    end
end

function train_phase(tstart, tend, nsamples; λ_phys=5, α_values=[0.05, 0.1, 0.3, 0.8, 1.5], initial_weights=nothing, maxiters=150)
    println("\nStart: [$tstart, $tend], samples=$nsamples, λ_phys=$λ_phys")

    tspan = (0.0, tend)
    tsteps = range(tstart, tend, length=nsamples)
    p0 = isnothing(initial_weights) ? 0 * ComponentVector{Float64}(p_init) : initial_weights

    results = Vector{Tuple{Float64, Float64, ComponentVector{Float64}}}(undef, length(α_values))

    @threads for i in 1:length(α_values)
        α = α_values[i]
        println("==> Training started for α = $α (thread $(threadid()))")

        cb = make_callback(α, print_every=5)
        adtype = Optimization.AutoForwardDiff()
        optf = Optimization.OptimizationFunction((x,p) -> total_loss(x, α, tspan, tsteps; λ=λ_phys), adtype)
        optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p0))

        w = Optimization.solve(optprob, ADAM(0.01), callback=cb, maxiters=maxiters)
        optprob2 = Optimization.OptimizationProblem(optf, w.u)
        final_sol = Optimization.solve(optprob2, ADAM(0.001), callback=cb, maxiters=Int(maxiters ÷ 10))

        final_loss = total_loss(final_sol.u, α, tspan, tsteps; λ=λ_phys)
        results[i] = (α, final_loss, final_sol.u)
        println("==> α = $α finished, loss = $final_loss")
    end

    best_idx = argmin([r[2] for r in results])
    best_alpha, best_loss, best_weights = results[best_idx]

    println("Best α = $best_alpha, loss = $best_loss")
    return best_weights, best_alpha, best_loss
end

println("\nCURRICULUM LEARNING")

# PHASE 1
original_data_part = original_data[901:1200, :]
phase1_weights, α1, loss1 = train_phase(6.0, 8.0, 300, λ_phys=20)

# PHASE 2
original_data_part = original_data[901:1350, :]
phase2_weights, α2, loss2 = train_phase(6.0, 9.0, 450, λ_phys=27, initial_weights=phase1_weights)

# PHASE 3
original_data_part = original_data[901:1500, :]
phase3_weights, α3, loss3 = train_phase(6.0, 10.0, 600, λ_phys=40, initial_weights=phase2_weights)

println("\nCurriculum finished:")
println("Phase 1: α=$α1, loss=$loss1")
println("Phase 2: α=$α2, loss=$loss2")
println("Phase 3: α=$α3, loss=$loss3")

best_weights = phase3_weights
best_alpha = α3

tspan_final = (0.0, 60.0)
tsteps_final = range(0.0, 60.0, length=9000)
trained_prob = ODEProblem((du,u,p,t)->NIK_PINN!(du,u,p,t,best_alpha), u0, tspan_final, best_weights)
sol_final = solve(trained_prob, Vern7(), dtmax=1e-2, saveat=tsteps_final, reltol=1e-7, abstol=1e-4)
data_to_save = hcat(sol_final[1,:], sol_final[2,:], sol_final[3,:], sol_final[4,:],
                    sol_final[5,:], sol_final[6,:], sol_final[7,:])
writedlm("src/data/curriculum_result.txt", data_to_save)

println("Wynik zapisany w src/data/curriculum_result.txt")
