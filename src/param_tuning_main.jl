using OrdinaryDiffEq, OptimizationOptimisers, Lux
using StableRNGs, DelimitedFiles, ComponentArrays

include("lib/PINN_Parameter_Tuner.jl")
using .PINNParamTuner

include("simple_CVS_model.jl")
using .simpleModel

data = readdlm("src/data/original_data.txt")
data = Array{Float64}(data)[1201:1500, :]

tspan = (0.0, 7.0)
num_samples = size(data, 1)
tsteps = range(5.0, 7.0, length=num_samples)

u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0]
params = [0.3, 0.45, 0.006, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]

function model_CVS!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
    τes, τep, Rmv, Zao, Rs, Csa, Csv, Emax, Emin = p

    τ = 1.0
    Eshift = 0.0

    elastance = simpleModel.ShiElastance(t, Emin, Emax, τ, τes, τep, Eshift)
    delastance = simpleModel.DShiElastance(t, Emin, Emax, τ, τes, τep, Eshift)

    du[1] = ((Qmv - Qav) * elastance + pLV / elastance * delastance)
    du[2] = ((Qav - Qs) / Csa)
    du[3] = ((Qs - Qmv) / Csv)
    du[4] = (Qmv - Qav)
    du[5] = simpleModel.Valve(Zao, (du[1] - du[2]), pLV - psa)
    du[6] = simpleModel.Valve(Rmv, (du[3] - du[1]), psv - pLV)
    du[7] = (du[2] - du[3]) / Rs
end

tune_params = [3, 5, 8, 9]    # Rmv, Rs, Emax, Emin

trained_nn_params, st, tuned_values =
    PINN_Parameter_Tuner(
        model_CVS!,
        u0,
        tspan,
        data;
        initial_params = params,
        tune_params = tune_params,
        range_fraction = 0.6,
        learning_rate = 0.001,
        iters = 3,
        rng = StableRNG(5958),
    )

println("\nTuned parameter samples:")
for (i, v) in enumerate(tuned_values)
    println("sample $i → ", v)
end

# Zapis do pliku
writedlm("src/data/New_params_values.txt", tuned_values)
println("Tuned parameters saved to src/data/New_params_values.txt")


tsim = (0.0, 60.0)
tsteps2 = range(0, 60, length=9000)

prob_final = ODEProblem(
    (du,u,p,t)->model_CVS!(du,u,vcat(tuned_values),t),
    u0, tsim, nothing
)

sol = solve(prob_final, Vern7(), dtmax=1e-2, saveat=tsteps2)

writedlm("src/data/Lib_tuned.txt", Array(sol)')
println("Simulation saved.")
