using CellMLToolkit, ModelingToolkit, OrdinaryDiffEq
using DelimitedFiles, StableRNGs

rng = StableRNG(5958)

ml = CellModel("../models/shi_hose_2009-a679cdc2e974/ModelMain.cellml")

# Training range
tspan = (0.0, 20.0)
# num_of_samples = 30
num_of_samples = 1000
tsteps = range(0.3, 10.3, length = num_of_samples)

prob = ODEProblem(ml, tspan)
main_sol = solve(prob, Tsit5(); saveat = tsteps, reltol = 1e-4, abstol = 1e-7, dtmax = 1e-2)

sys = ml.sys

data_to_save = hcat(
    main_sol[sys.LV.Pi],
    main_sol[sys.Sas.Pi],
    main_sol[sys.Svn.Po],
    main_sol[sys.LV.V],
    main_sol[sys.LV.Qo],
    main_sol[sys.LA.Qo],
    main_sol[sys.Svn.Qi]
)

# Zapisz dane do pliku tekstowego
# writedlm("data/original_data.txt", data_to_save)
writedlm("data/original_extrapolation.txt", data_to_save)

# println("Dane zapisane do pliku original_data.txt")
println("Dane zapisane do pliku original_extrapolation.txt")
