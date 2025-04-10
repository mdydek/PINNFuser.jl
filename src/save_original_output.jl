using CellMLToolkit, ModelingToolkit, OrdinaryDiffEq
using DelimitedFiles, StableRNGs

# RNG do powtarzalności wyników
rng = StableRNG(5958)

# Wczytaj model CellML
ml = CellModel("zero_dimensial_modeling/shi_hose_2009-a679cdc2e974/ModelMain.cellml")

# Zakres czasowy i punkty czasowe
tspan = (0.0, 100.0)
num_of_samples = 300
tsteps = range(10.0, 12.0, length = num_of_samples)

# Rozwiązanie równania różniczkowego
prob = ODEProblem(ml, tspan)
main_sol = solve(prob, Tsit5(); saveat = tsteps, reltol = 1e-4, abstol = 1e-7, dtmax = 1e-2)

sys = ml.sys

# Wybierz zmienne do zapisania
data_to_save = hcat(
    main_sol[sys.LV.Pi],
    main_sol[sys.Sas.Pi],
    main_sol[sys.Svn.Po],
    main_sol[sys.LV.V],
    main_sol[sys.LV.Qo],
    ones(num_of_samples),
    main_sol[sys.Svn.Qi]
)

# Zapisz dane do pliku tekstowego
writedlm("original_data.txt", data_to_save)

println("Dane zapisane do pliku original_data.txt")
