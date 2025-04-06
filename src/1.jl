using CellMLToolkit, ModelingToolkit, OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Optim, Measures, BenchmarkTools

rng = StableRNG(5958)

ml = CellModel("models/shi_hose_2009-a679cdc2e974/ModelMain.cellml")

tspan = (0.0, 20.0)
num_of_samples = 1000
x = LinRange(0.0, 20.0, num_of_samples)
prob = ODEProblem(ml, tspan)
main_sol = solve(prob, Tsit5(), saveat=x, reltol=1e-8, abstol=1e-8)
sys = ml.sys

# TODO
# 1, 2, 3, 4, 5, 6, 7
# sys.LV.Pi, sys.Sas.Pi, sys.Svn.Po, sys.LV.V, sys.LV.Qo, ???, sys.Svn.Qi
original_data = [main_sol[sys.LV.Pi],
    main_sol[sys.Sas.Pi],
    main_sol[sys.Svn.Po],
    main_sol[sys.LV.V],
    main_sol[sys.LV.Qo],
    ones(num_of_samples),
    main_sol[sys.Svn.Qi]]


function Valve(R, deltaP, open)
    dq = 0.0
    if (-open) < 0.0
        dq = deltaP / R
    else
        dq = 0.0
    end
    return dq

end

function ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    τₑₛ = τₑₛ * τ

    τₑₚ = τₑₚ * τ
    #τ = 4/3(τₑₛ+τₑₚ)
    tᵢ = rem(t + (1 - Eshift) * τ, τ)

    Eₚ = (tᵢ <= τₑₛ) * (1 - cos(tᵢ / τₑₛ * pi)) / 2 +
         (tᵢ > τₑₛ) * (tᵢ <= τₑₚ) * (1 + cos((tᵢ - τₑₛ) / (τₑₚ - τₑₛ) * pi)) / 2 +
         (tᵢ <= τₑₚ) * 0

    E = Eₘᵢₙ + (Eₘₐₓ - Eₘᵢₙ) * Eₚ

    return E
end

function DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)

    τₑₛ = τₑₛ * τ
    τₑₚ = τₑₚ * τ
    #τ = 4/3(τₑₛ+τₑₚ)
    tᵢ = rem(t + (1 - Eshift) * τ, τ)

    DEₚ = (tᵢ <= τₑₛ) * pi / τₑₛ * sin(tᵢ / τₑₛ * pi) / 2 +
          (tᵢ > τₑₛ) * (tᵢ <= τₑₚ) * pi / (τₑₚ - τₑₛ) * sin((τₑₛ - tᵢ) / (τₑₚ - τₑₛ) * pi) / 2
    (tᵢ <= τₑₚ) * 0
    DE = (Eₘₐₓ - Eₘᵢₙ) * DEₚ

    return DE
end


#Shi timing parameters
Eshift = 0.0
Eₘᵢₙ = 0.03
τₑₛ = 0.3
τₑₚ = 0.45
Eₘₐₓ = 1.5
Rmv = 0.006
τ = 1.0


function NIK!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
    τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = p
    # The differential equations
    du[1] = (Qmv - Qav) * ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) + pLV / ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) * DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    # 1 Left Ventricle
    du[2] = (Qav - Qs) / Csa #Systemic arteries     
    du[3] = (Qs - Qmv) / Csv # Venous
    du[4] = Qmv - Qav # LV volume
    du[5] = Valve(Zao, (du[1] - du[2]), u[1] - u[2])  # AV 
    du[6] = Valve(Rmv, (du[3] - du[1]), u[3] - u[1])  # MV
    du[7] = (du[2] - du[3]) / Rs # Systemic flow
end


function NIK_PINN!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
    τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params

    # Neural Network component (NN for correction)
    NN_output = NN(u, p, st)[1]

    # The differential equations with NN correction
    du[1] = (Qmv - Qav) * ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) +
            pLV / ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) *
            DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) +
            NN_output[1]
    du[2] = (Qav - Qs) / Csa + NN_output[2]  # Systemic arteries with NN correction
    du[3] = (Qs - Qmv) / Csv + NN_output[3]  # Venous with NN correction
    du[4] = Qmv - Qav + NN_output[4]  # LV volume
    du[5] = Valve(Zao, (du[1] - du[2]), u[1] - u[2]) + NN_output[5]  # AV Valve
    # du[6] = Valve(Rmv, (du[3] - du[1]), u[3] - u[1]) + NN_output[6]  # MV Valve
    # du[6] = Valve(Rmv, (du[3] - du[1]), u[3] - u[1])
    du[6] = 1
    du[7] = (du[2] - du[3]) / Rs + NN_output[7]  # Systemic flow
end


u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0]
params = [0.3, 0.45, 0.006, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]

# Simple model without NN
# prob = ODEProblem(NIK!, u0, tspan, params)
# @time simple_sol = solve(prob, Tsit5(), saveat = x, reltol = 1e-8, abstol = 1e-8)


NN = Lux.Chain(
    Lux.Dense(7, 32, elu),
    Lux.Dense(32, 32, elu),
    Lux.Dense(32, 7)
)

p, st = Lux.setup(rng, NN)
p = 0.5 * ComponentVector{Float64}(p)

prob_NN = ODEProblem(NIK_PINN!, u0, tspan, p)
s = solve(prob_NN, Vern7(), dtmax=1e-2, saveat=x, reltol=1e-7, abstol=1e-4)

function predict(p)
    temp_prob = remake(prob_NN, p=p)
    temp_sol = solve(temp_prob, Vern7(), dtmax=1e-2, saveat=x, reltol=1e-7, abstol=1e-4)
    return temp_sol
end

function loss(p)
    pred = predict(p)
    if size(pred) == size(main_sol)
        return mean(abs2, Array(pred) .- original_data)
    else
        return Inf, Array(pred) # Return infiinite loss if solution is unstable
    end
end

losses1_0 = Float64[]

callback = function (p, l)
    push!(losses1_0, l)
    if length(losses1_0) % 1 == 0
        println("Current loss after $(length(losses1_0)) iterations: $(losses1_0[end])")
    end
    return false
end

println("Hello, World!")

adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

# 1000 iterations using learning rate of 0.01
w1 = Optimization.solve(optprob, ADAM(0.01), callback=callback, maxiters=1000)

# 100 iteration using learning rate of 0.0001
optprob2 = Optimization.OptimizationProblem(optf, w1.u)
PINN_sol = Optimization.solve(optprob2, ADAM(0.0001), callback=callback, maxiters=100)


# Plot and MSE Error
# Najprościej sobie plotować w jupyter notebooku, ale fajnie żeby najpierw
# się udało trening doprowadzić do końca, a potem można sobie porobić wykresy
# i policzyć błąd MSE metody z PINN i bez PINN'a

# main_values = main_sol[sys.LV.Pi, 1000:2000]
# approx_values = PINN_sol[1, 1000:2000]
# PINN_MSE = mean((main_values .- approx_values) .^ 2)

# main_values = main_sol[sys.LV.Pi, 1000:2000]
# approx_values = simple_sol[1, 1000:2000]
# simple_MSE = mean((main_values .- approx_values) .^ 2)

# plot(simple_sol, idxs=[4], tspan = (10.0, 12.0))
# plot(PINN_sol, idxs=[4], tspan = (10.0, 12.0))
# plot(main_sol, idxs = [sys.LV.V], tspan = (10.0, 12.0))