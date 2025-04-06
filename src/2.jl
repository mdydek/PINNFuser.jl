using CellMLToolkit, ModelingToolkit, OrdinaryDiffEq
using Plots, ComponentArrays
using Flux, Flux.Optimise, Zygote
using Optim, Measures, BenchmarkTools

ml = CellModel("zero_dimensial_modeling/shi_hose_2009-a679cdc2e974/ModelMain.cellml")

tspan = (0.0, 20.0)
num_of_samples = 1000
x = LinRange(0.0, 20.0, num_of_samples)
prob = ODEProblem(ml, tspan)
main_sol = solve(prob, Tsit5(), saveat = x, reltol=1e-8, abstol=1e-8)
sys=ml.sys

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
        dq =  deltaP/R
    else
        dq = 0.0
    end
    return dq

end

function ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    τₑₛ = τₑₛ*τ
    
    τₑₚ = τₑₚ*τ
    #τ = 4/3(τₑₛ+τₑₚ)
    tᵢ = rem(t + (1 - Eshift) * τ, τ)

    Eₚ = (tᵢ <= τₑₛ) * (1 - cos(tᵢ / τₑₛ * pi)) / 2 +
         (tᵢ > τₑₛ) * (tᵢ <= τₑₚ) * (1 + cos((tᵢ - τₑₛ) / (τₑₚ - τₑₛ) * pi)) / 2 +
         (tᵢ <= τₑₚ) * 0

    E = Eₘᵢₙ + (Eₘₐₓ - Eₘᵢₙ) * Eₚ

    return E
end

function DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)

    τₑₛ = τₑₛ*τ
    τₑₚ = τₑₚ*τ
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
    du[2] = (Qav - Qs ) / Csa #Systemic arteries     
    du[3] = (Qs - Qmv) / Csv # Venous
    du[4] = Qmv - Qav # LV volume
    du[5]    = Valve(Zao, (du[1] - du[2]), u[1] - u[2])  # AV 
    du[6]   = Valve(Rmv, (du[3] - du[1]), u[3] - u[1])  # MV
    du[7]     = (du[2] - du[3]) / Rs # Systemic flow
end


u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0]
params = [0.3, 0.45, 0.006, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]

# Simple model without NN
# prob = ODEProblem(NIK!, u0, tspan, params)
# @time simple_sol = solve(prob, Tsit5(), saveat = x, reltol = 1e-8, abstol = 1e-8)

# Sieć
NN_model = Chain(
    Dense(7, 32, relu),
    Dense(32, 32, relu),
    Dense(32, 7)
)

# Wrapper na NN
# function PINN_NN(u, p)
#     input = vcat(u, p)
#     return NN_model(input)
# end

# Funkcja ODE z NN (closure z dostępem do NN_model)
function NIK_PINN!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u 
    τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = p

    println("HERE")

    NN_output = NN_model(u)

    println("STILL HERE")

    du[1] = (Qmv - Qav) * ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) +
            pLV / ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) *
            DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) +
            NN_output[1]
    du[2] = (Qav - Qs) / Csa + NN_output[2]
    du[3] = (Qs - Qmv) / Csv + NN_output[3]
    du[4] = Qmv - Qav + NN_output[4]
    du[5] = Valve(Zao, (du[1] - du[2]), u[1] - u[2]) + NN_output[5]
    du[6] = 1.0 + NN_output[6]
    du[7] = (du[2] - du[3]) / Rs + NN_output[7]
end

# Problem z PINN
prob_NN = ODEProblem(NIK_PINN!, u0, tspan, params)

# Funkcja błędu
function loss()
    sol = solve(prob_NN, Tsit5(), saveat=x, reltol=1e-6, abstol=1e-6)
    pred = Array(sol)
    return mean(pred .- original_data .^ 2)
end

# Trening
ps = Flux.trainable(NN_model)
opt = ADAM(0.001)

println("TRAINING START")

for epoch in 1:200
    grads = Zygote.gradient(() -> loss(), ps)
    Flux.Optimise.update!(opt, ps, grads)

    if epoch % 10 == 0
        @show epoch, loss()
    end
end