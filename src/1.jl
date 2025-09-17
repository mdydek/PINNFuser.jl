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

# Elu activation function
function elu(x)
    if x >= 0
        return x
    else
        return exp.(x) .- 1
    end
end

NN = Lux.Chain(
    Lux.Dense(7, 10, elu),
    Lux.Dense(10, 10, elu),
    Lux.Dense(10, 10, elu),
    Lux.Dense(10, 7)
)

p, st = Lux.setup(rng, NN)
p = 0.5*ComponentVector{Float64}(p)

function NIK_PINN!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
    τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params
    
    # Neural Network component
    NN_output = NN(u, p, st)[1]

    # Differential equations with NN correction
    du[1] = (Qmv - Qav) * ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) +
            pLV / ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) * 
            DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) + 
            NN_output[1]

    du[2] = (Qav - Qs ) / Csa + NN_output[2]       # Systemic arteries     
    du[3] = (Qs - Qmv) / Csv + NN_output[3]        # Venous
    du[4] = Qmv - Qav + NN_output[4]               # LV volume
    du[5] = Valve(Zao, (du[1] - du[2]), u[1] - u[2]) + NN_output[5]  # AV 
    du[6] = Valve(Rmv, (du[3] - du[1]), u[3] - u[1]) + NN_output[6]  # MV
    du[7] = (du[2] - du[3]) / Rs + NN_output[7]   # Systemic flow
end

prob_NN = ODEProblem(NIK_PINN!, u0, tspan, p)

s = solve(prob_NN, Vern7(), dtmax=1e-2, saveat=tsteps, reltol=1e-7, abstol=1e-4)

function predict(p)
    temp_prob = remake(prob_NN, p=p)
    temp_sol = solve(temp_prob, Vern7(), dtmax=1e-2, saveat=tsteps, reltol=1e-7, abstol=1e-4)
    return temp_sol
end

# --- Split predictions into pressures and flows ---
function split_pred(pred)
    # Pressures: pLV, psa, psv
    pressures = hcat(pred[1, :], pred[2, :], pred[3, :])
    
    # Flows / volumes: Qav, Qmv, Qs, Vlv
    flows = hcat(pred[4, :], pred[5, :], pred[6, :], pred[7, :])
    
    return pressures, flows
end

# --- Data loss on pressures ---
function data_loss(pred, original)
    pressures_pred, _ = split_pred(pred)
    pressures_true = original[:, 1:3]  # columns 1-3
    return mean(abs2, pressures_pred - pressures_true)
end

# --- Physics loss on flows/volumes ---
function physics_loss(pred, p)
    loss = 0.0
    for (i, t) in enumerate(tsteps)
        u = pred[:, i]
        du = similar(u)
        NIK_PINN!(du, u, p, t)
        
        if i < length(tsteps)
            dt = tsteps[i+1] - t
            du_fd = (pred[:, i+1] - u)/dt
            # Only apply to flows/volumes: indices 4:7
            loss += mean(abs2, du[4:7] - du_fd[4:7])
        end
    end
    return loss / length(tsteps)
end

# --- Total loss ---
function total_loss(p, λ=1.0)
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

# --- Optimization setup ---
adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, p) -> total_loss(x, 1.0), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

# Training: 1000 iterations with decreasing learning rate
w1 = Optimization.solve(optprob, ADAM(0.01), callback=callback, maxiters=900)
optprob1 = Optimization.OptimizationProblem(optf, w1.u)
PINN_sol = Optimization.solve(optprob1, ADAM(0.0001), callback=callback, maxiters=100)

prediction = predict(PINN_sol.u)

# --- Save training range data ---
data_to_save = hcat(prediction[1, :], prediction[2, :], prediction[3, :],
                    prediction[4, :], prediction[5, :], prediction[6, :], prediction[7, :])
writedlm("data/baseline_pinn_data.txt", data_to_save)
println("Data from training range saved to baseline_pinn_data.txt")

# --- Test range (extrapolation) ---
tspan2 = (0.0, 20.0)
num_of_samples = 3000
tsteps = range(0.0, 20.0, length=num_of_samples)

p_trained = PINN_sol.u
trained_NN = ODEProblem(NIK_PINN!, u0, tspan2, p_trained)
s = solve(trained_NN, Vern7(), dtmax=1e-2, saveat=tsteps, reltol=1e-7, abstol=1e-4)

data_to_save = hcat(s[1, :], s[2, :], s[3, :], s[4, :], s[5, :], s[6, :], s[7, :])
writedlm("data/baseline_pinn_extrapolation.txt", data_to_save)
println("PINN extrapolation saved to baseline_pinn_extrapolation.txt")