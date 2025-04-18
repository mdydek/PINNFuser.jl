using OrdinaryDiffEq
using Lux, Plots, Zygote, Statistics, StableRNGs, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Optim, Measures, BenchmarkTools
using DelimitedFiles, ForwardDiff

include("simpleModel.jl")
using .simpleModel

rng = StableRNG(5958)

u0 = [6.0, 6.0, 6.0, 200.0, 0.0, 0.0, 0.0]
params = [0.3, 0.45, 0.006, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]

#Shi timing parameters
Eshift = 0.0
Eₘᵢₙ = 0.03

τₑₛ = 0.3
τₑₚ = 0.45 
Eₘₐₓ = 1.5
Rmv = 0.006
τ = 1.0

# Simulate from 0.0 to 3.0 without training
settling_tspan = (0.0, 3.0)
tsteps = range(0.0, 3.0, length=300)
prob = ODEProblem(NIK!, u0, settling_tspan, params)
settling_sol = solve(prob, Tsit5(), saveat = tsteps, reltol = 1e-4, abstol = 1e-7, dtmax=1e-2)

# Use the final state as the new initial condition
u0_settled = settling_sol.u[end]

# Training range
tspan = (3.0, 3.5)
num_of_samples = 50
tsteps = range(3.0, 3.5, length = num_of_samples)

loaded_data = readdlm("dataNorm/original_data.txt")
original_data = Array{Float64}(loaded_data)


NN = Lux.Chain(
    Lux.Dense(7, 16, tanh),
    Lux.Dense(16, 16, tanh),
    Lux.Dense(16, 16, tanh),
    Lux.Dense(16, 7)
)


p, st = Lux.setup(rng, NN)
p = 0.5*ComponentVector{Float64}(p)

# Compute mins and maxes from your dataset (column-wise)
data_min = minimum(original_data, dims=1)
data_max = maximum(original_data, dims=1)

# Normalize and denormalize functions
function normalize(x)
    return 2 .* ((x .- data_min) ./ (data_max .- data_min)) .- 1
end

function denormalize(x)
    return ((x .+ 1) ./ 2) .* (data_max .- data_min) .+ data_min
end

# Apply normalization to your data
original_data = normalize(original_data)
u0_settled = normalize(reshape(u0_settled, 1, :))[:]

function NIK_PINN!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u
    τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = params

    # Normalize u
    # u_norm = (u .- μ) ./ σ

    # Neural Network component (NN for correction)
    NN_output = NN(u, p, st)[1]

    # Rescale back 
    # NN_output_rescaled = NN_output .* σ
    NN_output_rescaled = NN_output

    # The differential equations with NN correction
    du[1] = (Qmv - Qav) * ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) +
        pLV / ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) * 
        DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) + 
        NN_output_rescaled[1]
        du[2] = (Qav - Qs ) / Csa + NN_output_rescaled[2] #Systemic arteries     
        du[3] = (Qs - Qmv) / Csv + NN_output_rescaled[3] # Venous
        du[4] = Qmv - Qav + NN_output_rescaled[4] # LV volume
        du[5]    = Valve(Zao, (du[1] - du[2]), u[1] - u[2]) + NN_output_rescaled[5]  # AV 
        du[6]   = Valve(Rmv, (du[3] - du[1]), u[3] - u[1]) + NN_output_rescaled[6]  # MV
        du[7]     = (du[2] - du[3]) / Rs + NN_output_rescaled[7] # Systemic flow

end


prob_NN = ODEProblem(NIK_PINN!, u0_settled, tspan, p)
s = solve(prob_NN, Vern7(), dtmax=1e-2, saveat = tsteps, reltol=1e-7, abstol=1e-4)

function predict(p)
    temp_prob = remake(prob_NN, p=p)
    temp_sol = solve(temp_prob, Vern7(), dtmax=1e-2, saveat = tsteps, reltol=1e-7, abstol=1e-4)
    return temp_sol
end

λ = 1e-3  # Regularization coefficient

function regularization_term(p)
    return sum(abs2, values(p))
end

function loss(p)
    pred = predict(p)

    pred_vals = hcat(
        ForwardDiff.value(pred[1, :]),
        ForwardDiff.value(pred[2, :]),
        ForwardDiff.value(pred[3, :]),
        ForwardDiff.value(pred[4, :]),
        ForwardDiff.value(pred[5, :]),
        ForwardDiff.value(pred[6, :]),
        ForwardDiff.value(pred[7, :]))

    data_loss = mean(abs2, pred_vals .- original_data)
    reg_loss = regularization_term(p)
    return data_loss + λ * reg_loss
    
    # return mean(abs2, pred_vals .- original_data)
end

losses1_0 = Float64[]

callback = function (p, l)
    push!(losses1_0, l)
    if length(losses1_0)%1 == 0
        println("Current loss after $(length(losses1_0)) iterations: $(losses1_0[end])")
      end
    return false
  end
  
  println("Hello, World!")

adtype = Optimization.AutoForwardDiff()
# adtype = Optimization.AutoZygote()
# adtype = Optimization.AutoFiniteDiff()
optf = Optimization.OptimizationFunction((x, p)->loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

# 1000 iterations using learning rate of 0.01
w1 = Optimization.solve(optprob, ADAM(0.01), callback=callback, maxiters = 1000)

# 1000 iterations using learning rate of 0.001
# optprob1 = Optimization.OptimizationProblem(optf, w1.u)
# w2 = Optimization.solve(optprob1, ADAM(0.001), callback=callback, maxiters = 100)

# 100 iteration using learning rate of 0.0001
optprob1 = Optimization.OptimizationProblem(optf, w1.u)
PINN_sol = Optimization.solve(optprob1, ADAM(0.0001), callback=callback, maxiters = 100)

prediction = predict(PINN_sol.u)

println(size(prediction))
println(size(prediction[1, :]))


data_to_save = hcat(
    prediction[1, :],
    prediction[2, :],
    prediction[3, :],
    prediction[4, :],
    prediction[5, :],
    prediction[6, :],
    prediction[7, :]
)

data_to_save = denormalize(data_to_save)

writedlm("dataNorm/pinn_data.txt", data_to_save)
println("Data from training range saved to pinn_data.txt")


# Test range
tspan2 = (0.0, 20.0)
num_of_samples = 2000
tsteps = range(0.0, 20.0, length = num_of_samples)

p_trained = PINN_sol.u
u0 = normalize(reshape(u0, 1, :))[:]
trained_NN = ODEProblem(NIK_PINN!, u0, tspan2, p_trained)
s = solve(trained_NN, Vern7(), dtmax=1e-2, saveat = tsteps, reltol=1e-7, abstol=1e-4)

data_to_save = hcat(
    s[1, :],
    s[2, :],
    s[3, :],
    s[4, :],
    s[5, :],
    s[6, :],
    s[7, :]
)

data_to_save = denormalize(data_to_save)

writedlm("dataNorm/pinn_extrapolation.txt", data_to_save)
println("Pinn extrapolation saved to pinn_extrapolation.txt")
