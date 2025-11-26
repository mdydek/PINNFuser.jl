# Import packages
using OrdinaryDiffEq, DataDrivenDiffEq, SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, Plots, StableRNGs
using Optim
using Measures
gr()
using PlotlySave
using BenchmarkTools
rng = StableRNG(5958)

# Smith cardiovascular model
function CVS!(du, u, p, t)

    Qmt, Qav, Qtc, Qpv, Vlv, Vao, Vvc, Vrv, Vpa, Vpu = u

    Elvf,
    Eao,
    Evc,
    Ervf,
    Epa,
    Epu,
    Rmt,
    Rav,
    Rsys,
    Rtc,
    Rpv,
    Rpul,
    Lmt,
    Lav,
    Ltc,
    Lpv,
    Vdlvf,
    Vdao,
    Vdvc,
    Vdrvf,
    Vdpa,
    Vdpu,
    P0lvf,
    P0rvf,
    lambdalvf,
    lambdarvf,
    Espt,
    V0lvf,
    V0rvf,
    P0spt,
    P0pcd,
    V0spt,
    V0pcd,
    lambdaspt,
    lambdapcd,
    Vdspt,
    Pth = p

    e = exp(-80 * (mod(t, 0.75) - 0.375)^2)

    Vpcd = Vlv + Vrv
    Ppcd = P0pcd * (exp(lambdapcd * (Vpcd - V0pcd)) - 1)
    Pperi = Ppcd + Pth

    Vspt = 0
    dy = 0.01

    while abs(dy) > 1e-5 * abs(Vspt)
        f =
            e * Espt * (Vspt - Vdspt) +
            (1 - e) * P0spt * (exp(lambdaspt * (Vspt - V0spt)) - 1) -
            e * Elvf * (Vlv - Vspt - Vdlvf) -
            (1 - e) * P0lvf * (exp(lambdalvf * (Vlv - Vspt - V0lvf)) - 1) +
            e * Ervf * (Vrv + Vspt - Vdrvf) +
            (1 - e) * P0rvf * (exp(lambdarvf * (Vrv + Vspt - V0rvf)) - 1)

        df =
            e * Espt +
            lambdaspt * (1 - e) * P0spt * exp(lambdaspt * (Vspt - V0spt)) +
            e * Elvf +
            lambdalvf * (1 - e) * P0lvf * exp(lambdalvf * (Vlv - Vspt - V0lvf)) +
            e * Ervf +
            lambdarvf * (1 - e) * P0rvf * exp(lambdarvf * (Vrv + Vspt - V0rvf))

        dy = f / df
        Vspt = Vspt - dy
    end

    Vlvf = Vlv - Vspt
    Vrvf = Vrv + Vspt

    Plvf =
        e * Elvf * (Vlvf - Vdlvf) + (1 - e) * P0lvf * (exp(lambdalvf * (Vlvf - V0lvf)) - 1)
    Prvf =
        e * Ervf * (Vrvf - Vdrvf) + (1 - e) * P0rvf * (exp(lambdarvf * (Vrvf - V0rvf)) - 1)

    Plv = Plvf + Pperi
    Prv = Prvf + Pperi

    Pao = Eao * (Vao - Vdao)
    Pvc = Evc * (Vvc - Vdvc)
    Ppa = Epa * (Vpa - Vdpa) + Pth
    Ppu = Epu * (Vpu - Vdpu) + Pth

    Qsys = (Pao - Pvc) / Rsys
    Qpul = (Ppa - Ppu) / Rpul

    if Ppu - Plv > 0 || Qmt > 0
        du[1] = (Ppu - Plv - (Qmt * Rmt)) / Lmt
    else
        du[1] = 0
    end

    if Plv - Pao > 0 || Qav > 0
        du[2] = (Plv - Pao - (Qav * Rav)) / Lav
    else
        du[2] = 0
    end

    if Pvc - Prv > 0 || Qtc > 0
        du[3] = (Pvc - Prv - (Qtc * Rtc)) / Ltc
    else
        du[3] = 0
    end

    if Prv - Ppa > 0 || Qpv > 0
        du[4] = (Prv - Ppa - (Qpv * Rpv)) / Lpv
    else
        du[4] = 0
    end

    Qmt = max(Qmt, 0)
    Qav = max(Qav, 0)
    Qtc = max(Qtc, 0)
    Qpv = max(Qpv, 0)

    du[5] = Qmt - Qav
    du[6] = Qav - Qsys
    du[7] = Qsys - Qtc
    du[8] = Qtc - Qpv
    du[9] = Qpv - Qpul
    du[10] = Qpul - Qmt
end

# Training range
tspan = (0.0, 0.3)
tsteps = range(tspan[1], tspan[2], length = 30)

# Initial condition
u0 = [245.5813, 0, 190.0661, 0, 94.6812, 133.3381, 329.7803, 90.7302, 43.0123, 808.4579]

# Parameters
p_ = [
    2.8798,
    0.6913,
    0.0059,
    0.585,
    0.369,
    0.0073,
    0.0158,
    0.018,
    1.0889,
    0.0237,
    0.0055,
    0.1552,
    7.6968e-5,
    1.2189e-4,
    8.0093e-5,
    1.4868e-4,
    0,
    0,
    0,
    0,
    0,
    0,
    0.1203,
    0.2157,
    0.033,
    0.023,
    48.754,
    0,
    0,
    1.1101,
    0.5003,
    2,
    200,
    0.435,
    0.03,
    2,
    -4,
]

# Solve to generate synthetic "ground truth" data
prob = ODEProblem(CVS!, u0, tspan, p_)
@time solution =
    solve(prob, Tsit5(), dtmax = 1e-2, abstol = 1e-7, reltol = 1e-4, saveat = tsteps)
original_data = Array(solution)

# Add noise
noise_magnitude = 0.00
sd = std(original_data, dims = 2)
noisy_data =
    original_data .+
    (noise_magnitude * sd) .* randn(rng, eltype(original_data), size(original_data))

# Visualise training data
plot!(tsteps, noisy_data')

# Use solution to calculate "ground truth" Vspt and Pperi
Qmt = noisy_data[1, :]
Qav = noisy_data[2, :]
Qtc = noisy_data[3, :]
Qpv = noisy_data[4, :]
Vlv = noisy_data[5, :]
Vao = noisy_data[6, :]
Vvc = noisy_data[7, :]
Vrv = noisy_data[8, :]
Vpa = noisy_data[9, :]
Vpu = noisy_data[10, :]

Elvf,
Eao,
Evc,
Ervf,
Epa,
Epu,
Rmt,
Rav,
Rsys,
Rtc,
Rpv,
Rpul,
Lmt,
Lav,
Ltc,
Lpv,
Vdlvf,
Vdao,
Vdvc,
Vdrvf,
Vdpa,
Vdpu,
P0lvf,
P0rvf,
lambdalvf,
lambdarvf,
Espt,
V0lvf,
V0rvf,
P0spt,
P0pcd,
V0spt,
V0pcd,
lambdaspt,
lambdapcd,
Vdspt,
Pth = p_

e = exp.(-80 * (mod.(tsteps, 0.75) .- 0.375) .^ 2)
Vpcd = Vlv .+ Vrv
Ppcd = P0pcd .* (exp.(lambdapcd .* (Vpcd .- V0pcd)) .- 1)
Pperi = Ppcd .+ Pth

Vspt = zeros(length(tsteps))
dx = 0.01 .+ zeros(length(tsteps))

for i = 1:length(tsteps)
    while abs(dx[i]) > 1e-5 * abs(Vspt[i])
        f =
            e[i] .* Espt .* (Vspt[i] .- Vdspt) .+
            (1 .- e[i]) .* P0spt .* (exp.(lambdaspt .* (Vspt[i] .- V0spt)) .- 1) .-
            e[i] .* Elvf .* (Vlv[i] .- Vspt[i] .- Vdlvf) .-
            (1 .- e[i]) .* P0lvf .* (exp.(lambdalvf .* (Vlv[i] .- Vspt[i] .- V0lvf)) .- 1) .+
            e[i] .* Ervf .* (Vrv[i] .+ Vspt[i] .- Vdrvf) .+
            (1 .- e[i]) .* P0rvf .* (exp.(lambdarvf .* (Vrv[i] .+ Vspt[i] .- V0rvf)) .- 1)

        df =
            e[i] .* Espt .+
            lambdaspt .* (1 .- e[i]) .* P0spt .* exp.(lambdaspt .* (Vspt[i] .- V0spt)) .+
            e[i] .* Elvf .+
            lambdalvf .* (1 .- e[i]) .* P0lvf .*
            exp.(lambdalvf .* (Vlv[i] .- Vspt[i] .- V0lvf)) .+ e[i] .* Ervf .+
            lambdarvf .* (1 .- e[i]) .* P0rvf .*
            exp.(lambdarvf .* (Vrv[i] .+ Vspt[i] .- V0rvf))

        dx[i] = f / df
        Vspt[i] = Vspt[i] - dx[i]
    end
end

# Elu activation function
function elu(x)
    if x >= 0
        return x
    else
        return exp.(x) .- 1
    end
end

# Multilayer feedforward network
NN = Lux.Chain(
    Lux.Dense(5, 10, elu),
    Lux.Dense(10, 10, elu),
    Lux.Dense(10, 10, elu),
    Lux.Dense(10, 2),
)

# Get the initial network parameters
p, st = Lux.setup(rng, NN)
p = 0.5 * ComponentVector{Float64}(p)

# Hybrid neural ODE
function dudt!(du, u, p, t)

    Qmt, Qav, Qtc, Qpv, Vlv, Vao, Vvc, Vrv, Vpa, Vpu = u

    Elvf,
    Eao,
    Evc,
    Ervf,
    Epa,
    Epu,
    Rmt,
    Rav,
    Rsys,
    Rtc,
    Rpv,
    Rpul,
    Lmt,
    Lav,
    Ltc,
    Lpv,
    Vdlvf,
    Vdao,
    Vdvc,
    Vdrvf,
    Vdpa,
    Vdpu,
    P0lvf,
    P0rvf,
    lambdalvf,
    lambdarvf,
    Espt,
    V0lvf,
    V0rvf,
    P0spt,
    P0pcd,
    V0spt,
    V0pcd,
    lambdaspt,
    lambdapcd,
    Vdspt,
    Pth = p_

    e = exp(-80 * (mod(t, 0.75) - 0.375)^2)
    Vpcd = Vlv + Vrv
    # Network output
    z = NN([Vlv, Vao, Vvc, Vrv, Vpa], p, st)[1]
    # Network outputs defined as Pperi and Vspt
    Pperi = z[1]
    Vspt = z[2]

    Vlvf = Vlv - Vspt
    Vrvf = Vrv + Vspt

    Plvf =
        e * Elvf * (Vlvf - Vdlvf) + (1 - e) * P0lvf * (exp(lambdalvf * (Vlvf - V0lvf)) - 1)
    Prvf =
        e * Ervf * (Vrvf - Vdrvf) + (1 - e) * P0rvf * (exp(lambdarvf * (Vrvf - V0rvf)) - 1)

    Plv = Plvf + Pperi
    Prv = Prvf + Pperi

    Pao = Eao * (Vao - Vdao)
    Pvc = Evc * (Vvc - Vdvc)
    Ppa = Epa * (Vpa - Vdpa) + Pth
    Ppu = Epu * (Vpu - Vdpu) + Pth

    Qsys = (Pao - Pvc) / Rsys
    Qpul = (Ppa - Ppu) / Rpul

    if Ppu - Plv > 0 || Qmt > 0
        du[1] = (Ppu - Plv - (Qmt * Rmt)) / Lmt
    else
        du[1] = 0
    end

    if Plv - Pao > 0 || Qav > 0
        du[2] = (Plv - Pao - (Qav * Rav)) / Lav
    else
        du[2] = 0
    end

    if Pvc - Prv > 0 || Qtc > 0
        du[3] = (Pvc - Prv - (Qtc * Rtc)) / Ltc
    else
        du[3] = 0
    end

    if Prv - Ppa > 0 || Qpv > 0
        du[4] = (Prv - Ppa - (Qpv * Rpv)) / Lpv
    else
        du[4] = 0
    end

    Qmt = max(Qmt, 0)
    Qav = max(Qav, 0)
    Qtc = max(Qtc, 0)
    Qpv = max(Qpv, 0)

    du[5] = Qmt - Qav
    du[6] = Qav - Qsys
    du[7] = Qsys - Qtc
    du[8] = Qtc - Qpv
    du[9] = Qpv - Qpul
    du[10] = Qpul - Qmt
end

# Solve hybrid neural ODE to generate prediction
prob_NN = ODEProblem(dudt!, u0, tspan, p)
s = solve(prob_NN, Vern7(), dtmax = 1e-2, abstol = 1e-7, reltol = 1e-4, saveat = solution.t)

# Function to generate hybrid neural ODE prediction
function predict(p)
    temp_prob = remake(prob_NN, p = p)
    temp_sol = solve(
        temp_prob,
        Vern7(),
        dtmax = 1e-2,
        abstol = 1e-7,
        reltol = 1e-4,
        saveat = solution.t,
    )
    return temp_sol
end

# MSE loss function
function loss(p)
    pred = predict(p)
    if size(pred) == size(solution)
        return mean(abs2, Array(pred) - noisy_data)
    else
        return Inf, Array(pred) # Return infiinite loss if solution is unstable
    end
end

# Array to store losses
losses1_0 = Float64[]

# Callback function to print losses
callback = function (p, l)
    push!(losses1_0, l)
    if length(losses1_0) % 1 == 0
        #   println("Current loss after $(length(losses1_0)) iterations: $(losses4_5[end])")
        println("Current loss after $(length(losses1_0))")
    end
    return false
end

# Set up optimisation
adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

# 1000 iterations using learning rate of 0.01
w1 = Optimization.solve(optprob, ADAM(0.01), callback = callback, maxiters = 1000)

# 100 iteration using learning rate of 0.0001
optprob2 = Optimization.OptimizationProblem(optf, w1.u)
w1_0 = Optimization.solve(optprob2, ADAM(0.0001), callback = callback, maxiters = 100)

# Get hybrid neural ODE predictions (10 models in ensemble and 3 levels of noise)
prediction1_0 = predict(w1_0.u)[[5, 6, 7, 8, 9], :]
prediction1_2 = predict(w1_2.u)[[5, 6, 7, 8, 9], :]
prediction1_5 = predict(w1_5.u)[[5, 6, 7, 8, 9], :]

prediction2_0 = predict(w2_0.u)[[5, 6, 7, 8, 9], :]
prediction2_2 = predict(w2_2.u)[[5, 6, 7, 8, 9], :]
prediction2_5 = predict(w2_5.u)[[5, 6, 7, 8, 9], :]

prediction3_0 = predict(w3_0.u)[[5, 6, 7, 8, 9], :]
prediction3_2 = predict(w3_2.u)[[5, 6, 7, 8, 9], :]
prediction3_5 = predict(w3_5.u)[[5, 6, 7, 8, 9], :]

prediction4_0 = predict(w4_0.u)[[5, 6, 7, 8, 9], :]
prediction4_2 = predict(w4_2.u)[[5, 6, 7, 8, 9], :]
prediction4_5 = predict(w4_5.u)[[5, 6, 7, 8, 9], :]

prediction5_0 = predict(w5_0.u)[[5, 6, 7, 8, 9], :]
prediction5_2 = predict(w5_2.u)[[5, 6, 7, 8, 9], :]
prediction5_5 = predict(w5_5.u)[[5, 6, 7, 8, 9], :]

prediction6_0 = predict(w6_0.u)[[5, 6, 7, 8, 9], :]
prediction6_2 = predict(w6_2.u)[[5, 6, 7, 8, 9], :]
prediction6_5 = predict(w6_5.u)[[5, 6, 7, 8, 9], :]

prediction7_0 = predict(w7_0.u)[[5, 6, 7, 8, 9], :]
prediction7_2 = predict(w7_2.u)[[5, 6, 7, 8, 9], :]
prediction7_5 = predict(w7_5.u)[[5, 6, 7, 8, 9], :]

prediction8_0 = predict(w8_0.u)[[5, 6, 7, 8, 9], :]
prediction8_2 = predict(w8_2.u)[[5, 6, 7, 8, 9], :]
prediction8_5 = predict(w8_5.u)[[5, 6, 7, 8, 9], :]

prediction9_0 = predict(w9_0.u)[[5, 6, 7, 8, 9], :]
prediction9_2 = predict(w9_2.u)[[5, 6, 7, 8, 9], :]
prediction9_5 = predict(w9_5.u)[[5, 6, 7, 8, 9], :]

prediction10_0 = predict(w10_0.u)[[5, 6, 7, 8, 9], :]
prediction10_2 = predict(w10_2.u)[[5, 6, 7, 8, 9], :]
prediction10_5 = predict(w10_5.u)[[5, 6, 7, 8, 9], :]

# Get the learned network dynamics of the corresponding trained model
NN_dynamics1_0 = NN(prediction1_0, w1_0.u, st)[1]
NN_dynamics1_2 = NN(prediction1_2, w1_2.u, st)[1]
NN_dynamics1_5 = NN(prediction1_5, w1_5.u, st)[1]

NN_dynamics2_0 = NN(prediction2_0, w2_0.u, st)[1]
NN_dynamics2_2 = NN(prediction2_2, w2_2.u, st)[1]
NN_dynamics2_5 = NN(prediction2_5, w2_5.u, st)[1]

NN_dynamics3_0 = NN(prediction3_0, w3_0.u, st)[1]
NN_dynamics3_2 = NN(prediction3_2, w3_2.u, st)[1]
NN_dynamics3_5 = NN(prediction3_5, w3_5.u, st)[1]

NN_dynamics4_0 = NN(prediction4_0, w4_0.u, st)[1]
NN_dynamics4_2 = NN(prediction4_2, w4_2.u, st)[1]
NN_dynamics4_5 = NN(prediction4_5, w4_5.u, st)[1]

NN_dynamics5_0 = NN(prediction5_0, w5_0.u, st)[1]
NN_dynamics5_2 = NN(prediction5_2, w5_2.u, st)[1]
NN_dynamics5_5 = NN(prediction5_5, w5_5.u, st)[1]

NN_dynamics6_0 = NN(prediction6_0, w6_0.u, st)[1]
NN_dynamics6_2 = NN(prediction6_2, w6_2.u, st)[1]
NN_dynamics6_5 = NN(prediction6_5, w6_5.u, st)[1]

NN_dynamics7_0 = NN(prediction7_0, w7_0.u, st)[1]
NN_dynamics7_2 = NN(prediction7_2, w7_2.u, st)[1]
NN_dynamics7_5 = NN(prediction7_5, w7_5.u, st)[1]

NN_dynamics8_0 = NN(prediction8_0, w8_0.u, st)[1]
NN_dynamics8_2 = NN(prediction8_2, w8_2.u, st)[1]
NN_dynamics8_5 = NN(prediction8_5, w8_5.u, st)[1]

NN_dynamics9_0 = NN(prediction9_0, w9_0.u, st)[1]
NN_dynamics9_2 = NN(prediction9_2, w9_2.u, st)[1]
NN_dynamics9_5 = NN(prediction9_5, w9_5.u, st)[1]

NN_dynamics10_0 = NN(prediction10_0, w10_0.u, st)[1]
NN_dynamics10_2 = NN(prediction10_2, w10_2.u, st)[1]
NN_dynamics10_5 = NN(prediction10_5, w10_5.u, st)[1]

# Take mean of predictions of the models in the ensemble (input to SR)
avgIn_0 =
    (
        prediction1_0 +
        prediction2_0 +
        prediction3_0 +
        prediction4_0 +
        prediction5_0 +
        prediction6_0 +
        prediction7_0 +
        prediction8_0 +
        prediction9_0 +
        prediction10_0
    ) / 10

# Take mean of learned network dynamics in the ensemble (target for SR)
avgNN_0 =
    (
        NN_dynamics1_0 +
        NN_dynamics2_0 +
        NN_dynamics3_0 +
        NN_dynamics4_0 +
        NN_dynamics5_0 +
        NN_dynamics6_0 +
        NN_dynamics7_0 +
        NN_dynamics8_0 +
        NN_dynamics9_0 +
        NN_dynamics10_0
    ) / 10


# Run SR using PySR

# Learned terms with no noise
learned_Pperi0 = ((Vrv .- 105.56476) ./ (3227.8403 ./ Vrv)) .- 3.7175448
learned_Vspt0 =
    (970.5831 ./ (Vao .- Vpa)) .- (7.331516 .- (12.834902 ./ (Vao .+ (2.5382192 .- Vrv))))

# Learned terms with 2% noise
learned_Pperi2 = ((Vrv .+ -91.48883) ./ Vpa) .+ -4.1087065
learned_Vspt2 =
    -8.62723 .- ((-1080.363 .- exp.(Vrv ./ (Vpa .+ -10.053128))) ./ (Vao .- Vpa))

# Learned terms with 5% noise
learned_Pperi5 = ((Vrv .+ -88.12736) ./ Vpa) .+ -4.2060776
learned_Vspt5 =
    ((Vvc ./ (Vao .- Vpa)) .+ -2.4558141) .* exp.(exp.(Vrv ./ (Vpa .* (Vpa .* 0.29237527))))

# Visualise (averaged) learned dynamics of the models as well as the dynamics of the learned terms
fig6 = plot!(
    tsteps,
    Pperi,
    xlabel = "time (s)",
    ylabel = "pressure (mmHg)",
    label = "Pperi",
    legend = :bottom,
    legendfontsize = 9,
    color = :black,
    linewidth = 2,
    guidefontsize = :13,
    size = (650, 350),
)
plot!(tsteps, avgNN_0[1, :], label = "Averaged prediction", linewidth = 2, color = :red)
plot!(tsteps, learned_Pperi0, linewidth = 2, label = "Learned function", color = :green3)
plot!(tsteps, NN_dynamics1_0[1, :], label = "Model 1", linewidth = 1.5, color = :blue)
plot!(tsteps, NN_dynamics2_0[1, :], label = "Model 2", linewidth = 1.5, color = :green)
plot!(tsteps, NN_dynamics3_0[1, :], label = "Model 3", linewidth = 1.5, color = :gold)
plot!(tsteps, NN_dynamics4_0[1, :], label = "Model 4", linewidth = 1.5, color = :magenta)
plot!(tsteps, NN_dynamics5_0[1, :], label = "Model 5", linewidth = 1.5, color = :cyan)
plot!(
    tsteps,
    NN_dynamics6_0[1, :],
    label = "Model 6",
    linewidth = 1.5,
    color = :mediumvioletred,
)
plot!(tsteps, NN_dynamics7_0[1, :], label = "Model 7", linewidth = 1.5, color = :chocolate)
plot!(
    tsteps,
    NN_dynamics8_0[1, :],
    label = "Model 8",
    linewidth = 1.5,
    color = :navajowhite4,
)
plot!(
    tsteps,
    NN_dynamics9_0[1, :],
    label = "Model 9",
    linewidth = 1.5,
    color = :darkgoldenrod2,
)
plot!(tsteps, NN_dynamics10_0[1, :], label = "Model 10", linewidth = 1.5, color = :lime)

# Time span for extrapolation
tspan2 = (0.0, 10.0)
tsteps2 = range(tspan2[1], tspan2[2], length = 1000)

# Get "ground truth" extrapolation
prob2 = ODEProblem(CVS!, u0, tspan2, p_)
solution2 =
    solve(prob2, Vern7(), dtmax = 1e-2, abstol = 1e-7, reltol = 1e-4, saveat = tsteps2)
sol_ex = Array(solution2)

# Add noise to ground truth 
noise_magnitude_ex = 0.00
sd_ex = std(sol_ex, dims = 2)
sol_ex = sol_ex .+ (noise_magnitude_ex * sd_ex) .* randn(rng, eltype(sol_ex), size(sol_ex))

# Set up trained extrpolation problem
prob_NN2 = ODEProblem(dudt!, u0, tspan2, p)
train_prob = ODEProblem(dudt!, u0, tspan2, w1.u)

# Solve to generate trained extrapolation
@time train_sol = solve(
    train_prob,
    Tsit5(),
    dtmax = 1e-2,
    abstol = 1e-7,
    reltol = 1e-4,
    saveat = solution2.t,
)
learned_ex = Array(train_sol)

# Substitute learned terms back into cardiovascular model to generate partially learned model
function Learned!(du, u, p, t)

    Qmt, Qav, Qtc, Qpv, Vlv, Vao, Vvc, Vrv, Vpa, Vpu = u

    Elvf,
    Eao,
    Evc,
    Ervf,
    Epa,
    Epu,
    Rmt,
    Rav,
    Rsys,
    Rtc,
    Rpv,
    Rpul,
    Lmt,
    Lav,
    Ltc,
    Lpv,
    Vdlvf,
    Vdao,
    Vdvc,
    Vdrvf,
    Vdpa,
    Vdpu,
    P0lvf,
    P0rvf,
    lambdalvf,
    lambdarvf,
    Espt,
    V0lvf,
    V0rvf,
    P0spt,
    P0pcd,
    V0spt,
    V0pcd,
    lambdaspt,
    lambdapcd,
    Vdspt,
    Pth = p

    e = exp(-80 * (mod(t, 0.75) - 0.375)^2)

    Vpcd = Vlv + Vrv

    # Learned terms
    Pperi = ((Vrv .- 105.56476) ./ (3227.8403 ./ Vrv)) .- 3.7175448
    Vspt =
        (970.5831 ./ (Vao .- Vpa)) .-
        (7.331516 .- (12.834902 ./ (Vao .+ (2.5382192 .- Vrv))))

    Vlvf = Vlv - Vspt
    Vrvf = Vrv + Vspt

    Plvf =
        e * Elvf * (Vlvf - Vdlvf) + (1 - e) * P0lvf * (exp(lambdalvf * (Vlvf - V0lvf)) - 1)
    Prvf =
        e * Ervf * (Vrvf - Vdrvf) + (1 - e) * P0rvf * (exp(lambdarvf * (Vrvf - V0rvf)) - 1)

    Plv = Plvf + Pperi
    Prv = Prvf + Pperi

    Pao = Eao * (Vao - Vdao)
    Pvc = Evc * (Vvc - Vdvc)
    Ppa = Epa * (Vpa - Vdpa) + Pth
    Ppu = Epu * (Vpu - Vdpu) + Pth

    Qsys = (Pao - Pvc) / Rsys
    Qpul = (Ppa - Ppu) / Rpul

    if Ppu - Plv > 0 || Qmt > 0
        du[1] = (Ppu - Plv - (Qmt * Rmt)) / Lmt
    else
        du[1] = 0
    end

    if Plv - Pao > 0 || Qav > 0
        du[2] = (Plv - Pao - (Qav * Rav)) / Lav
    else
        du[2] = 0
    end

    if Pvc - Prv > 0 || Qtc > 0
        du[3] = (Pvc - Prv - (Qtc * Rtc)) / Ltc
    else
        du[3] = 0
    end

    if Prv - Ppa > 0 || Qpv > 0
        du[4] = (Prv - Ppa - (Qpv * Rpv)) / Lpv
    else
        du[4] = 0
    end

    Qmt = max(Qmt, 0)
    Qav = max(Qav, 0)
    Qtc = max(Qtc, 0)
    Qpv = max(Qpv, 0)

    du[5] = Qmt - Qav
    du[6] = Qav - Qsys
    du[7] = Qsys - Qtc
    du[8] = Qtc - Qpv
    du[9] = Qpv - Qpul
    du[10] = Qpul - Qmt
end

# Set up learned extrapolation problem
SRprob = ODEProblem(Learned!, u0, tspan2, p_)

# Solve to generate learned extrapolation
@time SRsol =
    solve(SRprob, Tsit5(), dtmax = 1e-2, abstol = 1e-7, reltol = 1e-4, saveat = tsteps2)
SR_ex = Array(SRsol)

# Cardiovascular model with Vspt and Pperi omitted for comparison
function NoVI!(du, u, p, t)

    Qmt, Qav, Qtc, Qpv, Vlv, Vao, Vvc, Vrv, Vpa, Vpu = u

    Elvf,
    Eao,
    Evc,
    Ervf,
    Epa,
    Epu,
    Rmt,
    Rav,
    Rsys,
    Rtc,
    Rpv,
    Rpul,
    Lmt,
    Lav,
    Ltc,
    Lpv,
    Vdlvf,
    Vdao,
    Vdvc,
    Vdrvf,
    Vdpa,
    Vdpu,
    P0lvf,
    P0rvf,
    lambdalvf,
    lambdarvf,
    Espt,
    V0lvf,
    V0rvf,
    P0spt,
    P0pcd,
    V0spt,
    V0pcd,
    lambdaspt,
    lambdapcd,
    Vdspt,
    Pth = p

    e = exp(-80 * (mod(t, 0.75) - 0.375)^2)

    Vlvf = Vlv
    Vrvf = Vrv

    Plvf =
        e * Elvf * (Vlvf - Vdlvf) + (1 - e) * P0lvf * (exp(lambdalvf * (Vlvf - V0lvf)) - 1)
    Prvf =
        e * Ervf * (Vrvf - Vdrvf) + (1 - e) * P0rvf * (exp(lambdarvf * (Vrvf - V0rvf)) - 1)

    Plv = Plvf
    Prv = Prvf

    Pao = Eao * (Vao - Vdao)
    Pvc = Evc * (Vvc - Vdvc)
    Ppa = Epa * (Vpa - Vdpa) + Pth
    Ppu = Epu * (Vpu - Vdpu) + Pth

    Qsys = (Pao - Pvc) / Rsys
    Qpul = (Ppa - Ppu) / Rpul

    if Ppu - Plv > 0 || Qmt > 0
        du[1] = (Ppu - Plv - (Qmt * Rmt)) / Lmt
    else
        du[1] = 0
    end

    if Plv - Pao > 0 || Qav > 0
        du[2] = (Plv - Pao - (Qav * Rav)) / Lav
    else
        du[2] = 0
    end

    if Pvc - Prv > 0 || Qtc > 0
        du[3] = (Pvc - Prv - (Qtc * Rtc)) / Ltc
    else
        du[3] = 0
    end

    if Prv - Ppa > 0 || Qpv > 0
        du[4] = (Prv - Ppa - (Qpv * Rpv)) / Lpv
    else
        du[4] = 0
    end

    Qmt = max(Qmt, 0)
    Qav = max(Qav, 0)
    Qtc = max(Qtc, 0)
    Qpv = max(Qpv, 0)

    du[5] = Qmt - Qav
    du[6] = Qav - Qsys
    du[7] = Qsys - Qtc
    du[8] = Qtc - Qpv
    du[9] = Qpv - Qpul
    du[10] = Qpul - Qmt
end

# Set up problem with no ventricular interaction
NoVIprob = ODEProblem(NoVI!, u0, tspan2, p_)

# Solve to generate extrapolation without ventricular interaction
@time NoVIsol =
    solve(NoVIprob, Vern7(), dtmax = 1e-2, abstol = 1e-7, reltol = 1e-4, saveat = tsteps2)
NoVIsol = Array(NoVIsol)
