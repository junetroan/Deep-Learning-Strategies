#NODE-MS Financial System

# Test againt Julias multiple shooting
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Distributions
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
using ComponentArrays, Lux, DiffEqFlux, Optimization, OptimizationPolyalgorithms, DifferentialEquations, Plots
using DiffEqFlux: group_ranges
using Statistics
using StatsBase
using CSV, Tables, DataFrames
using OptimizationOptimisers
using StableRNGs
using LinearAlgebra
gr()

# Collecting data
data_path = "Multiple Shooting (MS)/ANODE-MS/Case Studies/Financial System/financial_time_series.csv"
data = CSV.read(data_path, DataFrame, header = true)

#Train/test Splits
split_ration = 0.75
train = data[1:Int(round(split_ration*size(data, 1))), 2]
test = data[Int(round(split_ration*size(data, 1))):end, 2]

# Data Cleaning and Normalization
t = data[:,1]
ir = data[:,2]

transformer = fit(ZScoreTransform, ir)
X_train_ref = StatsBase.transform(transformer, train)#[:, 1])
X_train = X_train_ref[1:1500]
#X_train = [X_train X_train]'
X_train = [X_train zeros(length(X_train))]'
t_train = data[1:1500, 1]

X_test = StatsBase.transform(transformer, test[:, 1])
t_test = collect(Int(round(split_ration*size(data, 1))):size(data, 1))

rng = StableRNG(1111)

group_size = 2
state = 2
iters = 2
tspan = (Float32(minimum(t_train)), Float32(maximum(t_train)))
datasize = length(X_train[1,:])
tsteps = range(tspan[1], tspan[2], length = datasize)
continuity_term = 10
u0 = [X_train[1], 0.0f0]

i = 1
rng_1 = StableRNG(i)
rng_2 = StableRNG(i + 1)

# Define the Neural Network
nn = Lux.Chain(Lux.Dense(state, 30, tanh), Lux.Dense(30, state))
p_init, st = Lux.setup(rng_1, nn)

params = ComponentVector(ANN = p_init, u0_states = 0.5*randn(rng_2, Float32, state - 1))

neuralode = NeuralODE(nn, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps,
 abstol = 1f-6, reltol = 1f-6, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))

function rhs(du, u, p, t)
    du[1:end] = nn(u, p.ANN, st)[1]
end

prob_node = ODEProblem(rhs, [u0[1]; params.u0_states], tspan, params, saveat = tsteps)

function loss_function(data, pred)
    return mean(abs2, data[1, :] - pred[1, :])
end

function continuity_loss_function(u_end, u_0)
    return mean(abs2, u_end - u_0)
end

function loss_multiple_shooting(p)
    new_prob = remake(prob_node, u0 = [u0[1]; p.u0_states], p = p)
    return multiple_shoot(p, X_train, tsteps, new_prob, loss_function, continuity_loss_function, AutoTsit5(Rosenbrock23(autodiff = false)),
                        group_size; continuity_term, abstol = 1f-6, reltol = 1f-6, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
end

function predict_final(p)
    prob_new = remake(prob_node, u0 = [u0[1]; p.u0_states], p = p)
    return Array(solve(prob_new, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, abstol = 1f-6, reltol = 1f-6))[1, :]
end

function final_loss(θ)
    X̂ = predict_final(θ)
    prediction_error = mean(abs2, X_train[1, :] .- X̂)
    prediction_error
end


function predict_single_shooting(p)
    prob_new = remake(prob_node, u0 = [u0[1]; p.u0_states], p = p)
    return Array(solve(prob_new, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, abstol = 1f-6, reltol = 1f-6))[1, :]
end

losses = Float32[]

callback = function (p, l, preds; doplot = false)
    push!(losses, final_loss(p))
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")

    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
@time res_ms = Optimization.solve(optprob, ADAM(), maxiters = 5000; callback = callback)

losses_df = DataFrame(loss = losses)
CSV.write("Multiple Shooting (MS)/ANODE-MS/Case Studies/Financial System/Loss Data/NODE-MS/Losses NODE-MS Financial System.csv", losses_df, writeheader = false)

# Evaluate Single Shooting
function loss_single_shooting(p)
    pred = predict_single_shooting(p)
    l = loss_function(Xₙ, pred)
    return l, pred
end

full_traj = predict_final(res_ms.u)

function plot_results(tp,real, pred)
    plot(tp, pred, label = "Training Prediction", title="Trained NODE-MS Model predicting FinCompSys", xlabel = "Time", ylabel = "Population")
    plot!(tp, real[1,:], label = "Training Data")
    plot!(legend=:topright)
    savefig("Multiple Shooting (MS)/ANODE-MS/Case Studies/Financial System/Plots/Plot NODE-MS Financial System.png")
end

plot_results(t_train, X_train, full_traj)

