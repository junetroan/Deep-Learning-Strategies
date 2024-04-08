# Test - Lynx Hare PEM 

using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
using CSV, Tables, DataFrames
using DataInterpolations
using OrdinaryDiffEq
using OptimizationPolyalgorithms
using DiffEqFlux
using Plots
using Statistics
using StatsBase
gr()


data_path = "Multiple Shooting (MS)/ANODE-MS/Data/lynx_hare_data.csv"
data = CSV.read(data_path, DataFrame)

#Train/test Splits
split_ration = 0.75
train = data[1:Int(round(split_ration*size(data, 1))), :]
test = data[Int(round(split_ration*size(data, 1))):end, :]

# Data Cleaning and Normalization
hare_data = data[:, 2]
lynx_data = data[:, 3]

transformer = fit(ZScoreTransform, hare_data)
X_train = Float32.(StatsBase.transform(transformer, train[:, 2]))
X_test = Float32.(StatsBase.transform(transformer, test[:, 2]))
X_new = reshape(X_train, 1, :)
unknown_X = zeros(Float32, 1, 68)
X = vcat(X_new, unknown_X)
y = X[1,:]

t = Float32.(collect(1:size(data, 1)))
t_train = Float32.(collect(1:Int(round(split_ration*size(data, 1)))))
t_test = Float32.(collect(Int(round(split_ration*size(data, 1))):size(data, 1)))
tspan = (t_train[1], t_train[end])
tsteps = range(tspan[1], tspan[2], length = length(y))

rng = StableRNG(1111)
rng1 = StableRNG(1003)
rng2 = StableRNG(1004)

# Interpolation of given data
y_zoh = ConstantInterpolation(y, tsteps)

# Definition of neural network
iters = 2
state = 2
U = Lux.Chain(Lux.Dense(state, 30, tanh),Lux.Dense(30, state))
p, st = Lux.setup(rng1, U)
K = rand(rng, Float32, 2)
u0 = [y[1], y[1]]
rng1 = StableRNG(iters+1)
rng2 = StableRNG(iters+2)

U = Lux.Chain(Lux.Dense(state, 30, tanh),Lux.Dense(30, state))
p, st = Lux.setup(rng1, U)

K = rand(rng2, Float32, 2)

function predictor!(du,u,p,t)
    û = U(u, p.vector_field_model, st)[1]
    yt = y_zoh(t)
    e = yt .- û[1]
    du[1:end] =  û[1:end] .+ abs.(p.K) .* e
end

params = ComponentVector{Float32}(vector_field_model = p, K = K)
prob_nn = ODEProblem(predictor!, u0 , tspan, params, saveat = 1.0f0 )
soln_nn = Array(solve(prob_nn, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-8, reltol = 1e-8, saveat = 1.0f0 ))

function prediction(p)
    _prob = remake(prob_nn, u0 = u0, p = p)
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
    Array(solve(_prob, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-8, reltol = 1e-8, saveat = tsteps , sensealg = sensealg))
end

prediction(params)

function predloss(p)
    yh = prediction(p)
    e2 = mean(abs2, y .- yh[1,:])
    return e2
end

predloss(params)

losses = Float32[]
callback = function (p, l)
    push!(losses, predloss(p))

    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> predloss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
res_ms = Optimization.solve(optprob, ADAM(), maxiters = 10000, verbose = false, callback=callback)

losses_df = DataFrame(losses = losses)
CSV.write("Prediction Error Methods (PEM)/Simulations/Results/Loss Data/Losses $i.csv", losses_df, writeheader = false)

full_traj = prediction(res_ms.u)
full_traj_loss = predloss(res_ms.u)
push!(fulltraj_losses, full_traj_loss)

function plot_results(t, real, pred)
    plot(t, pred[1,:], label = "Training Prediction", title="Iteration $i of Randomised PEM Model", xlabel = "Time", ylabel = "Population")
    scatter!(t, real[1,:], label = "Training Data")
    plot!(legend=:topright)
    savefig("Prediction Error Methods (PEM)/Simulations/Results/Plots/Simulation $i.png")
end

plot_results(t, Xₙ, full_traj)