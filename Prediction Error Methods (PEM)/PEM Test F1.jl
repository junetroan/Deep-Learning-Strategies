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
plotly()

rng1 = StableRNG(1111)
data_path = "Multiple Shooting (MS)/ANODE-MS/Case Studies/F1 Telemetry/test-f1.csv"
data = CSV.read(data_path, DataFrame, header = true)

#Train/test Splits
split_ration = 0.4
train = data[1:Int(round(split_ration*size(data, 1))), :]
test = data[Int(round(split_ration*size(data, 1))):end, :]

# Data Cleaning and Normalization
t = 0.0:1.0:581
speed = data[:,1]

train_data = convert(Vector{Float32}, train[:,1])
test_data = convert(Vector{Float32}, test[:,1])

transformer = fit(ZScoreTransform, train_data)
X_train = StatsBase.transform(transformer, train_data)
X_test = StatsBase.transform(transformer, test_data)
t_test = convert(Vector{Float32}, collect(Int(round(split_ration*size(data, 1))):size(data, 1)))
t_train = convert(Vector{Float32}, collect(1:Int(round(split_ration*size(data, 1)))))
tspan = (t_train[1], t_train[end])
tsteps = range(tspan[1], tspan[2], length = length(X_train))

y = X_train
# Interpolation of given data
y_zoh = ConstantInterpolation(X_train, tsteps)

iters = 2
state = 2
u0 = [0.0f0, 0.0f0]

# Definition of neural network
state = 2
U = Lux.Chain(Lux.Dense(state, 30, tanh),Lux.Dense(30, state))
p, st = Lux.setup(rng1, U)
K = Float32[0.214566, 0.028355]

y_zoh = ConstantInterpolation(y, tsteps)

#Definition of the model 
function predictor!(du,u,p,t)
    û = U(u, p.vector_field_model, st)[1]
    yt = y_zoh(t)
    e = yt .- û[1]
    du[1:end] =  û[1:end] .+ abs.(p.K) .* e
end
params = ComponentVector{Float32}(vector_field_model = p, K = K)
prob_nn = ODEProblem(predictor!, u0 , tspan, params, saveat=tsteps)
soln_nn = Array(solve(prob_nn, Tsit5(), abstol = 1e-8, reltol = 1e-8, saveat = 1.0f0))

# Predict function
function prediction(p)
    _prob = remake(prob_nn, u0 = u0, p = p)
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
    Array(solve(_prob, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-8, reltol = 1e-8, saveat = 1.0f0, sensealg = sensealg))
end

# Loss function
function predloss(p)
    yh = prediction(p)
    e2 = mean(abs2, y .- yh[1,:])
    return e2
end

losses = Float32[]
callback = function (θ, l)
    push!(losses, predloss(θ))

    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

# Optimization to find the best hyperparameters
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> predloss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
@time res_ms = Optimization.solve(optprob, ADAM(), maxiters = 10000, verbose = false, callback  = callback)

# Predictions
y_pred = prediction(res_ms.u)
plot(y_pred[1,:])
plot!(y[1:end], label = "Training Data")

# Testing
## Generating data
y_test = X_test
tsteps_test = range(t_test[1], t_test[end], length = length(X_test))
tspan_test = (t_test[1], t_test[end])

y_zoh2 = ConstantInterpolation(y_test, tsteps_test)

function simulator!(du,u,p,t)
    û = U(u, p.vector_field_model, st)[1]
    du[1:end] =  û[1:end]
end

params_test = ComponentVector{Float32}(vector_field_model = p)
prob_test = ODEProblem(simulator!, u0 , tspan_test, params_test, saveat=tsteps_test)
prob = remake(prob_test, p = res_ms.u, tspan = tspan_test)
soln_nn = Array(solve(prob, Tsit5(), abstol = 1e-8, reltol = 1e-8, saveat = 1.0f0))

plot(y_pred[1,:],label = "Training Prediction", title = "Training and Test Predicitons of PEM Model", xlabel = "Time", ylabel = "Population")
plot!(soln_nn[1,:], label = "Test Prediction")
scatter!(t, real, label = "Training Data")
scatter!(t, real_new, label = "Test Data")
vline!(t[pred[end]], label = "Training/Test Split", color = :black)    


function plot_results(t, real, real_new,  pred, pred_new)
    plot(t, pred[1,:],label = "Training Prediction", title = "Training and Test Predicitons of PEM Model", xlabel = "Time", ylabel = "Population")
    plot!(t, pred_new[1,pred[end]:end], label = "Test Prediction")
    scatter!(t, real, label = "Training Data")
    scatter!(t, real_new, label = "Test Data")
    vline!(t[pred[end]], label = "Training/Test Split", color = :black)    
end