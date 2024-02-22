# PEM Financial System

#Test againt Julias multiple shooting
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
using StatsBase
using CSV, Tables, DataFrames
using DataInterpolations
using OrdinaryDiffEq
using OptimizationPolyalgorithms
using DiffEqFlux
using Plots

gr()

# Collecting data
data_path = "Multiple Shooting (MS)/ANODE-MS/Case Studies/financial_time_series.csv"
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
t_train = data[1:1500, 1]

X_test = StatsBase.transform(transformer, test[:, 1])
t_test = collect(Int(round(split_ration*size(data, 1))):size(data, 1))

# Interpolation of given data
tspan = (minimum(t_train), maximum(t_train))
tsteps = range(tspan[1], tspan[2], length = length(X_train)) 
y_zoh = ConstantInterpolation(X_train, tsteps)

# Definition of neural network
state = 2
rng1 = StableRNG(1112)
rng2 = StableRNG(1113)
rng3 = StableRNG(1114)

U = Lux.Chain(Lux.Dense(state, 30, tanh),Lux.Dense(30, state))
p, st = Lux.setup(rng1, U)
K = randn(rng2, state)
u0 =  vcat(X_train[1], randn(rng3, Float32, state - 1))

#Definition of the model 
function predictor!(du,u,p,t)
    û = U(u, p.vector_field_model, st)[1]
    yt = y_zoh(t)
    e = yt .- û[1]
    du[1:end] =  û[1:end] .+ abs.(p.K) .* e
end

params = ComponentVector{Float32}(vector_field_model = p, K = K)
#ps = getdata(params)
prob_nn = ODEProblem(predictor!, u0 , tspan, params, saveat=tsteps)
soln_nn = Array(solve(prob_nn, Tsit5(), abstol = 1e-8, reltol = 1e-8))

# Predict function
function prediction(p)
    _prob = remake(prob_nn, u0 = u0, p = p)
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
    Array(solve(_prob, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-8, reltol = 1e-8, sensealg = sensealg))
end

prediction(params)

# Loss function
function predloss(p)
    yh = prediction(p)
    e2 = mean(abs2, X_train .- yh[1,:])
    return e2
end

predloss(params)

losses = Float32[]
callback = function (θ, l)
    push!(losses, predloss(θ))

    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

#p0 = [0.7, 1.0] 
# Optimization to find the best hyperparameters
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> predloss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
@time res_ms = Optimization.solve(optprob, ADAM(), maxiters = 5000, callback = callback)

# Predictions
y_pred = prediction(res_ms.u)

plot(y_pred[1,:])
scatter!(X_train)

losses_df = DataFrame(loss = losses)
CSV.write("Multiple Shooting (MS)/ANODE-MS/Case Studies/Losses PEM Financial System.csv", losses_df, writeheader = false)

function plot_results(tp,real, pred)
    plot(tp, pred[1,:], label = "Training Prediction", title="Trained PEM Model predicting FinCompSys", xlabel = "Time", ylabel = "Population")
    plot!(tp, real, label = "Training Data")
    plot!(legend=:topright)
    savefig("Multiple Shooting (MS)/ANODE-MS/Case Studies/Plot PEM Financial System.png")
end

plot_results(t_train, X_train, y_pred)