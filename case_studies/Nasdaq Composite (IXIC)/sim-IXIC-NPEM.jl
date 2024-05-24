#=

Simulation file for the Single Simulation of the NPEM Model on the F1 telemetry data
Model has not been run in the current configuration
Author: June Mari Berge Trøan (@junetroan)
Last updated: 2024-05-24

=#

# Loading libraries
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

# Collecting Data
data_path = "case_studies/Nasdaq Composite (IXIC)/data/^IXIC.csv"
data = CSV.read(data_path, DataFrame)
date = data[:, 1]
open_price = data[:, 2]



#Train/test Splits
split_ratio = 0.25
train = open_price[1:Int(round(split_ratio*size(open_price, 1))), :]
test = open_price[Int(round(split_ratio*size(open_price, 1))):end, :]

# Data Cleaning and Normalization
t_train = Float64.(collect(1:Int(round(split_ratio*size(open_price, 1)))))
t_test = Float64.(collect(Int(round(split_ratio*size(open_price, 1))):size(open_price, 1)))
transformer = fit(ZScoreTransform, open_price)
X_train = vec(Float64.(StatsBase.transform(transformer, train)))
X_test = vec(Float64.(StatsBase.transform(transformer, test)))
t = collect(1:size(data, 1))
t = Float64.(t)
tspan = (minimum(t_train), maximum(t_train))
tsteps = range(tspan[1], tspan[2], length = length(X_train))

# Interpolation of given data
y_zoh = LinearInterpolation(X_train, tsteps)

#Generating random numbers
i = 1 # Number of points in each trajectory
rng1 = StableRNG(i+1)
rng2 = StableRNG(i+2)

# Defining the experimental parameters
state = 2  # Total number of states used for prediction - always one more than observed state, due to augmentation
K = rand(rng2, Float32, 2) # Randomly initialize the gain parameter

# Simple neural network to predict system dynamics
U = Lux.Chain(Lux.Dense(state, 30, tanh),Lux.Dense(30, state))

# GEt the initial parameters and state variables of the model
p, st = Lux.setup(rng1, U)

# Constructing the ODE Problem
u0 = [X_train[1], 0]
params = ComponentVector{Float64}(vector_field_model = p, K = K)
prob_nn = ODEProblem(predictor!, u0 , tspan, params, saveat = 1.0)
soln_nn = Array(solve(prob_nn, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-8, reltol = 1e-8, saveat = 1.0))

# Predictor function for training the model
function predictor!(du,u,p,t)
    û = U(u, p.vector_field_model, st)[1]
    yt = y_zoh(t)
    e = yt .- û[1]
    du[1:end] =  û[1:end] .+ abs.(p.K) .* e
end

# Prediction function for gathering the predictions of the trained model
function prediction(p)
    _prob = remake(prob_nn, u0 = u0, p = p)
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
    Array(solve(_prob, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-8, reltol = 1e-8, saveat = tsteps , sensealg = sensealg))
end

# Loss function for training the model
function predloss(p)
    yh = prediction(p)
    e2 = mean(abs2, X_train .- yh[1,:])
    return e2
end
        
# Defining arrays to store loss data and K-value for each iteration
losses = Float64[]
K = []

# Callback function to store loss data and K-value for each iteration, in addition to printing the current loss
callback = function (p, l)
    push!(losses, predloss(p))
    push!(K, p.K[1:end])
        
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

# Training the model
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> predloss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
@time res_ms = Optimization.solve(optprob, ADAM(), maxiters = 5000, verbose = false, callback=callback)

# Saving the loss data
losses_df = DataFrame(losses = losses)
CSV.write("sim-IXIC-NPEM/Loss Data/Losses.csv", losses_df, writeheader = false)
    
# Gathering the predictions from the trained model
full_traj = prediction(res_ms.u)

# Plotting the training results
function plot_results(t, real, pred)
    plot(t, pred[1,:], label = "Training Prediction", title="Trained NPEM Model predicting IXIC data", xlabel = "Time", ylabel = "Opening Price")
    plot!(t, real, label = "Training Data")
    plot!(legend=:topleft)
    savefig("sim-IXIC-NPEM/Training NPEM Model on IXIC data.png")
end

plot_results(tsteps, X_train, full_traj)

# Testing the model
# Definining a simulation model for the testing of the model
function simulator!(du,u,p,t)
    û = U(u, p.vector_field_model, st)[1]
    du[1:end] =  û[1:end]
end

# Defining the test problem
y_test = X_test
tsteps_test = range(t_test[1], t_test[end], length = length(X_test))
tspan_test = (t_test[1], t_test[end])
params_test = ComponentVector{Float64}(vector_field_model = p)
prob_test = ODEProblem(simulator!, u0 , tspan_test, params_test, saveat=tsteps_test)
prob = remake(prob_test, p = res_ms.u, tspan = tspan_test)

# Predicting the test data
soln_nn = Array(solve(prob, Tsit5(), abstol = 1e-8, reltol = 1e-8, saveat = 1.0))

# Calculating the test loss
test_loss = y_test .- soln_nn[1,:]
actual_loss = mean(abs2, test_loss)

# Plotting the training and testing results
function plot_results(train_t, test_t, train_x, test_x, train_pred, test_pred)
    plot(train_t, train_pred[1,:], label = "Training Prediction", title="Training and Testing of NPEM Model", xlabel = "Time", ylabel = "Opening Price")
    plot!(test_t, test_pred[1,:], label = "Test Prediction")
    scatter!(train_t, train_x, label = "Training Data")
    scatter!(test_t, test_x, label = "Test Data")
    vline!([test_t[1]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("sim-IXIC-NPEM/Plots/Training and testing of NPEM Model on IXIC data.png")
end

plot_results(t_train, t_test, X_train, X_test, full_traj, soln_nn)
