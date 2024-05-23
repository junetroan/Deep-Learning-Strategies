#=

Simulation file for the Single Simulation of the ANODE-MS II Model on the F1 telemetry data
Model has not been run in the current configuration
Author: June Mari Berge Trøan (@junetroan)
Last updated: 2024-05-23

=#

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
using PlotlyKaleido

PlotlyKaleido.start()
plotly()

# Loading the data
data_path = "case_studies/F1 Telemetry (F1)/data/telemetry_data_LEC_spain_2023_qualifying.csv"
all_data = CSV.read(data_path, DataFrame, header = true)
data = all_data[:, 2] # Loading the speed telemetry

#Train/test Splits
split_ratio = 0.25
train = data[1:Int(round(split_ratio*size(data, 1))), :]
test = data[Int(round(split_ratio*size(data, 1))):end, :]

# Data Cleaning and Normalization
train_data = convert(Vector{Float32}, train[:,1])
test_data = convert(Vector{Float32}, test[:,1])
transformer = fit(ZScoreTransform, train_data)
X_train = StatsBase.transform(transformer, train_data)
X_test = StatsBase.transform(transformer, test_data)
t_test = convert(Vector{Float32}, collect(Int(round(split_ratio*size(data, 1))):size(data, 1)))
t_train = convert(Vector{Float32}, collect(1:Int(round(split_ratio*size(data, 1)))))
tspan = (t_train[1], t_train[end])
tsteps = range(tspan[1], tspan[2], length = length(X_train))

# Interpolation of the data
y_zoh = ConstantInterpolation(X_train, tsteps)

#Generating random numbers
i = 1 # Number of points in each trajectory
rng1 = StableRNG(i+1)
rng2 = StableRNG(i+2)

# Defining the experimental parameters
state = 2  # Total number of states used for prediction - always one more than observed state, due to augmentation
K = rand(rng2, Float32, 2) # Randomly initialize the gain parameter

# Simple neural network to predict system dynamics
U = Lux.Chain(Lux.Dense(state, 30, tanh),Lux.Dense(30, state))

# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng1, U)

# Constructing the ODE Problem
u0 = [X_train[1], 0] 
params = ComponentVector{Float32}(vector_field_model = p, K = K)
prob_nn = ODEProblem(predictor!, u0 , tspan, params, saveat = 1.0f0 )
soln_nn = Array(solve(prob_nn, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-8, reltol = 1e-8, saveat = 1.0f0))

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
    Array(solve(_prob, AutoVern7(KenCarp4(autodiff=true)), abstol = 1e-6, reltol = 1e-6, saveat = tsteps , sensealg = sensealg))
end

# Loss function for training the model
function predloss(p)
    yh = prediction(p)
    e2 = mean(abs2, X_train .- yh[1,:])
    return e2
end
        
# Defining arrays to store loss data and K-value for each iteration
losses = Float32[]
K = []

# Callback function to store loss data and K-value for each iteration, in addition to prints the loss after every 50 iterations to facilitate monitoring
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
@time res_ms = Optimization.solve(optprob, ADAM(), maxiters = 5000, verbose = false, callback=callback) #5000 iterations doesn't work???? stiffness issues. Stopped at 3300, therefore, sat to 3000
# Doesn't work at 5000 with AutoTsit5(Rosenbrock23(autodiff = true))- maxiters/stiffness problems reported. Set to 550, which works. AutoVern7(KenCarp4(autodiff = true)) works at 5000 iterations
# The abstol and reltol is also changed from 10e-8 to 10e-6

# Saving the loss data
losses_df = DataFrame(losses = losses)
CSV.write("sim-F1-PEM/Loss Data/Losses.csv", losses_df, writeheader = false)

# Gathering the predictions from the trained model
full_traj = prediction(res_ms.u)

# Plotting the training results
function plot_results(t, real, pred)
    plot(t, pred[1,:], label = "Training Prediction", title="Trained NPEM Model predicting F1 Telemetry", xlabel = "Time", ylabel = "Speed")
    plot!(t, real, label = "Training Data")
    plot!(legend=:bottomright)
    Plots.savefig("sim-F1-NPEM/Plots/Training NPEM Model on F1 data.png")
end

plot_results(tsteps, X_train, full_traj)

# Testing the model
# Defining a simulation model for the testing of the model
function simulator!(du,u,p,t)
    û = U(u, p.vector_field_model, st)[1]
    du[1:end] =  û[1:end]
end

# Defining the test problem
y_test = X_test
tsteps_test = range(t_test[1], t_test[end], length = length(X_test))
tspan_test = (t_test[1], t_test[end])
params_test = ComponentVector{Float32}(vector_field_model = p)
prob_test = ODEProblem(simulator!, u0 , tspan_test, params_test, saveat=tsteps_test)
prob = remake(prob_test, p = res_ms.u, tspan = tspan_test)
soln_nn = Array(solve(prob, Tsit5(), abstol = 1e-8, reltol = 1e-8, saveat = 1.0f0))

# Plotting the training and testing results
function plot_results(train_t, test_t, train_x, test_x, train_pred, test_pred)
    plot(train_t, train_pred[1,:], label = "Training Prediction", title="Training and Test Predictions of PEM Model", xlabel = "Time", ylabel = "Speed")
    plot!(test_t, test_pred[1,:], label = "Test Prediction")
    scatter!(train_t, train_x, label = "Training Data")
    scatter!(test_t, test_x, label = "Test Data")
    vline!([test_t[1]], label = "Training/Test Split")
    plot!(legend=:topright)
    Plots.savefig("sim-F1-NPEM/Plots/Training and testing of NPEM Model on F1 data.png")
end

plot_results(t_train, t_test, X_train, X_test, full_traj, soln_nn)