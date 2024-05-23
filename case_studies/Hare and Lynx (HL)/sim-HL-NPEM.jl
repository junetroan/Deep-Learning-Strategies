#=

Simulation file for the Single Simulation of the NPEM Model on the Hare and Lynx data
Results were used in the master's thesis of the author - "Novel Deep Learning Strategies for Time Series Forecasting"
Author: June Mari Berge Trøan (@junetroan)
Last updated: 2024-05-23

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

# Loading the data
data_path = "case_studies/Hare and Lynx (HL)/data/lynx_hare_data.csv"
data = CSV.read(data_path, DataFrame, header = true)

#Train/test Splits
split_ratio = 0.25
train = data[1:Int(round(split_ratio*size(data, 1))), :]
test = data[Int(round(split_ratio*size(data, 1))):end, :]

# Data Cleaning and Normalization
t = 0.0:1.0:581
speed = data[:,1]
train_data = convert(Vector{Float64}, train[:,2])
test_data = convert(Vector{Float64}, test[:,2])
transformer = fit(ZScoreTransform, train_data)
X_train = StatsBase.transform(transformer, train_data)
X_test = StatsBase.transform(transformer, test_data)
t_test = convert(Vector{Float64}, collect(Int(round(split_ratio*size(data, 1))):size(data, 1)))
t_train = convert(Vector{Float64}, collect(1:Int(round(split_ratio*size(data, 1)))))
tspan = (t_train[1], t_train[end])
tsteps = range(tspan[1], tspan[2], length = length(X_train))

# Interpolation of data
y_zoh = ConstantInterpolation(X_train, tsteps)

# Generating random numbers
i = 2 # Number for random number generation
rng1 = StableRNG(i+1)
rng2 = StableRNG(i+2)

# Defining the experimental parameters
state = 2 # Total number of states used for prediction - always one more than observed state, due to augmentation
K = rand(rng2, Float64, 2) # Randomly initialize the gain parameter

# Simple neural network to predict system dynamics
U = Lux.Chain(Lux.Dense(state, 30, tanh),Lux.Dense(30, state))

# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng1, U)

# Predictor function for training the model
function predictor!(du,u,p,t)
    û = U(u, p.vector_field_model, st)[1]
    yt = y_zoh(t)
    e = yt .- û[1]
    du[1:end] =  û[1:end] .+ abs.(p.K) .* e
end

# Constructing the ODE problem
u0 = [X_train[1], mean(X_train)]
params = ComponentVector{Float64}(vector_field_model = p, K = K)
prob_nn = ODEProblem(predictor!, u0 , tspan, params, saveat = 1.0)
soln_nn = Array(solve(prob_nn, Tsit5(), abstol = 1e-8, reltol = 1e-8, saveat = 1.0))

# Predictor function for training the model
function prediction(p)
    _prob = remake(prob_nn, u0 = u0, p = p)
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
    #Array(solve(_prob, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-8, reltol = 1e-8, saveat = tsteps , sensealg = sensealg))
    Array(solve(_prob, AutoVern7(KenCarp4(autodiff = true)), abstol = 1e-6, reltol = 1e-6, saveat = tsteps, sensealg = sensealg))
    
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
@time res_ms = Optimization.solve(optprob, ADAM(), maxiters = 5000, verbose = false, callback=callback) 
# Doesn't work at 5000 with AutoTsit5(Rosenbrock23(autodiff = true))- maxiters/stiffness problems reported. Set to 550, which works. AutoVern7(KenCarp4(autodiff = true)) works at 5000 iterations
# The abstol and reltol is also changed from 10e-8 to 10e-6

# Saving the loss data
losses_df = DataFrame(losses = losses)
CSV.write("sim-HL-PEM/Loss Data/Losses.csv", losses_df, writeheader = false)
            
# Gathering the predictions from the trained model
full_traj = prediction(res_ms.u)

# Plotting the training results
function plot_results(t, real, pred)
    plot(t, pred[1,:], label = "Training Prediction", title="Trained NPEM Model predicting Hare data", xlabel = "Time", ylabel = "Population")
    plot!(t, real, label = "Training Data")
    plot!(legend=:topleft)
    savefig("sim-HL-NPEM/Plots/Training NPEM Model on Hare and Lynx data.png")
end

plot_results(tsteps, X_train, full_traj)

# Testing the model
# Defining a simulation model for the testing of the model
function simulator!(du,u,p,t)
    û = U(u, p.vector_field_model, st)[1]
    du[1:end] =  û[1:end]
end

# Defining the test problem
all_data = vcat(X_train, X_test)
tspan_test = (t_test[1], t_test[end])
tsteps_test = range(tspan_test[1], tspan_test[2], length = length(X_test))
u0 = [X_test[1], mean(X_test)]
params_test = ComponentVector{Float64}(vector_field_model = p)
prob_test = ODEProblem(simulator!, u0 , tspan_test, params_test, saveat=tsteps_test)
prob = remake(prob_test, p = res_ms.u, tspan = tspan_test)
soln_nn = Array(solve(prob, AutoVern7(KenCarp4(autodiff=true)), abstol = 1e-6, reltol = 1e-6, saveat = 1.0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))

t1 = t_train |> collect
t3 = t_test |> collect

# Calculating the test loss
test_loss = X_test - soln_nn[1,:]
total_test_loss = abs(sum(abs, test_loss))

# Plotting the training and testing results
function plot_results(train_t, test_t, train_x, test_x, train_pred, test_pred)
    plot(train_t, train_pred[1,:], label = "Training Prediction", title="Training and Test Predictions of NPEM Model", xlabel = "Time", ylabel = "Population")
    plot!(test_t, test_pred[1,:], label = "Test Prediction")
    scatter!(train_t, train_x, label = "Training Data")
    scatter!(test_t, test_x, label = "Test Data")
    vline!([test_t[1]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("sim-HL-NPEM/Training and testing of NPEM Model on Hare and Lynx data.png")
end

plot_results(t_train, t_test, X_train, X_test, full_traj, soln_nn)