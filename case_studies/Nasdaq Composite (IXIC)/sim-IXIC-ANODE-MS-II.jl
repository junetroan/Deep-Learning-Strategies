#=

Simulation file for the Single Simulation of the ANODE-MS II Model on the historical Nasdaq Composite (IXIC) data
Results were used in the master's thesis of the author - "Novel Deep Learning Strategies for Time Series Forecasting"
Author: June Mari Berge Trøan (@junetroan)
Last updated: 2024-05-24

=#

# Loading libraries
using CSV, DataFrames, Plots, Statistics, StatsBase
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
gr()

# Loading the data
data_path = "case_studies/Nasdaq Composite (IXIC)/data/^IXIC.csv"
data = CSV.read(data_path, DataFrame)
date = data[:, 1]
open_price = data[:, 2]

#Train/test Splits
split_ratio = 0.25
train = open_price[1:Int(round(split_ratio*size(open_price, 1))), :]
test = open_price[Int(round(split_ratio*size(open_price, 1))):end, :]

# Data Cleaning, Normalization and Definition
t_train = collect(1:Int(round(split_ratio*size(open_price, 1))))
t_test = collect(Int(round(split_ratio*size(open_price, 1))):size(open_price, 1))
transformer = fit(ZScoreTransform, open_price)
X_train = vec(StatsBase.transform(transformer, train))
X_test = vec(StatsBase.transform(transformer, test))
t = collect(1:size(data, 1))
tspan = (minimum(t_train), maximum(t_train))

#Generating random numbers
i = 1 # Number for random number generation
rng = StableRNG(i-1)
rng1 = StableRNG(i)
rng2 = StableRNG(i+2)
rng3 = StableRNG(i+3)

# Define the experimental parameters
groupsize = 5 # Number of points in each trajectory
predsize = 5 # Number of points to predict
state = 2 # Total number of states used for prediction - always one more than observed state, due to augmentation

# Simple neural network to predict system dynamics
U = Lux.Chain(Lux.Dense(state, 30, tanh),
Lux.Dense(30, state))

# Retrieving the initial parameters and state variables of the model
p, st = Lux.setup(rng1, U)

# Simple neural network to predict initial points for use in multiple-shooting training
U0_nn = Lux.Chain(Lux.Dense(groupsize, 30, tanh), Lux.Dense(30, state - 1))
p0, st0 = Lux.setup(rng2, U0_nn)

# Defining the hybrid model
function ude_dynamics!(du, u, p, t)
    û = U(u, p.vector_field_model, st)[1] # Network prediction
    du[1:end] = û[1:end]
end

# Closure with the known parameter
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t)

# Construct ODE Problem
augmented_u0 = vcat(X_train[1], randn(rng3, state - 1))
params = ComponentVector{Float64}(vector_field_model = p, initial_condition_model = p0)
prob_nn = ODEProblem(nn_dynamics!, augmented_u0, tspan, params, saveat = t_train)

# Grouping the data into trajectories for multiple shooting
function group_x(X::Vector, groupsize, predictsize)
    parent = [X[i: i + max(groupsize, predictsize) - 1] for i in 1:(groupsize-1):length(X) - max(groupsize, predictsize) + 1]
    parent = reduce(hcat, parent)
    targets = parent[1:groupsize,:]
    nn_predictors = parent[1:predictsize,:]
    u0 = parent[1, :]
    return parent, targets, nn_predictors, u0
end

pas, targets, nn_predictors, u0_vec = group_x(X_train, groupsize, predsize)

# Prediction function utilising the multiple shooting method for training the model
function predict(θ)
    function prob_func(prob, i, repeat)
        u0_nn = U0_nn(nn_predictors[:, i], θ.initial_condition_model, st0)[1]
        u0_all = vcat(u0_vec[i], u0_nn)
        remake(prob, u0 = u0_all, tspan = (t_train[1], t_train[groupsize]))
    end
    sensealg = ReverseDiffAdjoint()
    shooting_problem = EnsembleProblem(prob = prob_nn, prob_func = prob_func) 
    Array(solve(shooting_problem, verbose = false,  AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-6, reltol = 1e-6, 
    p=θ, saveat = t_train, trajectories = length(u0_vec), sensealg = sensealg))
end

# Loss function for the prediction function
function loss(θ)
    X̂ = predict(θ)
    continuity = mean(abs2, X̂[:, end, 1:end - 1] - X̂[:, 1, 2:end])
    prediction_error = mean(abs2, targets .- X̂[1,:,:])
    prediction_error + continuity*10.0
end

# Final prediction function for the trained model
function predict_final(θ)
    predicted_u0_nn = U0_nn(nn_predictors[:,1], θ.initial_condition_model, st0)[1]
    u0_all = vcat(u0_vec[1], predicted_u0_nn)
    prob_nn_updated = remake(prob_nn, p=θ, u0=u0_all) # no longer updates u0 nn
    X̂ = Array(solve(prob_nn_updated, AutoTsit5(Rosenbrock23()), abstol = 1e-6, reltol = 1e-6, 
    saveat = t_train, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
    X̂
end

# Loss function for the final prediction function
function final_loss(θ)
    X̂ = predict_final(θ)
    prediction_error = mean(abs2, X_train .- X̂[1, :])
    prediction_error
end

# Defining an array to store loss data for each iteration
losses = Float64[]

# Callback that prints the current loss every 50 iterations to facilitate monitoring
callback = function (θ, l)

    push!(losses, final_loss(θ))
    
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

# Pre-training the model
adtype = Optimization.AutoZygote()  
optf = Optimization.OptimizationFunction((x,p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
@time res_ms = Optimization.solve(optprob, ADAM(), callback=callback, maxiters = 5000)

# Saving the loss data
losses_df = DataFrame(loss = losses)
CSV.write("sim-IXIC-ANODE-MS-II/Loss Data/Losses.csv", losses_df, writeheader = false)

# Gathering the predictions from the pre-trained model and calculating the loss
full_traj = predict_final(res_ms.u)
full_traj_loss = final_loss(res_ms.u)

# Training the model 
optf_final = Optimization.OptimizationFunction((x,p) -> final_loss(x), adtype)
optprob_final = Optimization.OptimizationProblem(optf_final, res_ms.u)
@time res_final = Optimization.solve(optprob_final, BFGS(initial_stepnorm = 0.01), callback=callback, maxiters = 1000, allow_f_increases = true)

# Gathering the predictions from the trained model and calculating the loss
full_traj2 = predict_final(res_final.u)
actual_loss = X_train .- full_traj2[1, :]
total_loss = mean(abs2, actual_loss)

# Plotting the training results
function plot_results(tp, real, pred)
    plot(tp, pred[1,:], label = "Training Prediction", title="Trained ANODE-MS II Model predicting IXIC data", xlabel = "Time", ylabel = "Opening Price")
    plot!(tp, real, label = "Training Data")
    plot!(legend=:topleft)
    savefig("sim-IXIC-ANODE-MS-II/Training ANODE-MS II Model on IXIC data.png")
end

plot_results(t_train, X_train, full_traj2)

# Testing the model
# Defining the test problem
test_tspan = (t_test[1], t_test[end])
predicted_u0_nn = U0_nn(nn_predictors[:, 1], res_final.u.initial_condition_model, st0)[1]
u0_all = vcat(u0_vec[1], predicted_u0_nn)
prob_nn_updated = remake(prob_nn, p = res_final.u, u0 = u0_all, tspan = test_tspan)

# Predicting the test data
prediction_new = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff = true)),  abstol = 1e-6, reltol = 1e-6,
saveat =1.0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))

# Calculating the loss of the test data
test_loss = X_test - prediction_new[1, :]
total_loss_test = mean(abs2, test_loss)

# Plotting the training and test results
function plot_results(train_t, test_t, train_x, test_x, train_pred, test_pred)
    plot(train_t, train_pred[1,:], label = "Training Prediction", title="Training and Test Predictions of ANODE-MS II Model", xlabel = "Time", ylabel = "Opening Price")
    plot!(test_t, test_pred[1,:], label = "Test Prediction")
    scatter!(train_t, train_x, label = "Training Data")
    scatter!(test_t, test_x, label = "Test Data")
    vline!([test_t[1]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("sim-IXIC-ANODE-MS-II/Plots/Training and testing of ANODE-MS II Model on IXIC data.png")
end

plot_results(t_train, t_test, X_train, X_test, full_traj2, prediction_new)