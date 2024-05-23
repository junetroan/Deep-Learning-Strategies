#=

Simulation file for the Single Simulation of the MNODE-MS Model on the Hare and Lynx data
Results were used in the master's thesis of the author - "Novel Deep Learning Strategies for Time Series Forecasting"
Author: June Mari Berge Trøan (@junetroan)
Last updated: 2024-05-23

=#

# Loading libraries
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots
using DiffEqFlux
using Distributions
using CSV, Tables, DataFrames
using Random
using StatsBase
gr()

# Loading the data
data_path = "case_studies/Hare and Lynx (HL)/data/lynx_hare_data.csv"
data = CSV.read(data_path, DataFrame)

#Train/test Splits
split_ratio = 0.25
train = data[1:Int(round(split_ratio*size(data, 1))), :]
test = data[Int(round(split_ratio*size(data, 1))):end, :]

# Data Cleaning, Normalization and Definition
hare_data = data[:, 2]
lynx_data = data[:, 3]
transformer = fit(ZScoreTransform, hare_data)
X_train = Float64.(StatsBase.transform(transformer, train[:, 2]))
datasize = size(X_train, 1)
X_test = Float64.(StatsBase.transform(transformer, test[:, 2]))
t = Float64.(collect(1:size(data, 1)))
t_train = Float64.(collect(1:Int(round(split_ratio*size(data, 1)))))
t_test = Float64.(collect(Int(round(split_ratio*size(data, 1))):size(data, 1)))
tspan = (Float64(minimum(t_train)), Float64(maximum(t_train)))
tsteps = range(tspan[1], tspan[2], length = datasize)
step = 1.0f0
x = X_train

# Generating random numbers
i = 1 # Number for random number generation
rng1 = StableRNG(i)
rng2 = StableRNG(i+2)

# Defining the experimental parameters
group_size = 5 # Number of points in each trajectory
state = 2 # Total number of states used for prediction - always one more than observed state, due to augmentation
continuity_term = 10.0 # Define the continuity factor that penalises the difference between the last state in the previous prediction and the current initial condition

# Simple neural network to predict system dynamics
U = Lux.Chain(Lux.Dense(state, 30, tanh), Lux.Dense(30, state))

# Retrieving the initial parameters and state variables of the model
p, st = Lux.setup(rng1, U)

# Constructing the ODE Problem
params = ComponentVector{Float64}(vector_field_model = p)
u0 = vcat(X_train[1], randn(rng2, state - 1))
neuralode = NeuralODE(U, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = steps, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
prob_node = ODEProblem((u,p,t) -> U(u, p, st)[1][1:end], u0, tspan, ComponentArray(p))

# Grouping the data into trajectories for multiple shooting
function group_ranges(datasize::Integer, groupsize::Integer)
    2 <= groupsize <= datasize || throw(DomainError(groupsize,
    "datasize must be positive and groupsize must to be within [2, datasize]"))
    return [i:min(datasize, i + groupsize - 1) for i in 1:(groupsize - 1):(datasize - 1)]
end

# Gathering the trajectories and intial conditions
ranges = group_ranges(datasize, groupsize)
u0 = Float64(X_train[first(1:5)])
u0_init = [[X_train[first(rg)]; fill(mean(X_train[rg]), state - 1)] for rg in ranges] 
u0_init = mapreduce(permutedims, vcat, u0_init)

# Modified multiple shooting method 
function multiple_shoot_mod(p, ode_data, tsteps, prob::ODEProblem, loss_function::F,
    continuity_loss::C, solver::SciMLBase.AbstractODEAlgorithm, group_size::Integer;
    continuity_term::Real = 100, kwargs...) where {F, C}

    datasize = size(ode_data, 1)

    if group_size < 2 || group_size > datasize
        throw(DomainError(group_size, "group_size can't be < 2 or > number of data points"))
    end

    ranges = group_ranges(datasize, group_size)

    sols = [solve(remake(prob_node; p = p.θ, tspan = (tsteps[first(rg)], tsteps[last(rg)]),
        u0 = p.u0_init[index, :]),
        solver, saveat = tsteps[rg], sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))) 
        for (index, rg) in enumerate(ranges)]

    group_predictions = Array.(sols)


    loss = 0

    for (i, rg) in enumerate(ranges)
        u = X_train[rg] # TODO: make it generic for observed states > 1
        û = group_predictions[i][1, :]
        loss += loss_function(u, û)

        if i > 1
            # Ensure continuity between last state in previous prediction
            # and current initial condition in ode_data
            loss += continuity_term *
                    continuity_loss(group_predictions[i - 1][:, end], group_predictions[i][:, 1])
        end
    end

    return loss, group_predictions
end

# Define the loss function and continuity loss function
loss_function(data, pred) = sum(abs2, data - pred)
continuity_loss(uᵢ₊₁, uᵢ) = sum(abs2, uᵢ₊₁ - uᵢ)

# Defining the prediction function utilising single shooting method
predict_single_shooting(p) = Array(first(neuralode(u0_init[1,:],p,st)))
tester_pred_ss = predict_single_shooting(params.vector_field_model)

# Loss function for the single shooting method
function loss_single_shooting(p)
    pred = predict_single_shooting(p)
    l = loss_function(X_train, pred[1,:])
    return l, pred
end

# Gathering the prediction of the single shooting method
ls, ps = loss_single_shooting(params.vector_field_model)

# Defining the parameters for the prediction function utilising the multiple shooting method
params = ComponentVector{Float64}(θ = p, u0_init = u0_init)
ls, ps = multiple_shoot_mod(params, X_train, tsteps, prob_node, loss_function, continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), groupsize; continuity_term)

# Running the prediction function utilising the mulitple shooting method
multiple_shoot_mod(params, X_train, tsteps, prob_node, loss_function,
    continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), groupsize;
    continuity_term)


# Defining the loss function for the multiple shooting method
function loss_multiple_shoot(p)
    return multiple_shoot_mod(p, X_train, tsteps, prob_node, loss_function,
        continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), groupsize;
        continuity_term)[1]
end

# Defining an array to store loss data for each iteration
losses = Float64[]

# Callback that prints the loss after every 50 iterations to facilitate monitoring
callback = function (θ, l)
    push!(losses, loss_multiple_shoot(θ))
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

# Training the model
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shoot(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
@time res_ms = Optimization.solve(optprob, ADAM(),  maxiters = 5000, callback = callback) 

# Saving the loss data
losses_df = DataFrame(loss = losses)
CSV.write("sim-HL-MNODE-MS/Loss Data/Losses.csv", losses_df, writeheader = false)

# Gathering the predictions and loss from the trained model 
loss_ms, _ = loss_single_shooting(res_ms.u.θ)
preds = predict_single_shooting(res_ms.u.θ)

# Plotting the training results
function plot_results(real, pred, t)
    plot(t, pred[1,:], label = "Training Prediction", title="Trained MNODE-MS Model predicting Hare data", xlabel="Time", ylabel="Population")
    plot!(t, real, label = "Training Data")
    plot!(legend=:topleft)
    savefig("sim-HL-MNODE-MS/Plots/Training MNODE-MS Model on Hare and Lynx data.png")
end

plot_results(X_train, preds, t_train)

# Testing the Model
# Defining the test problem
tspan_test = (Float64(minimum(t_test)), Float64(maximum(t_test)))
tsteps_test = range(tspan_test[1], tspan_test[2], length = length(X_test))
u0 = vcat(res_ms.u.u0_init[1,:])
prob_nn_updated = remake(prob_node, u0 = u0, tspan = tspan_test, p = res_ms.u.θ)

# Predicting the test data
prediction_new = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff = true)), abstol = 1e-6, reltol = 1e-6, saveat = 1.0f0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))

# Calculating the loss of the test data
test_loss = X_test - prediction_new[1,:]
total_test_loss = sum(abs, test_loss)

# Plotting the training and testing results
function plot_results(real_train, real_test, train_pred, test_pred)
    plot(t1, train_pred[1,:], label = "Training Prediction", title="Training and Test Predictions of MNODE-MS Model", xlabel = "Time", ylabel = "Population")
    plot!(t3, test_pred[1,:], label = "Test Prediction")
    scatter!(t1, real_train, label = "Training Data")
    scatter!(t3, real_test, label = "Test Data")
    vline!([t3[1]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("sim-HL-MNODE-MS/Training and testing of MNODE-MS Model on Hare and Lynx data.png")
end

plot_results(X_train, X_test, preds, prediction_new)
