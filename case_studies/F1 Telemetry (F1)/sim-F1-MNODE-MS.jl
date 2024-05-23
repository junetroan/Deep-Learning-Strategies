#=

Simulation file for the Single Simulation of the MNODE-MS Model on the F1 telemetry data
The model is based on the proposed "multiple_shoot" method from the library DiffEqFlux in Julia, but modified to have an augmented state
Results were NOT used in the master's thesis of the author - "Novel Deep Learning Strategies for Time Series Forecasting"
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

# Data Cleaning, Normalization and Definition
t = 0.0:1.0:581
speed = data[:,1]
train_data = convert(Vector{Float32}, train[:,1])
test_data = convert(Vector{Float32}, test[:,1])
transformer = fit(ZScoreTransform, train_data)
X_train = StatsBase.transform(transformer, train_data)
X_test = StatsBase.transform(transformer, test_data)
t_test = convert(Vector{Float32}, collect(Int(round(split_ratio*size(data, 1))):size(data, 1)))
t_train = convert(Vector{Float32}, collect(1:Int(round(split_ratio*size(data, 1)))))
tspan = (t_train[1], t_train[end])
tsteps = range(tspan[1], tspan[2], length = length(X_train))
step = 1.0f0
datasize = size(X_train, 1)

#Generating random numbers
i = 1 # Number for random number generation
rng1 = StableRNG(i)
rng2 = StableRNG(i+2)

# Define the experimental parameters
groupsize = 5 # Number of points in each trajectory
predsize = 5 # Number of points to predict
state = 2 # Total number of states used for prediction - always one more than observed state, due to augmentation
continuity_term = 10.0 # Define the continuity factor that penalises the difference between the last state in the previous prediction and the current initial condition

# Simple neural network to predict system dynamics
U = Lux.Chain(Lux.Dense(state, 30, tanh),
Lux.Dense(30, state))

# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng1, U)

# Constructing the ODE Problem
params = ComponentVector{Float32}(vector_field_model = p)
u0 = vcat(X_train[1], randn(rng2, Float32, state - 1))
neuralode = NeuralODE(U, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = step, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
prob_node = ODEProblem((u,p,t) -> U(u, p, st)[1][1:end], u0, tspan, ComponentArray(p))

# Grouping the data into trajectories for multiple shooting
function group_ranges(datasize::Integer, groupsize::Integer)
    2 <= groupsize <= datasize || throw(DomainError(groupsize,
    "datasize must be positive and groupsize must to be within [2, datasize]"))
    return [i:min(datasize, i + groupsize - 1) for i in 1:(groupsize - 1):(datasize - 1)]
end

# Gathering trajectories and initial conditions
ranges = group_ranges(datasize, groupsize)
u0 = Float32(X_train[first(1:5)])
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
        solver, saveat = tsteps[rg],sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))) 
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
params = ComponentVector{Float32}(θ = p, u0_init = u0_init)
ls, ps = multiple_shoot_mod(params, X_train, tsteps, prob_node, loss_function, continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), groupsize; continuity_term)

# Running the prediction function utilising the multiple shooting method
multiple_shoot_mod(params, X_train, tsteps, prob_node, loss_function,
    continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), groupsize;
    continuity_term)

# Defining the loss function for the multiple shooting method
function loss_multiple_shoot(p)
    return multiple_shoot_mod(p, X_train, tsteps, prob_node, loss_function,
        continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), groupsize;
        continuity_term)[1]
end

# Gathering the predictions from the trained model and finding the loss
test_multiple_shoot = loss_multiple_shoot(params)
loss_single_shooting(params.θ)[1]

# Defining an array to store loss data for each iteration
losses = Float32[]

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
CSV.write("/sim-LV-MNODE-MS/Loss Data/Losses.csv", losses_df, writeheader = false)

# Gathering the predictions and loss from the trained model 
preds = predict_single_shooting(res_ms.u.θ)
loss_ms, _ = loss_single_shooting(res_ms.u.θ)

# Plotting the training results
function plot_results(real, pred, t)
    plot(t, pred[1,:], label = "Training Prediction", title="Iteration $i of Randomised MNODE-MS Model", xlabel="Time", ylabel="Population")
    plot!(t, real, label = "Training Data")
    plot!(legend=:bottomright)
    Plots.savefig("sim-F1-MNODE-MS/Plots/Training ANODE-MS II Model on F1 data.png.png")
end

plot_results(X_train, preds, t_train)

# Testing the model
# Defining the test problem
test_tspan = (t_test[1], t_test[end])
u0 = vcat(res_ms.u.u0_init[1,:])
prob_nn_updated = remake(prob_node, p = res_ms.u.θ, u0 = u0, tspan = test_tspan)

# Predicting the test data
prediction_new = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff = true)), abstol = 1e-6, reltol = 1e-6, saveat = 1.0f0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))

# Plotting the training and testing results
function plot_results(train_t, test_t, train_x, test_x, train_pred, test_pred)
    plot(train_t, train_pred[1,:], label = "Training Prediction", title="Training and Test Predictions of MNODE-MS Model", xlabel = "Time", ylabel = "Speed")
    plot!(test_t, test_pred[1,:], label = "Test Prediction")
    scatter!(train_t, train_x, label = "Training Data")
    scatter!(test_t, test_x, label = "Test Data")
    vline!([test_t[1]], label = "Training/Test Split")
    plot!(legend=:bottomleft)
    Plots.savefig("sim-F1-MNODE-MS/Plots/Training and testing of MNODE-MS Model on F1 data.png")
end

plot_results(t_train, t_test, X_train, X_test, preds, prediction_new)

