#=

Simulation file for the Single Simulation of the MNODE-MS Model on the Lotka-Volterra data
Results were used in the master's thesis of the author - "Novel Deep Learning Strategies for Time Series Forecasting"
Author: June Mari Berge Trøan (@junetroan)
Last updated: 2024-05-24

=#

# Loading libraries
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots
using DiffEqFlux
using Plots
using Distributions
using CSV, Tables, DataFrames
gr()

# Function for the Lotka-Volterra model
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Generating synthetic Lotka-Volterra data
tspan = (0.0f0, 10.0f0)
tsteps = 0.25f0
t_steps = 0.0f0:0.25f0:10.0f0
rng = StableRNG(1111)
u0 = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
@time solution = solve(prob, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)

# Adding noise to the synthetic data
X = Array(solution)
t = solution.t
x = X[1,:]
datasize = size(x, 1)
tspan = (0.0f0, 10.0f0) # Original (0.0f0, 10.0f0
tsteps = range(tspan[1], tspan[2]; length = datasize)
steps = 0.25f0

#=
x̄ = mean(X, dims = 2)
noise_magnitude = 10f-3
noise_magnitude2 = 62.3f-2
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))
x = Xₙ[1, :]
=#

# Generating random numbers
i = 1 # Number for random number 
rng1 = StableRNG(i)
rng2 = StableRNG(i+2)

# Defining the experimental parameters
groupsize = 5 # Number of points in each trajectory
state = 2 # Total number of states used for prediction - always one more than observed state, due to augmentation
continuity_term = 10.0 # Define the continuity factor that penalises the difference between the last state in the previous prediction and the current initial condition

# Simple neural network to predict system dynamics  
U = Lux.Chain(Lux.Dense(state, 30, tanh),
Lux.Dense(30, state))

# Retrieving the initial parameters and state variables of the model
p, st = Lux.setup(rng1, U)

# Constructing the ODE Problem
params = ComponentVector{Float32}(vector_field_model = p)
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
u0 = Float32(x[first(1:5)])
u0_init = [[x[first(rg)]; fill(mean(x[rg]), state - 1)] for rg in ranges] 
u0_init = mapreduce(permutedims, vcat, u0_init)

# Modified multiple shooting method
function multiple_shoot_mod(p, ode_data, tsteps, prob::ODEProblem, loss_function::F,
    continuity_loss::C, solver::SciMLBase.AbstractODEAlgorithm, groupsize::Integer;
    continuity_term::Real = 100, kwargs...) where {F, C}

    datasize = size(ode_data, 1)

    if groupsize < 2 || groupsize > datasize
        throw(DomainError(groupsize, "groupsize can't be < 2 or > number of data points"))
    end

    ranges = group_ranges(datasize, groupsize)

    sols = [solve(remake(prob_node; p = p.θ, tspan = (t_steps[first(rg)], t_steps[last(rg)]),
    u0 = p.u0_init[index, :]),
    solver; saveat = tsteps[rg]) 
    for (index, rg) in enumerate(ranges)]

    group_predictions = Array.(sols)


    loss = 0

    for (i, rg) in enumerate(ranges)
        u = x[rg] # TODO: make it generic for observed states > 1
        û = group_predictions[i][1, :]
        loss += loss_function(u, û)

        if i > 1
            # Ensure continuity between last state in previous prediHLtion
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
    l = loss_function(x, pred[1,:])
    return l, pred
end

# Gathering the prediction of the single shooting method
ls, ps = loss_single_shooting(params.vector_field_model)

# Constructing the ODE Problem
params = ComponentVector{Float32}(θ = p, u0_init = u0_init)
ls, ps = multiple_shoot_mod(params, x, tsteps, prob_node, loss_function, continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), groupsize; continuity_term)

# Modified multiple_shoot method 
multiple_shoot_mod(params, x, tsteps, prob_node, loss_function,
    continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), groupsize;
    continuity_term)

# Defining the loss function for the multiple shooting method
function loss_multiple_shoot(p)
    return multiple_shoot_mod(p, x, tsteps, prob_node, loss_function,
        continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), groupsize;
        continuity_term)[1]
end

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
@time res_ms = Optimization.solve(optprob, ADAM(),  maxiters = 5000, callback = callback) #callback  = callback,

# Saving the loss data
losses_df = DataFrame(loss = losses)
CSV.write("sim-LV-MNODE-MS/Loss Data/Losses.csv", losses_df, writeheader = false)

# Gathering the predictions and loss from the trained model 
loss_ms, _ = loss_single_shooting(res_ms.u.θ)
preds = predict_single_shooting(res_ms.u.θ)

# Plotting the training results
function plot_results(real, pred, t)
    plot(t, pred[1,:], label = "Training Prediction", title="Trained MNODE-MS Model predicting Lotka-Volterra data", xlabel="Time", ylabel="Population")
    plot!(t, real, label = "Training Data")
    plot!(legend=:topleft)
    savefig("sim-LV-MNODE-MS/Plots/Training MNODE-MS Model on Lotka Volterra data.png")
end

plot_results(x, preds, t)

# Testing the model
# Defining the test problem
t_train  = 0.0:0.25:10.0 |> collect
t1 = t_train |> collect
tsteps = 0.0:0.25:40.0
t3 = tsteps[41:123] |> collect
u0 = vcat(res_ms.u.u0_init[1,:])
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob_new = ODEProblem(lotka!, u0, (0.0f0, 40.0f0), p_)
@time solution_new = solve(prob_new, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)
x_test = solution_new[1,:]
prob_nn_updated = remake(prob_node, u0 = u0, tspan = (0.0f0, 40.0f0), p = res_ms.u.θ)

# Predicting the test data
prediction_new = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff = true)), abstol = 1e-6, reltol = 1e-6, saveat = 0.25f0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))

# Calculating the loss of the test data
test_loss = solution_new[1,:] - prediction_new[1,:]
total_test_loss = sum(abs, test_loss)

# Plotting the training and testing results
function plot_results(real, pred, pred_new)
    plot(t1, pred[1,:], label = "Training Prediction", title="Training and Test Predictions of MNODE-MS Model", xlabel = "Time", ylabel = "Population")
    plot!(t3, pred_new[1,41:123], label = "Test Prediction")
    scatter!(t1, real, label = "Training Data")
    scatter!(t3, solution_new[1,41:123], label = "Test Data")
    vline!([t3[1]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("sim-LV-MNODE-MS/Plots/MNODE-MS LV Training and Testing.png")
end

plot_results(x, preds, prediction_new)







