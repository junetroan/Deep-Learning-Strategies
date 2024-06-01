#=

Simulation file for the Single Simulation of the ANODE-MS I Model on the Lotka-Volterra data
Results were used in the master's thesis of the author - "Novel Deep Learning Strategies for Time Series Forecasting"
Author: June Mari Berge Trøan (@junetroan)
Last updated: 2024-05-24

=#

# Loading libraries
using Pkg
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots
gr()

# Function for the Lotka-Volterra model
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Generating the synthetic Lotka-Volterra data
rng = StableRNG(1111)
tspan = (0.0f0, 10.0f0)
u0 = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
@time solution = solve(prob, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)

using LaTeXStrings
using GR

# Set the font to Computer Modern
gr(fontfamily = "Computer Modern")

# Create the plot with specified font and fontsize
LV_plot = Plots.plot(solution, title="Synthetic Lotka-Volterra data", xlabel = "Time", ylabel = "Population", label = ["Prey" "Predator"], color = [:purple :orange], legendfontsize = 10, guidefontsize = 10, tickfontsize = 10)

# Save the plot
Plots.savefig("case_studies/Lotka-Volterra (LV)/Synthetic Lotka-Volterra data.png")

# Adding noise to the synthetic data
X = Array(solution)
t = solution.t
x̄ = mean(X, dims = 2)
noise_magnitude = 10f-3
noise_magnitude2 = 62.3f-2
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))

scatter(t, Xₙ[1,:], label = "Synthetic Data", title="Synthetic Lotka-Volterra data with noise", xlabel = "Time", ylabel = "Population", color="orange")


# Generating random numbers
i = 1 # Number for random number generation
rng1 = StableRNG(i)
rng2 = StableRNG(i+2)
rng3 = StableRNG(i+3)

# Defining the experimental parameters
state = 2
groupsize = 5
predsize = 5
datasize = size(Xₙ[1,:], 1)

# Simple neural network to predict Lotka-Volterra dynamics
U = Lux.Chain(Lux.Dense(state, 30, tanh), Lux.Dense(30, state))

# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, U)

# Simple neural network to predict initial points for use in multiple-shooting training
U0_nn = Lux.Chain(Lux.Dense(groupsize, 30, tanh), Lux.Dense(30, state-1))
p0, st0 = Lux.setup(rng, U0_nn)

# Defining the hybrid model
function ude_dynamics!(du, u, p, t, p_true)
    û = U(u, p.vector_field_model, st)[1] # Network prediction
    du[1:end] = û[1:end]
end

# Closure with the known parameter
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)

# Construct ODE Problem
x = Xₙ[1,:]
augmented_u0 = [[x[first(rg)]; fill(mean(x[rg]), state - 1)] for rg in ranges] 
params = ComponentVector{Float32}(vector_field_model = p, initial_condition_model = p0)
prob_nn = ODEProblem(nn_dynamics!, augmented_u0, tspan, params, saveat = 0.25f0)

# Grouping the data into trajectories for multiple shooting
function group_x(X::Vector, groupsize, predictsize, datasize)
    parent = [X[i: i + max(groupsize, predictsize) - 1] for i in 1:(groupsize-1):length(X) - max(groupsize, predictsize) + 1]
    parent = reduce(hcat, parent)
    targets = parent[1:groupsize,:]
    nn_predictors = parent[1:predictsize,:]
    u0 = parent[1, :]
    ranges = [i:min(datasize, i + groupsize - 1) for i in 1:(groupsize - 1):(datasize - 1)]
    return parent, targets, nn_predictors, u0, ranges
end

pas, targets, nn_predictors, u0_vec, ranges = group_x(Xₙ[1,:], groupsize, predsize, datasize)

# Prediction function utilising the multiple shooting method for training the model
function predict(θ)
    function prob_func(prob, i, repeat)
        u0_nn = U0_nn(nn_predictors[:, i], θ.initial_condition_model, st0)[1]
        u0_all = vcat(u0_vec[i], u0_nn)
        remake(prob, u0 = u0_all, tspan = (t[1], t[groupsize]))
        #remake(prob, u0 = u0_all, tspan = (t[first(rg)], t[last(rg)]) for rg in enumerate(ranges))
    end
    #sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))
    sensealg = ReverseDiffAdjoint()
    shooting_problem = EnsembleProblem(prob = prob_nn, prob_func = prob_func) 
    Array(solve(shooting_problem, verbose = false,  AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1f-6, reltol = 1f-6, 
    p=θ, trajectories = length(u0_vec), sensealg = sensealg))#[1,:,:]
end

# Loss function for the prediction function
function loss(θ)
    X̂ = predict(θ)
    continuity = mean(abs2, X̂[:, end, 1:end - 1] - X̂[:, 1, 2:end])
    prediction_error = mean(abs2, targets .- X̂[1,:,:])
    prediction_error + continuity*10f0
end

# Prediction function for the final model
function predict_final(θ)
    predicted_u0_nn = U0_nn(nn_predictors[:,1], θ.initial_condition_model, st0)[1]
    u0_all = vcat(u0_vec[1], predicted_u0_nn)
    prob_nn_updated = remake(prob_nn, p=θ, u0=u0_all) # no longer updates u0 nn
    X̂ = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff=true)), 
    abstol = 1f-6, reltol = 1f-6, 
    saveat = 0.25f0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
    X̂
end

# Loss function for the final prediction function
function final_loss(θ)
    X̂ = predict_final(θ)
    prediction_error = mean(abs2, Xₙ[1,:] .- X̂[1, :])
    prediction_error
end

# Defining an array to store loss data for each iteration
losses = Float32[]

# Callback that prints the current loss after every 50 iterations
callback = function (θ, l)
    push!(losses, final_loss(θ))

    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

# Training the model
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
@time res_ms = Optimization.solve(optprob, ADAM(), callback=callback, maxiters = 5000)

# Saving the loss data
losses_df = DataFrame(loss = losses)
CSV.write("sim-LV-ANODE-MS-I/Loss Data/Losses.csv", losses_df, writeheader = false)

# Gathering the predictions from the trained model and calculating the loss 
final_traj = predict_final(res_ms.u)[1,:]
full_traj_loss = final_loss(res_ms.u)
actual_loss = Xₙ[1,:] - final_traj
total_loss = abs(sum(actual_loss))

# Plotting the training results
function plot_results(tp,real, pred)
    plot(tp, pred[1,:], label = "Training Prediction", title="Trained ANODE-MS II Model predicting Lotka-Volterra data", xlabel = "Time", ylabel = "Population")
    plot!(tp, real, label = "Training Data")
    plot!(legend=:topleft)
    savefig("sim-LV-ANODE-MS-II/Plots/Training ANODE-MS II Model on Lotka-Volterra data.png")
end

plot_results(t, Xₙ[1,:], full_traj)

# Testing 
# Generating synthetic test data
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob_new = ODEProblem(lotka!, u0, (0.0f0, 40.0f0), p_)
@time solution_new = solve(prob_new, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)

# Defining the test pronlem
predicted_u0_nn = U0_nn(nn_predictors[:, 1], res_ms.u.initial_condition_model, st0)[1]
u0_all = vcat(u0_vec[1], predicted_u0_nn)
prob_nn_updated = remake(prob_nn, p = res_ms.u, u0 = u0_all, tspan = (0.0f0, 40.0f0))

# Predicting the test data
prediction_new = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff = true)),  abstol = 1f-6, reltol = 1f-6,
saveat = 0.25f0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
updated_obsgrid = 0.0:0.25:40.0

# Calculating the test loss
test_loss = solution_new[1,:] - prediction_new[1,:]
total_test_loss = sum(abs, test_loss)

# Plotting the training and testing results
function plot_results(t, real, pred, pred_new)
    plot(t1, pred[1,:], label = "Training Prediction", title="Training and Test Predictions of ANODE-MS II Model", xlabel = "Time", ylabel = "Population")
    plot!(t3, pred_new[1,41:123], label = "Test Prediction")
    scatter!(t1, real[1,:], label = "Training Data")
    scatter!(t3, solution_new[1,41:123], label = "Test Data")
    vline!([t1[41]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("sim-LV-ANODE-MS-I/Plots/ANODE-MS-II LV Training and Testing.png")
end

plot_results(t, Xₙ, full_traj, prediction_new)
