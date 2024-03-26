using CSV, DataFrames, Plots, Statistics, StatsBase
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
gr()

data_path = "Multiple Shooting (MS)/ANODE-MS/Data/Telemetry Data - VER Spain 2023 Qualifying.csv"
data = CSV.read(data_path, DataFrame, header = true)

#Train/test Splits
distance = convert(Vector{Float32}, data[:,1])
speed = convert(Vector{Float32}, data[:,2])

split_ration = 0.7

distance_train = distance[1:Int(round(split_ration*size(data, 1)))]
distance_test = distance[Int(round(split_ration*size(data, 1))):end]
speed_train = speed[1:Int(round(split_ration*size(data, 1)))]
speed_test = speed[Int(round(split_ration*size(data, 1))):end]

# Data Cleaning and Normalization
t = Float32.(0.0:1.0:size(data, 1)-1)
t_train = Float32.(t[1:Int(round(split_ration*size(data, 1)))])
t_test = Float32.(t[Int(round(split_ration*size(data, 1))):end])
tspan_train = (t_train[1], t_train[end])
tspan_test = (t_test[1], t_test[end])
tsteps_train = range(tspan_train[1], tspan_train[2], length = length(distance_train))
tsteps_test = range(tspan_test[1], tspan_test[2], length = length(distance_test))

transformer_distance = fit(ZScoreTransform, distance_train)
d_train = StatsBase.transform(transformer_distance, distance_train)
d_test = StatsBase.transform(transformer_distance, distance_test)

transformer_speed = fit(ZScoreTransform, speed_train)
s_train = StatsBase.transform(transformer_speed, speed_train)
s_test = StatsBase.transform(transformer_speed, speed_test)

y_train = [s_train]'
y_test = [s_test]'

rng = StableRNG(1111)
groupsize = 5
predsize = 5
state = 2
i = 1000

rng1 = StableRNG(i)
rng2 = StableRNG(i+2)
rng3 = StableRNG(i+3)

# Simple NN to predict dynamics

U = Lux.Chain(Lux.Dense(state, 30, tanh),
Lux.Dense(30, state))

# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng1, U)

# Simple NN to predict initial points for use in multiple-shooting training
U0_nn = Lux.Chain(Lux.Dense(groupsize, 30, tanh), Lux.Dense(30, state - 1))
p0, st0 = Lux.setup(rng2, U0_nn)

# Define the hybrid model
function ude_dynamics!(du, u, p, t)
    û = U(u, p.vector_field_model, st)[1] # Network prediction
    du[1:end] = û[1:end]
end

# Closure with the known parameter
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t)

# Construct ODE Problem
augmented_u0 = vcat(X_train[1], randn(rng3, Float32, state - 1))
params = ComponentVector{Float32}(vector_field_model = p, initial_condition_model = p0)
prob_nn = ODEProblem(nn_dynamics!, augmented_u0, tspan, params, saveat = t_train)

function group_x(X::Vector, groupsize, predictsize)
    parent = [X[i: i + max(groupsize, predictsize) - 1] for i in 1:(groupsize-1):length(X) - max(groupsize, predictsize) + 1]
    parent = reduce(hcat, parent)
    targets = parent[1:groupsize,:]
    nn_predictors = parent[1:predictsize,:]
    u0 = parent[1, :]
    return parent, targets, nn_predictors, u0
end

pas, targets, nn_predictors, u0_vec = group_x(X_train, groupsize, predsize)

function predict(θ)
    function prob_func(prob, i, repeat)
        u0_nn = U0_nn(nn_predictors[:, i], θ.initial_condition_model, st0)[1]
        u0_all = vcat(u0_vec[i], u0_nn)
        remake(prob, u0 = u0_all, tspan = (t_train[1], t_train[groupsize]))
    end
    sensealg = ReverseDiffAdjoint()
    shooting_problem = EnsembleProblem(prob = prob_nn, prob_func = prob_func) 
    Array(solve(shooting_problem, verbose = false,  AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1f-6, reltol = 1f-6, 
    p=θ, saveat = t_train, trajectories = length(u0_vec), sensealg = sensealg))
end

function loss(θ)
    X̂ = predict(θ)
    continuity = mean(abs2, X̂[:, end, 1:end - 1] - X̂[:, 1, 2:end])
    prediction_error = mean(abs2, targets .- X̂[1,:,:])
    prediction_error + continuity*10f0
end

function predict_final(θ)
    predicted_u0_nn = U0_nn(nn_predictors[:,1], θ.initial_condition_model, st0)[1]
    u0_all = vcat(u0_vec[1], predicted_u0_nn)
    prob_nn_updated = remake(prob_nn, p=θ, u0=u0_all) # no longer updates u0 nn
    X̂ = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff=true)), abstol = 1f-6, reltol = 1f-6, 
    saveat = t_train, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
    X̂
end

function final_loss(θ)
    X̂ = predict_final(θ)
    prediction_error = mean(abs2, X_train .- X̂[1, :])
    prediction_error
end

losses = Float32[]

callback = function (θ, l)
    push!(losses, final_loss(θ))      
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()  
optf = Optimization.OptimizationFunction((x,p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
res_ms = Optimization.solve(optprob, ADAM(), callback=callback, maxiters = 5000)

losses_df = DataFrame(loss = losses)
CSV.write("sim-F1-ANODE-MS/Loss Data/Losses $i.csv", losses_df, writeheader = false)
        
full_traj = predict_final(res_ms.u)
full_traj_loss = final_loss(res_ms.u)
push!(fulltraj_losses, full_traj_loss)

function predict_test(θ)
    pred_u0_nn = U0_nn(s_train, θ.initial_condition_model, st0)[1]
    u0_all = vcat(u0_vec[1], pred_u0_nn)
    prob_nn_updated = remake(prob_nn, p=θ, u0=u0_all)

    X̂ = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff=true)), abstol = 1f-6, reltol = 1f-6, 
    saveat = t_test, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
    X̂
end

test_pred = predict_test(res_ms.u)
test = test_pred[2,:]
t1 = t_train
t3 = t_test # Check

function plot_results(t1, t3, pred, pred_new, real, sol_new)
    plot(t1, pred[1,:], label = "Training Prediction", title="Trained ANODE-MS Model prediction F1 telemetry", xlabel = "Time", ylabel = "Speed")
    plot!(t3, pred_new[1,41:123], label = "Test Prediction")
    scatter!(t1, real[1,:], label = "Training Data")
    scatter!(t3, sol_new[1,41:123], label = "Test Data")
    vline!([t1[41]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("Results/ANODE-MS F1 Training and Testing.png")
end