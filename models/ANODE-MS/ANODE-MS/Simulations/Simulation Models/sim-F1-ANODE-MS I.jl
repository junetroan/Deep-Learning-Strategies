#sim-ANODE-MS-F1

using CSV, DataFrames, Plots, Statistics, StatsBase
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
gr()

data_path = "Multiple Shooting (MS)/ANODE-MS/Data/Telemetry Data - LEC Spain 2023 Qualifying.csv"
data_frame = CSV.read(data_path, DataFrame, header = true)
data = data_frame[:,2]

#Train/test Splits
split_ration = 0.25
train = data[1:Int(round(split_ration*size(data, 1))), :]
test = data[Int(round(split_ration*size(data, 1))):end, :]

# Data Cleaning and Normalization
train_data = convert(Vector{Float32}, train[:,1])
test_data = convert(Vector{Float32}, test[:,1])

transformer = fit(ZScoreTransform, data)
X_train = StatsBase.transform(transformer, train_data)
X_test = StatsBase.transform(transformer, test_data)
t_test = convert(Vector{Float32}, collect(Int(round(split_ration*size(data, 1))):size(data, 1)))
t_train = convert(Vector{Float32}, collect(1:Int(round(split_ration*size(data, 1)))))
tspan = (t_train[1], t_train[end])
tsteps = range(tspan[1], tspan[2], length = length(X_train))

# Define the experimental parameter
rng = StableRNG(1111)
groupsize = 5
predsize = 5
state = 2

fulltraj_losses = Float32[]

# NUMBER OF ITERATIONS OF THE SIMULATION
i = 1

#Generating random numbers
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
@time res_ms = Optimization.solve(optprob, ADAM(), callback=callback, maxiters = 5000)

losses_df = DataFrame(loss = losses)
#CSV.write("sim-F1-ANODE-MS/Loss Data/Losses $i.csv", losses_df, writeheader = false)

full_traj = predict_final(res_ms.u)
full_traj_loss = final_loss(res_ms.u)
push!(fulltraj_losses, full_traj_loss)

function plot_results(tp, real, pred)
    plot(tp, pred[1,:], label = "Training Prediction", title="Trained ANODE-MS I Model predicting F1 Telemetry", xlabel = "Time", ylabel = "Speed")
    plot!(tp, real, label = "Training Data")
    plot!(legend=:topright)
    savefig("Results/F1/Training ANODE-MS I Model on F1 data.png")
end

plot_results(t_train, X_train, full_traj)

test_tspan = (t_test[1], t_test[end])
predicted_u0_nn = U0_nn(nn_predictors[:, 1], res_ms.u.initial_condition_model, st0)[1]
u0_all = vcat(u0_vec[1], predicted_u0_nn)
prob_nn_updated = remake(prob_nn, p = res_ms.u, u0 = u0_all, tspan = test_tspan)
prediction_new = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff = true)),  abstol = 1f-6, reltol = 1f-6,
saveat =1.0f0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
#t1 = t_train |> collect
#t3 = t_test|> collect

test_loss = X_test - prediction_new[1, :]
total_test_loss = mean(abs2, test_loss)

function plot_results(train_t, test_t, train_x, test_x, train_pred, test_pred)
    plot(train_t, train_pred[1,:], label = "Training Prediction", title="Training and Test Predictions of ANODE-MS I Model", xlabel = "Time", ylabel = "Speed")
    plot!(test_t, test_pred[1,:], label = "Test Prediction")
    scatter!(train_t, train_x, label = "Training Data")
    scatter!(test_t, test_x, label = "Test Data")
    vline!([test_t[1]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("Results/F1/Training and testing of ANODE-MS I Model on F1 data.png")
end

plot_results(t_train, t_test, X_train, X_test, full_traj, prediction_new)