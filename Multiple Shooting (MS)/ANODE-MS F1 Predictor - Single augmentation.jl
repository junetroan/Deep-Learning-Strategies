#ANODE-MS F1 Predictor
using CSV, DataFrames, Plots, Statistics, StatsBase
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
using DataInterpolations
using PlotlyKaleido
# Set the backend for Plots to Plotly

PlotlyKaleido.start()
plotly()


data_path = "Multiple Shooting (MS)/ANODE-MS/Data/Telemetry Data - VER Spain 2023 Qualifying.csv"
data = CSV.read(data_path, DataFrame, header = true)

#Train/test Splits
distance = convert(Vector{Float32}, data[:,1])
speed = convert(Vector{Float32}, data[:,2])
throttle = convert(Vector{Float32}, data[:,3])
brake = convert(Vector{Float32}, data[:,4])

split_ratio = 0.25

distance_train = distance[1:Int(round(split_ratio*size(data, 1)))]
distance_test = distance[Int(round(split_ratio*size(data, 1))):end]
speed_train = speed[1:Int(round(split_ratio*size(data, 1)))]
speed_test = speed[Int(round(split_ratio*size(data, 1))):end]
throttle_train = throttle[1:Int(round(split_ratio*size(data, 1)))]
throttle_test = throttle[Int(round(split_ratio*size(data, 1))):end]
brake_train = brake[1:Int(round(split_ratio*size(data, 1)))]
brake_test = brake[Int(round(split_ratio*size(data, 1))):end]

# Data Cleaning and Normalization
t = Float32.(0.0:1.0:size(data, 1)-1)
t_train = Float32.(t[1:Int(round(split_ratio*size(data, 1)))])
t_test = Float32.(t[Int(round(split_ratio*size(data, 1))):end])
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

transformer_throttle = fit(ZScoreTransform, throttle_train)
th_train = StatsBase.transform(transformer_throttle, throttle_train)
th_test = StatsBase.transform(transformer_throttle, throttle_test)

transformer_brake = fit(ZScoreTransform, brake_train)
b_train = StatsBase.transform(transformer_brake, brake_train)
b_test = StatsBase.transform(transformer_brake, brake_test)

d_train_interpolation = LinearInterpolation(d_train, t_train)
s_train_interpolation = LinearInterpolation(s_train, t_train)
th_train_interpolation = LinearInterpolation(th_train, t_train)
b_train_interpolation = LinearInterpolation(b_train, t_train)

y_train = [s_train th_train b_train]'
y_test = [s_test th_test b_test]'

# Define the experimental parameter
unknown_states = 1
known_states  = 3
groupsize = 5
predsize = 5
obsgrid = 5:5:length(t_train)

state = unknown_states + known_states
i = 1000
rng1 = StableRNG(i)
rng2 = StableRNG(i+2)
rng3 = StableRNG(i+3)

# Define the neural network
U = Lux.Chain(Lux.Dense(unknown_states+known_states, 30, tanh), Lux.Dense(30, 2))

# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng1, U)

# Simple NN to predict initial points for use in multiple-shooting training
U0_nn = Lux.Chain(Lux.Dense(groupsize, 30, tanh), Lux.Dense(30,  1))
p0, st0 = Lux.setup(rng2, U0_nn)

throttle_interpolation = LinearInterpolation(th_train, t_train, extrapolate = true)
brake_interpolation = ConstantInterpolation(b_train, t_train, extrapolate = true)

#=
# Define the hybrid model
function ude_dynamics!(du, u, p, t)
    û = U(u, p.vector_field_model, st)[1] # Network prediction
    th = th(t)
    br = br(t)
    du[1:end] = û[1:end]
end
=#
function ude_dynamics!(du, u, p, t)

    s = u[1]
    h = u[2]

    th = throttle_interpolation(t)
    br = brake_interpolation(t)

    inputs = [s;h;th;br]

    # Evaluate neural network to get derivatives
    derivatives = U(inputs, p.vector_field_model, st)[1]

    # Assign derivatives to the output array `du`
    du[1] = derivatives[1]
    du[2] = derivatives[2]
end

# Closure with the known parameter

nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t)

# Construct ODE Problem
rands = randn(rng3, Float32, unknown_states)
augmented_u0 = vcat(y_train[1,1], rands)
params = ComponentVector{Float32}(vector_field_model = p, initial_condition_model = p0)
prob_nn = ODEProblem(nn_dynamics!, augmented_u0, tspan_train, params, saveat = t_train)

# with 0.25: 36
# with 0.75: 108

# TODO: Make number of trajectories follow from parent to firsts
function group_x(xdim, y , groupsize, predictsize)
    parent = [y[:,i: i + max(groupsize, predictsize) - 1] for i in 1:(groupsize-1):length(xdim) - max(groupsize, predictsize) + 1]
    #trajs = size(parent, 3)
    firsts =  hcat([parent[i][1, :] for i in 1:36]...) # Speed
    parent = cat(parent..., dims=3)
    u0 = firsts[1,:]
    return parent, firsts, u0
end

pas, first_series, u0_vec = group_x(t_train, y_train, groupsize, predsize)

#plotly()
plot(s_train, label = "Speed")
#plot(th_train, label = "Throttle")
#plot(b_train, label = "Brake")

plot(s_train, label = "Speed", title = "Training Data", xlabel = "Time", ylabel = "Speed")
#discont = [68, 89, 172, 199, 235, 254, 301, 316, 403, 429]

discont = [68, 89]
#108
function tpredictor(θ)
    function prob_func(prob, i, repeat)
        u0_nn_first = U0_nn(first_series[:, i], θ.initial_condition_model, st0)[1]
        u0_all = vcat(u0_vec[i], u0_nn_first)
        remake(prob, u0 = u0_all, tspan = (t[1], t[groupsize]))
    end
    sensealg = ReverseDiffAdjoint()
    shooting_problem = EnsembleProblem(prob = prob_nn, prob_func = prob_func) 
    Array(solve(shooting_problem, verbose = false, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1f-6, reltol = 1f-6, 
    p=θ, saveat = t_train, trajectories = 36 , sensealg = sensealg, tstops = discont))
end

tester = tpredictor(params)


function loss(θ)
    X̂ = tpredictor(θ)
    continuity = mean(abs2, X̂[:, end, 1:end - 1] - X̂[:, 1, 2:end])
    prediction_error = mean(abs2, pas[1,:,:] .- X̂[1,:,:])
    prediction_error + continuity*10f0
end

lossezzz = loss(params)

function predict_final(θ)
    u0_nn_first = U0_nn(first_series[:, 1], θ.initial_condition_model, st0)[1]
    u0_all = vcat(u0_vec[1], u0_nn_first)
    prob_nn_updated = remake(prob_nn, p = θ, u0 = u0_all)

   # no longer updates u0 nn
    X̂ = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff=true)), abstol = 1f-6, reltol = 1f-6, 
    saveat = t_train, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),tstops = discont))
    X̂
end

final_preds = predict_final(params) #INFINITY?????

function final_loss(θ)
    X̂ = predict_final(θ)
    prediction_error = mean(abs2, y_train[1,:] .- X̂[1,:])
    prediction_error
end

final_loss(params)

losses = Float32[]

callback = function (θ, l)
    push!(losses, final_loss(θ))
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()  
optf = Optimization.OptimizationFunction((x,p) -> final_loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
@time res_ms = Optimization.solve(optprob, ADAM(), callback=callback, maxiters = 5000) # RUNS ONLY FROM INFINITY....

#WORKING UNTIL HERE 💕🤓🥺

losses_df = DataFrame(loss = losses)
CSV.write("Results/F1/Loss Data/ANODE-MS F1 Single Augmentation Loss - 16.05.24.csv", losses_df, writeheader = false)

full_traj = predict_final(res_ms.u)
full_traj_loss = final_loss(res_ms.u)

plot(full_traj[1,:], label = "Prediction")
plot!(brake_train, label = "Data")
plot!(y_train[1,:], label = "Data")

optf_final = Optimization.OptimizationFunction((x,p) -> final_loss(x), adtype)
optprob_final = Optimization.OptimizationProblem(optf_final, res_ms.u)
@time res_final = Optimization.solve(optprob_final, BFGS(initial_stepnorm = 0.01), callback=callback, maxiters = 1000, allow_f_increases = true)

full_traj2 = predict_final(res_final.u)
actual_loss = y_train[1,:] - full_traj2[1,:]
plot(full_traj2[1,:], label = "Prediction")
plot!(y_train[1,:], label = "Data")
total_loss = abs(sum(actual_loss))

plotly()
function plot_training(tp, real, pred)
    plot(tp, pred, label = "Training Prediction", title="Trained ANODE-MS Model predicting F1 data", xlabel = "Time", ylabel = "Speed")
    plot!(tp, real, label = "Training Data")
    plot!(legend=:bottomright)
    Plots.savefig("Results/F1/ANODE-MS II Single Augmentation on F1 data.png")
end

plot_training(t_train, y_train[1,:], full_traj2[1,:])


######################################################################################
# Testing
y_train
y_test 
t_test
tspan_test = (t_test[1], t_test[end])
tsteps_test = range(tspan_test[1], tspan_test[2], length = length(y_test[1,:]))
predicted_u0_nn = U0_nn(first_series[:, 1], res_final.u.initial_condition_model, st0)[1]
u0_all = vcat(u0_vec[1], predicted_u0_nn)
prob_nn_updated = remake(prob_nn, p = res_final.u, u0 = u0_all, tspan = tspan_test)
prediction_new = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff = true)),  abstol = 1e-6, reltol = 1e-6,
saveat = 1.0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))

t1 = t_train |> collect
t3 = t_test |> collect
t = t |> collect

test_loss = y_test[1,:] - prediction_new[1,:]
total_test_loss = sum(abs, test_loss)

function plot_results(real_train, real_test, pred, pred_new)
    plot(t1, pred[1,:], label = "Training Prediction", title="Training and Test Predictions of ANODE-MS II Model", xlabel = "Time", ylabel = "Speed")
    plot!(t3, pred_new[1,:], label = "Test Prediction")
    scatter!(t1, real_train, label = "Training Data")
    scatter!(t3, real_test, label = "Test Data")
    vline!([t3[1]], label = "Training/Test Split")
    plot!(legend=:bottomright)
    Plots.savefig("Results/F1/Training and Testing of ANODE-MS II Model on F1 data.png")
end

plot_results(y_train[1,:], y_test[1,:], full_traj2, prediction_new)

function loss_plot(losses)
    p = plot(losses, label = "Loss", title = "Loss of ANODE-MS Model", xlabel = "Iterations", ylabel = "Loss")
    display(p)
    #savefig("Results/F1/Loss of ANODE-MS II Model on F1 data.png")
end

loss_plot(losses)

function average_loss_plot(df, title)
    row_means = mean.(eachrow(df))
    row_stds = std.(eachrow(df))
    p = Plots.plot(1:nrow(df), row_means, ribbon=row_stds, label="Average Loss with Std Dev", title=title, xlabel="Iteration", ylabel="Average Loss", legend=:bottomright, yscale=:log10)
    display(p)
    #Plots.savefig(p, title * ".png")
end

average_loss_plot(losses_df, "Average Loss Evolution of ANODE-MS II Model")