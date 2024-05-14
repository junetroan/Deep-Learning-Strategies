#ANODE-MS F1 Predictor
using CSV, DataFrames, Plots, Statistics, StatsBase
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
using DataInterpolations
gr()

data_path = "Multiple Shooting (MS)/ANODE-MS/Data/Telemetry Data - VER Spain 2023 Qualifying.csv"
data = CSV.read(data_path, DataFrame, header = true)

#Train/test Splits
distance = convert(Vector{Float32}, data[:,1])
speed = convert(Vector{Float32}, data[:,2])
throttle = convert(Vector{Float32}, data[:,3])
brake = convert(Vector{Float32}, data[:,4])

split_ration = 0.25

distance_train = distance[1:Int(round(split_ration*size(data, 1)))]
distance_test = distance[Int(round(split_ration*size(data, 1))):end]
speed_train = speed[1:Int(round(split_ration*size(data, 1)))]
speed_test = speed[Int(round(split_ration*size(data, 1))):end]
throttle_train = throttle[1:Int(round(split_ration*size(data, 1)))]
throttle_test = throttle[Int(round(split_ration*size(data, 1))):end]
brake_train = brake[1:Int(round(split_ration*size(data, 1)))]
brake_test = brake[Int(round(split_ration*size(data, 1))):end]

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
U0_nn = Lux.Chain(Lux.Dense(groupsize, 30, tanh), Lux.Dense(30,  2))
p0, st0 = Lux.setup(rng2, U0_nn)

throttle = LinearInterpolation(th_train, t_train)
brake = ConstantInterpolation(b_train, t_train)

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
    h = u[4]

    th = throttle(t)
    br = brake(t)

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
rands = randn(rng3, Float32, length(s_train), unknown_states)'
augmented_u0 = vcat(rands, y_train)
params = ComponentVector{Float32}(vector_field_model = p, initial_condition_model = p0)
prob_nn = ODEProblem(nn_dynamics!, augmented_u0, tspan_train, params, saveat = t_train)

#=
for i in 1:(groupsize-1):length(t_train) - max(groupsize, predsize) + 1
    println(i)
    println(i + max(groupsize, predsize) - 1)
end

parent = [y_train[:,i: i + max(groupsize, predsize) - 1] for i in 1:(groupsize-1):length(t_train) - max(groupsize, predsize) + 1]
parent[1]
parent[2]
parent[10]
parent[101]

parent[1][1,:]
parent[101][1,:]
parent[101]

parent[1:groupsize,:]

parent[1][1,:]

parent[:][1,:]
test = parent[:][1,:]

u0 = parent[:][1,1]

parent[1][1,1]
parent[2][1,1]
parent[3][1,1]
parent[4][1,1]
parent[101][1,1]
u0_nn11 = U0_nn(first[:,1], params.initial_condition_model, st0)[1]
u0_nn12 = U0_nn(first[:,2], params.initial_condition_model, st0)[1]
u0_nn1101 = U0_nn(first[:,101], params.initial_condition_model, st0)[1]
u0_nn21 = U0_nn(second[:,1], params.initial_condition_model, st0)[1]
u0_nn2101 = U0_nn(second[:,101], params.initial_condition_model, st0)[1]

u0vec1 = u0_vec[1][1,1]
u0vec2 = u0_vec[1][2,1]
u0vec3 = u0_vec[1][3,1]

u0all = vcat(u0vec1, u0_nn11)

testerrr = remake(prob_nn, u0 = u0all, tspan = (t_train[1], t_train[groupsize]))

u0_vec = [x[row, 1] for x in parent for row in 1:3]

first_elements = [x[1, 1] for x in parent]

[x[:,1] for x in parent]


targets = vcat([parent[i][j, :] for i in 1:101 for j in 1:3]...)


y_train
pt = [y_train[:,i: i + max(groupsize, predsize) - 1] for i in 1:(groupsize-1):length(t_train) - max(groupsize, predsize) + 1]
ps = cat(pt..., dims=3)

ft =  hcat([pt[i][1, :] for i in 1:101]...)
u0 = ft[1,:]
targs = 
# current targets: 5×303 Matrix{Float32}
# current parents: 101-element Vector{Matrix{Float32}}:
# pred: 4×5×101 Array{Float32, 3}:
=#

function group_x(xdim, y , groupsize, predictsize)
    parent = [y[:,i: i + max(groupsize, predictsize) - 1] for i in 1:(groupsize-1):length(xdim) - max(groupsize, predictsize) + 1]
    firsts =  hcat([parent[i][1, :] for i in 1:36]...) # Throttle
    seconds = hcat([parent[i][2, :] for i in 1:36]...) # Brake
    thirds = hcat([parent[i][3, :] for i in 1:36]...) # Speed
    targets = reshape(vcat([parent[i][j, :] for i in 1:36 for j in 1:3]...), groupsize, 108)
    parent = cat(parent..., dims=3)
    u0 = hcat(firsts[1,:], thirds[1,:])
    return parent, targets, firsts, seconds, thirds, u0
end

pas, targets, first_series, second_series, third_series, u0_vec = group_x(t_train, y_train, groupsize, predsize)
#=
function tester()
    u0_nn_first = []
    u0_nn_second = []
    u0_nn_third = []
    for j in 1:size(first_series, 2)
        u0_nn = U0_nn(first_series[:, j], params.initial_condition_model, st0)[1]
        push!(u0_nn_first, u0_nn)
                
        u0_nn = U0_nn(second_series[:, j], params.initial_condition_model, st0)[1]
        push!(u0_nn_second, u0_nn)
                
        u0_nn = U0_nn(third_series[:, j], params.initial_condition_model, st0)[1]
        push!(u0_nn_third, u0_nn)
    end
    return u0_nn_first, u0_nn_second, u0_nn_third
end


first, second, third = tester()
u0_all = vcat(u0_vec[1], first[1], second[1], third[1])
=#
plotly()
plot(s_train, label = "Speed")
plot(th_train, label = "Throttle")
plot(b_train, label = "Brake")

discont = [66, 68, 69,85]

function tpredictor(θ)
    function prob_func(prob, i, repeat)
        u0_nn_third = [U0_nn(first_series[:, j], θ.initial_condition_model, st0)[1] for j in 1:size(first_series, 2)]
        u0_all = vcat(u0_vec[i,:], u0_nn_third[i])
        remake(prob, u0 = u0_all, tspan = (t_train[1], t_train[groupsize]))
    end
    sensealg = ReverseDiffAdjoint()
    shooting_problem = EnsembleProblem(prob = prob_nn, prob_func = prob_func) 
    Array(solve(shooting_problem, verbose = false, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1f-6, reltol = 1f-6, 
    p=θ, saveat = t_train, trajectories = 36, sensealg = sensealg, tstops = discont))
end

#=
function predictor(θ)
    function prob_func(prob, i, repeat)
        u0_nn_first = []

        for j in 1:size(first_series, 2)
            u0_nn = U0_nn(first_series[:, j], θ.initial_condition_model, st0)[1]
            push!(u0_nn_first, u0_nn)
            
        end

        u0_all = vcat(u0_vec[i], u0_nn_first[i], u0_nn_second[i], u0_nn_third[i])
        remake(prob, u0 = u0_all, tspan = (t_train[1], t_train[groupsize]))
    end
    sensealg = ReverseDiffAdjoint()
    shooting_problem = EnsembleProblem(prob = prob_nn, prob_func = prob_func) 
    Array(solve(shooting_problem, verbose = false,  AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1f-6, reltol = 1f-6, 
    p=θ, saveat = t_train, trajectories = 36, sensealg = sensealg))
end
=#
#=
pred = predictor(params)
all_preds = [pred[1,:,:] pred[2,:,:] pred[3,:,:]]'

pred[2:end,:,:] .- ps
=#

function loss(θ)
    X̂ = tpredictor(θ)
    continuity = mean(abs2, X̂[:, end, 1:end - 1] - X̂[:, 1, 2:end])
    prediction_error = mean(abs2, pas .- X̂[2:end,:,:])
    prediction_error + continuity*10f0
end

lossezzz = loss(params)
#=
first_series[:, 1]
U0_nn(first_series[:, 1], params.initial_condition_model, st0)[1]
pred_u0_nn_first = U0_nn(first_series[:, 1], params.initial_condition_model, st0)[1]
pred_u0_nn_second = U0_nn(second_series[:, 1], params.initial_condition_model, st0)[1]
pred_u0_nn_third = U0_nn(third_series[:, 1], params.initial_condition_model, st0)[1]
pred_u0_nn = vcat(pred_u0_nn_first, pred_u0_nn_second, pred_u0_nn_third)
u0_1 = vcat(u0_vec[1], pred_u0_nn)
=#

function predict_final(θ)
    u0_nn_first = U0_nn(first_series[:, 1], θ.initial_condition_model, st0)[1]
    u0_all = vcat(u0_vec[1,:], u0_nn_first)
    prob_nn_updated = remake(prob_nn, p = θ, u0 = u0_all)

   # no longer updates u0 nn
    X̂ = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff=true)), abstol = 1f-6, reltol = 1f-6, 
    saveat = t_train, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),tstops = discont))
    X̂
end

final_preds = predict_final(params) #INFINITY?????

function final_loss(θ)
    X̂ = predict_final(θ)
    final_pred = X̂[[1, 3, 4], :]
    prediction_error = mean(abs2, y_train .- final_pred)
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
CSV.write("Results/F1/Loss Data/ANODE-MS F1 Single Augmentation Loss - 07.05.24.csv", losses_df, writeheader = false)

full_traj = predict_final(res_ms.u)
full_traj_loss = final_loss(res_ms.u)

optf_final = Optimization.OptimizationFunction((x,p) -> final_loss(x), adtype)
optprob_final = Optimization.OptimizationProblem(optf_final, res_ms.u)
@time res_final = Optimization.solve(optprob_final, BFGS(initial_stepnorm = 0.01), callback=callback, maxiters = 1000, allow_f_increases = true)

full_traj2 = predict_final(res_final.u)
actual_loss = Xₙ[1,:] - full_traj2[1,:]
total_loss = abs(sum(actual_loss))


function plot_training(tp, real, pred)
    plot(tp, pred, label = "Training Prediction", title="Trained ANODE-MS Model predicting F1 data", xlabel = "Time", ylabel = "Speed")
    plot!(tp, real, label = "Training Data")
    plot!(legend=:topright)
    savefig("Results/F1/Training ANODE-MS II Model on F1 data.png")
end

plot_training(t_train, y_train[1,:], full_traj[2,:])

function predict_test(θ)
    pred_u0_nn_first = U0_nn(first_series[:, 1], θ.initial_condition_model, st0)[1]
    pred_u0_nn_second = U0_nn(second_series[:, 1], θ.initial_condition_model, st0)[1]
    pred_u0_nn_third = U0_nn(third_series[:, 1], θ.initial_condition_model, st0)[1]
    pred_u0_nn = vcat(pred_u0_nn_first, pred_u0_nn_second, pred_u0_nn_third)
    u0_all = vcat(u0_vec[1], pred_u0_nn)
    prob_nn_updated = remake(prob_nn, p=θ, u0=u0_all)

    X̂ = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff=true)), abstol = 1f-6, reltol = 1f-6, 
    saveat = t_test, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
    X̂
end

##### Based on plot no need in testing, since training was unsuccessful.... 
test_pred = predict_test(res_ms.u)
test = test_pred[2,:]
test_loss = y_test[1,:] - test
test_total_loss = abs(sum(test_loss))

t1 = t_train
t3 = t_test # Check whether this starts at the end of the t_train

function plot_results(t1, t3, pred, pred_new, real, sol_new)
    plot(t1, pred[1,:], label = "Training Prediction", title="Training and Test Prediction sof ANODE-MS II Model", xlabel = "Time", ylabel = "Speed")
    plot!(t3, pred_new[1,41:123], label = "Test Prediction")
    scatter!(t1, real[1,:], label = "Training Data")
    scatter!(t3, sol_new[1,41:123], label = "Test Data")
    vline!([t1[41]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("Results/F1/Training and testing of ANODE-MS II Model on F1 data.png")
end


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