# sim-HL-ANODE-MS

using CSV, DataFrames, Plots, Statistics, StatsBase
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
gr()
#plotly()

# Collecting Data
data_path = "Multiple Shooting (MS)/ANODE-MS/Data/lynx_hare_data.csv"
data = CSV.read(data_path, DataFrame)

#Train/test Splits
split_ration = 0.25
train = data[1:Int(round(split_ration*size(data, 1))), :]
test = data[Int(round(split_ration*size(data, 1))):end, :]

# Data Cleaning and Normalization
hare_data = data[:, 2]
lynx_data = data[:, 3]

transformer = fit(ZScoreTransform, hare_data)
X_train = StatsBase.transform(transformer, train[:, 2])
X_test = StatsBase.transform(transformer, test[:, 2])

t = collect(1:size(data, 1))
t_train = collect(1:Int(round(split_ration*size(data, 1))))
t_test = collect(Int(round(split_ration*size(data, 1))):size(data, 1))
tspan = (minimum(t_train), maximum(t_train))

# Define the experimental parameter
groupsize = 5
predsize = 5
state = 2


fulltraj_losses = Float64[]

# NUMBER OF ITERATIONS OF THE SIMULATION
iters = 2

@time begin
    for i in 1:iters
        
        println("Simulation $i")
        
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
        augmented_u0 = vcat(X_train[1], randn(rng3, Float63, state - 1))
        params = ComponentVector{Float63}(vector_field_model = p, initial_condition_model = p0)
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
            Array(solve(shooting_problem, verbose = false,  AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-6, reltol = 1e-6, 
            p=θ, saveat = t_train, trajectories = length(u0_vec), sensealg = sensealg))
        end

        function loss(θ)
            X̂ = predict(θ)
            continuity = mean(abs2, X̂[:, end, 1:end - 1] - X̂[:, 1, 2:end])
            prediction_error = mean(abs2, targets .- X̂[1,:,:])
            prediction_error + continuity*10
        end

        function predict_final(θ)
            predicted_u0_nn = U0_nn(nn_predictors[:,1], θ.initial_condition_model, st0)[1]
            u0_all = vcat(u0_vec[1], predicted_u0_nn)
            prob_nn_updated = remake(prob_nn, p=θ, u0=u0_all) # no longer updates u0 nn
            X̂ = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff=true)), abstol = 1e-6, reltol = 1e-6, 
            saveat = t_train, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
            X̂
        end

        function final_loss(θ)
            X̂ = predict_final(θ)
            prediction_error = mean(abs2, X_train .- X̂[1, :])
            prediction_error
        end

        losses = Float63[]

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


        
        full_traj = predict_final(res_ms.u)
        full_traj_loss = final_loss(res_ms.u)
        push!(fulltraj_losses, full_traj_loss)

        optf_final = Optimization.OptimizationFunction((x,p) -> final_loss(x), adtype)
        optprob_final = Optimization.OptimizationProblem(optf_final, res_ms.u)
        @time res_final = Optimization.solve(optprob_final, BFGS(initial_stepnorm = 0.01), callback=callback, maxiters = 1000, allow_f_increases = true)

        full_traj2 = predict_final(res_final.u)

        losses_df = DataFrame(loss = losses)
        CSV.write("Results/HL/Loss Data/Losses $i.csv", losses_df, writeheader = false)

        function plot_results(tp,tr, real, pred)
            plot(tp, pred[1,:], label = "Training Prediction", title="Trained ANODE-MS II Model predicting Hare data", xlabel = "Time", ylabel = "Population")
            plot!(tp, real, label = "Training Data")
            plot!(legend=:topright)
            savefig("Results/HL/Plots/Training ANODE-MS II Model on Hare data.png")
        end

        plot_results(t_train, t, X_train, full_traj2)


        if i==iters
            println("Simulation finished")
            break 
        end

    end
end

#=

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
augmented_u0 = vcat(X_train[1], randn(rng3, state - 1))
params = ComponentVector{Float64}(vector_field_model = p, initial_condition_model = p0)
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
    Array(solve(shooting_problem, verbose = false,  AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-6, reltol = 1e-6, 
    p=θ, saveat = t_train, trajectories = length(u0_vec), sensealg = sensealg))
end

function loss(θ)
    X̂ = predict(θ)
    continuity = mean(abs2, X̂[:, end, 1:end - 1] - X̂[:, 1, 2:end])
    prediction_error = mean(abs2, targets .- X̂[1,:,:])
    prediction_error + continuity*10
end

function predict_final(θ)
    predicted_u0_nn = U0_nn(nn_predictors[:,1], θ.initial_condition_model, st0)[1]
    u0_all = vcat(u0_vec[1], predicted_u0_nn)
    prob_nn_updated = remake(prob_nn, p=θ, u0=u0_all) # no longer updates u0 nn
    X̂ = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff=true)), abstol = 1e-6, reltol = 1e-6, 
    saveat = t_train, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
    X̂
end

function final_loss(θ)
    X̂ = predict_final(θ)
    prediction_error = mean(abs2, X_train .- X̂[1, :])
    prediction_error
end

losses = Float64[]

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
CSV.write("sim-HL-ANODE-MS/Loss Data/Losses $i.csv", losses_df, writeheader = false)
        
full_traj = predict_final(res_ms.u)
full_traj_loss = final_loss(res_ms.u)
push!(fulltraj_losses, full_traj_loss)


optf_final = Optimization.OptimizationFunction((x,p) -> final_loss(x), adtype)
optprob_final = Optimization.OptimizationProblem(optf_final, res_ms.u)
@time res_final = Optimization.solve(optprob_final, BFGS(initial_stepnorm = 0.01), callback=callback, maxiters = 1000, allow_f_increases = true)

full_traj2 = predict_final(res_final.u)

function plot_results(tp,tr, real, pred)
    plot(tp, pred[1,:], label = "Training Prediction", title="Trained ANODE-MS II Model predicting Hare data", xlabel = "Time", ylabel = "Population")
    plot!(tp, real, label = "Training Data")
    plot!(legend=:topright)
    savefig("Results/HL/Training ANODE-MS II Model on Hare data.png")
end

plot_results(t_train, t, X_train, full_traj2)

#############################################################################################################################################################################################
# Testing 
X_train
X_test
all_data = vcat(X_train[1:22], X_test)
X_test 
t_test
tspan_test = (t_test[1], t_test[end])
tsteps_test = range(tspan_test[1], tspan_test[2], length = length(X_test))

predicted_u0_nn = U0_nn(nn_predictors[:, 1], res_final.u.initial_condition_model, st0)[1]
u0_all = vcat(u0_vec[1], predicted_u0_nn)
prob_nn_updated = remake(prob_nn, p = res_final.u, u0 = u0_all, tspan = tspan_test)
prediction_new = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff = true)),  abstol = 1e-6, reltol = 1e-6,
saveat = 1.0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))

t1 = t_train |> collect
t3 = t_test |> collect
t = t |> collect

test_loss = X_test - prediction_new[1,:]
total_test_loss = sum(abs, test_loss)

gr()
function plot_results(real_train, real_test, pred, pred_new)
    plot(t1, pred[1,:], label = "Training Prediction", title="Training and Test Predictions of ANODE-MS II Model", xlabel = "Time", ylabel = "Population")
    plot!(t3, pred_new[1,:], label = "Test Prediction")
    scatter!(t1, real_train, label = "Training Data")
    scatter!(t[23:end], real_test, label = "Test Data")
    vline!([t3[1]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("Results/HL/Training and testing of ANODE-MS II Model on Hare and Lynx data.png")
end

plot_results(X_train, X_test, full_traj2, prediction_new)

plot(full_traj[1,:])
plot!(prediction_new[1,:])

=#