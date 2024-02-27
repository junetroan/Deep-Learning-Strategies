#Test with F1 telemetry data

using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
using CSV, Tables, DataFrames
using DataInterpolations
using OrdinaryDiffEq
using OptimizationPolyalgorithms
using DiffEqFlux
using Plots
using Statistics
using StatsBase
gr()

data_path = "test-f1.csv"
data = CSV.read(data_path, DataFrame, header = true)

#Train/test Splits
split_ration = 0.4
train = data[1:Int(round(split_ration*size(data, 1))), :]
test = data[Int(round(split_ration*size(data, 1))):end, :]

# Data Cleaning and Normalization
t = 0.0:1.0:581
speed = data[:,1]

train_data = convert(Vector{Float32}, train[:,1])
test_data = convert(Vector{Float32}, test[:,1])

transformer = fit(ZScoreTransform, train_data)
X_train = StatsBase.transform(transformer, train_data)
X_test = StatsBase.transform(transformer, test_data)
t_test = convert(Vector{Float32}, collect(Int(round(split_ration*size(data, 1))):size(data, 1)))
t_train = convert(Vector{Float32}, collect(1:Int(round(split_ration*size(data, 1)))))
tspan = (t_train[1], t_train[end])
tsteps = range(tspan[1], tspan[2], length = length(X_train))

iters = 1
state = 2
group_size = 5
continuity_term = 100.0f0

u0 = [X_train[1], 0]

fulltraj_losses = Float32[]

@time begin
    for i in 1:iters

        println("Simulation $i")

        # Random numbers
        rng_1 = StableRNG(i)
        rng_2 = StableRNG(i + 1)

        # Define the Neural Network
        nn = Lux.Chain(Lux.Dense(state, 30, tanh), Lux.Dense(30, state))
        p_init, st = Lux.setup(rng_1, nn)

        params = ComponentVector(ANN = p_init, u0_states = 0.5*randn(rng_2, Float32, state - 1))

        neuralode = NeuralODE(nn, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps,
        abstol = 1f-6, reltol = 1f-6, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))

        function rhs(du, u, p, t)
            du[1:end] = nn(u, p.ANN, st)[1]
        end

        prob_node = ODEProblem(rhs, [u0[1]; params.u0_states], tspan, params, saveat = tsteps)

        function loss_function(data, pred)
            return mean(abs2, data[1, :] - pred[1, :])
        end

        function continuity_loss_function(u_end, u_0)
            return mean(abs2, u_end - u_0)
        end

        function loss_multiple_shooting(p)
            new_prob = remake(prob_node, u0 = [u0[1]; p.u0_states], p = p)
            return multiple_shoot(p, X_train, tsteps, new_prob, loss_function, continuity_loss_function, AutoTsit5(Rosenbrock23(autodiff = false)),
                                group_size; continuity_term, abstol = 1f-6, reltol = 1f-6, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
        end

        function predict_final(p)
            prob_new = remake(prob_node, u0 = [u0[1]; p.u0_states], p = p)
            return Array(solve(prob_new, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, abstol = 1f-6, reltol = 1f-6))[1, :]
        end

        function final_loss(θ)
            X̂ = predict_final(θ)
            prediction_error = mean(abs2, X_train[1, :] .- X̂)
            prediction_error
        end


        function predict_single_shooting(p)
            prob_new = remake(prob_node, u0 = [u0[1]; p.u0_states], p = p)
            return Array(solve(prob_new, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, abstol = 1f-6, reltol = 1f-6))[1, :]
        end

        losses = Float32[]

        callback = function (p, l, preds; doplot = false)
            push!(losses, final_loss(p))
            if length(losses) % 50 == 0
                println("Current loss after $(length(losses)) iterations: $(losses[end])")
            end
            return false
        end

        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shooting(x), adtype)
        optprob = Optimization.OptimizationProblem(optf, params)
        @time res_ms = Optimization.solve(optprob, ADAM(), maxiters = 5000; callback = callback)

        losses_df = DataFrame(loss = losses)
        CSV.write("sim-FS-NODE-MS/Loss Data/Losses $i.csv", losses_df, writeheader = false)

        # Evaluate Single Shooting
        function loss_single_shooting(p)
            pred = predict_single_shooting(p)
            l = loss_function(Xₙ, pred)
            return l, pred
        end

        full_traj = predict_final(res_ms.u)
        full_traj_loss = final_loss(res_ms.u)
        push!(fulltraj_losses, full_traj_loss)

        function plot_results(tp, real, pred)
            plot(tp, pred, label = "Training Prediction", title="Iteration $i of Randomised NODE-MS Model", xlabel="Time", ylabel="Population")
            scatter!(tp, real[1,:], label = "Training Data")
            plot!(legend=:topright)
            savefig("sim-FS-NODE-MS/Plots/Simulation $i.png")
        end

        plot_results(t_train, X_train, full_traj)

        if i==iters
            println("Simulation finished")
            break 
        end

    end
end


