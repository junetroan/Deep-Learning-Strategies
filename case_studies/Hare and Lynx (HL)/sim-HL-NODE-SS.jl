# NOT finished
# Unsure how to change to be SS. sim-HL-NODE-MS.jl is adapted and works. 


# sim-HL-NODE-SS.jl
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Distributions
using ComponentArrays, Lux, OptimizationPolyalgorithms, Plots
using DiffEqFlux
using DiffEqFlux: group_ranges
using StableRNGs
using Random
using CSV, Tables, DataFrames
using Statistics
using StatsBase

gr()

# Collecting Data
data_path = "Multiple Shooting (MS)/ANODE-MS/Data/lynx_hare_data.csv"
data = CSV.read(data_path, DataFrame)

#Train/test Splits
split_ration = 0.75
train = data[1:Int(round(split_ration*size(data, 1))), :]
test = data[Int(round(split_ration*size(data, 1))):end, :]

# Data Cleaning and Normalization
hare_data = data[:, 2]
lynx_data = data[:, 3]

transformer = fit(ZScoreTransform, hare_data)
X_train = Float32.(StatsBase.transform(transformer, train[:, 2]))
X_test = Float32.(StatsBase.transform(transformer, test[:, 2]))
X_new = reshape(X_train, 1, :)
unknown_X = zeros(Float32, 1, 68)
X = vcat(X_new, unknown_X)

t = Float32.(collect(1:size(data, 1)))
t_train = Float32.(collect(1:Int(round(split_ration*size(data, 1)))))
t_test = Float32.(collect(Int(round(split_ration*size(data, 1))):size(data, 1)))

# Define the experimental parameter
rng = StableRNG(1111)
group_size = 5
state = 2
tspan = (Float32(minimum(t_train)), Float32(maximum(t_train)))
datasize = length(X_train)
tsteps = range(tspan[1], tspan[2], length = datasize)
continuity_term = 10.0f0
u0 = Float32[X_train[1], 0.0f0]

fulltraj_losses = Float32[]


iters = 2

@time begin
    for i in 1:iters
        
        println("Simulation $i")

        # Random numbers
        rng_i = StableRNG(i)

        # Define the Neural Network
        nn = Lux.Chain(Lux.Dense(state, 30, tanh), Lux.Dense(30, state-1))
        p_init, st = Lux.setup(rng_i, nn)

        neuralode = NeuralODE(nn, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
        prob_node = ODEProblem((u,p,t) -> nn(u,p,st)[1], [u0[1]; zeros(state -1)], tspan, ComponentArray(p_init))

        function loss_function(data, pred)
            return mean(abs2, data - pred)
        end

        function continuity_loss_function(u_end, u_0)
            return mean(abs2, u_end - u_0)
        end

        function loss_multiple_shooting(p)
            return multiple_shoot(p, X, tsteps, prob_node, loss_function, continuity_loss_function,  AutoTsit5(Rosenbrock23(autodiff=false)), #ERROR: DimensionMismatch: dimensions must match: a has dims (Base.OneTo(1),), b has dims (Base.OneTo(2),), mismatch at 1
                                group_size; continuity_term, abstol = 1e-8, reltol = 1e-8)
        end

        losses = Float32[]
        callback = function (p, l, preds; doplot = false)
            push!(losses, l)
            if length(losses) % 50 == 0
                println("Current loss after $(length(losses)) iterations: $(losses[end])")
            end
            return false
        end

        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((p,_) -> loss_multiple_shooting(p), adtype)
        optprob = Optimization.OptimizationProblem(optf, ComponentArray(p_init))
        #ptprob = Optimization.OptimizationProblem(optf,p_init)
        @time res_ms = Optimization.solve(optprob, ADAM(), maxiters=5000; callback = callback)

        losses_df = DataFrame(loss = losses)
        #CSV.write("Simulations/Results/sim-HL-NODE-SS/Loss Data/Losses $i.csv", losses_df, writeheader = false)

        function predict_final(θ)
            return Array(neuralode([u0[1]; zeros(state -1)], θ, st)[1])
        end

        function final_loss(θ)
            X̂ = predict_final(θ)
            prediction_error = mean(abs2, Xₙ[1,:] .- X̂[1, :])
            prediction_error
        end

        full_traj = predict_final(res_ms.u)
        full_traj_loss = final_loss(res_ms.u)
        push!(fulltraj_losses, full_traj_loss)

        function plot_results(real, pred)
            plot(t, pred[1,:], label = "Training Prediction", title="Iteration $i of Randomised NODE-SS Model", xlabel="Time", ylabel="Population")
            scatter!(t, real[1,:], label = "Training Data")
            plot!(legend=:topright)
            #savefig("Simulations/Results/sim-HL-NODE-SS/Plots/Simulation $i.png")
        end

        plot_results(X, full_traj)

        if i==iters
            println("Simulation finished")
            break 
        end

    end
end