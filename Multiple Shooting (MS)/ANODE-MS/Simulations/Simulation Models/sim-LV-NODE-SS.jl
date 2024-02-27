# sim-LV-NODE-SS
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

function trueODEfunc(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

group_size = 5
iters = 2
state = 2

rng = StableRNG(1111)
datasize = 41
tsteps = 0.25f0
tspan = (0.0f0, 10.0f0) # Original (0.0f0, 10.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)
noise_magnitude = 10f-3
continuity_term = 10.0f0
u0 = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob_trueode = ODEProblem(trueODEfunc, u0, tspan, p_)
solution = solve(prob_trueode, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = tsteps)

X = Array(solution)
t = solution.t
x̄ = mean(X, dims = 2)
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))

fulltraj_losses = Float32[]

@time begin
    for i in 1:iters
        
        println("Simulation $i")

        # Random numbers
        rng_i = StableRNG(i)

        # Define the Neural Network
        nn = Lux.Chain(Lux.Dense(state, 30, tanh), Lux.Dense(30, state))
        p_init, st = Lux.setup(rng_i, nn)

        neuralode = NeuralODE(nn, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
        prob_node = ODEProblem((u,p,t) -> nn(u,p,st)[1], [u0[1]; zeros(state -1)], tspan, ComponentArray(p_init))

        function loss_function(data, pred)
            return mean(abs2, data - pred)
        end

        function loss_multiple_shooting(p)
            return multiple_shoot(p, Xₙ, tsteps, prob_node, loss_function, AutoTsit5(Rosenbrock23(autodiff=false)),
                                group_size; continuity_term)
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
        optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shooting(x), adtype)
        optprob = Optimization.OptimizationProblem(optf, ComponentArray(p_init))
        @time res_ms = Optimization.solve(optprob, ADAM(), maxiters=5000; callback = callback)

        losses_df = DataFrame(loss = losses)
        CSV.write("Simulations/Results/sim-LV-NODE-SS/Loss Data/Losses $i.csv", losses_df, writeheader = false)

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
            savefig("Simulations/Results/sim-LV-NODE-SS/Plots/Simulation $i.png")
        end

        plot_results(Xₙ, full_traj)

        if i==iters
            println("Simulation finished")
            break 
        end

    end
end

# Time
# Totat time = 596.560500 seconds 