using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
using CSV, Tables, DataFrames
gr()

function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

rng = StableRNG(1111)

# Experimental parameters
state = 4 # number of state variables in the neural network
tspan = (0.0f0, 10.0f0)
# Producing data
u0 = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)
X = Array(solution)
t = solution.t
x̄ = mean(X, dims = 2)
noise_magnitude = 10f-3
noise_magnitude2 = 62.3f-2
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))



# Simple NN to predict lotka-volterra dynamics
rng1 = StableRNG(1003)
U = Lux.Chain(Lux.Dense(state, 30, tanh),Lux.Dense(30, state))
p, st = Lux.setup(rng1, U)

# Define the hybrid model
function ude_dynamics!(du, u, p, t)
    û = U(u, p.vector_field_model, st)[1] # Network prediction
    du[1:end] = û[1:end]
end

# Construct ODE Problem
augmented_u0 = vcat(u0[1], zeros(Float32, state - 1))
params = ComponentVector{Float32}(vector_field_model = p)
prob_nn = ODEProblem(ude_dynamics!, augmented_u0, tspan, params, saveat = 0.25f0)


sol = solve(prob_nn, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)

plot(sol)



function predictor(du,u,p,t)
    K = p 
    θ = u[1]
    yt = y(t)
    e = yt - θ
    du[1] = dθ + K*e
end


function zeroh(x, xp, yp)
    # Inner function to handle single value interpolation
    function interpolate(x0)
        if x0 <= xp[1]
            return yp[1]
        elseif x0 >= xp[end]
            return yp[end]
        end
        k = 1
        while x0 > xp[k]
            k += 1
        end
        return yp[k-1]
    end
    
    # Handle different types of input x
    if isa(x, Number)
        return interpolate(x)
    elseif isa(x, AbstractArray)
        return map(interpolate, x)
    else
        error("x must be a Number or an AbstractArray")
    end
end


f(x) = sin(x)
xp = collect(range(0, stop=10, length=20))
yp = f.(xp)
x = collect(range(0, stop=12, length=1000))

y = zeroh(x,xp,yp)

plot(x, f.(x), label="f(x)")
scatter!(xp, yp, label="data points")
plot!((x), (y), label="zeroth order hold")

full_t = collect(0:0.25:40)

yp2 = solution(t)[1,:]
y1 = zeroh(full_t, t, yp2)


plot(solution[1,:],label="f(x)")
scatter!(yp2, label="data points")
plot!(y1[1:41], label="zeroth order hold")

