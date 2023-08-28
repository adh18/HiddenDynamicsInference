using Random, Statistics, LinearAlgebra
using CairoMakie
include("HiddenDynamicsInference.jl")
using .HiddenDynamicsInference
# Simulate test data
truep = vcat([0.5, 1.0, -1.0, 0.0, 0.0, 0.0, -1/3, 0.0, 0.0, 0.0], [0.7, 1.0, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] / 12.5 )
tspan = (0.0, 250.0)
function fhn_odefun!(du, u, p, t)
    du[1] = 0.5 + u[1] - u[2] - u[1]^3/3
    du[2] = (0.7 + u[1] - 0.8 * u[2])/12.5
end
prob = HiddenDynamicsInference.ODEProblem(fhn_odefun!, [1.0, 1.0], tspan)
truedata = HiddenDynamicsInference.solve(prob, HiddenDynamicsInference.Tsit5(), saveat = 0.5, abstol = 1e-14, reltol = 1e-14)
truep = [truedata[:, 141]; normalize(truep[1:10]); normalize(truep[11:20]); norm(truep[1:10]); norm(truep[11:20])]

t = truedata.t; t .-= minimum(t)
truedata = truedata[1:2, :]
window = 141:375

# add noise to the data
rng = MersenneTwister(1111)
noise_level = 0
data = truedata + noise_level / 100 * std(truedata, dims = 2) .* randn(rng, size(truedata))
println("Data loss with true FHN: $(sum(abs2, data - truedata) / length(truedata))")

## Set up the HDIproblem
HDIprob = HiddenDynamicsInference.PolynomialHiddenDynamicsInferenceProblem(powgenerate(2, 3), t[window] .- t[window[1]], data[1:1, window])
optps = Vector{Float64}[]
p0s = Vector{Float64}[]
optvals = Float64[]
for i = 1:25
    println("Run $i beginning")
    p0 = HiddenDynamicsInference.randinit(HDIprob)
    optp, optval = HiddenDynamicsInference.optimize(HDIprob, p0, lambda = 1e-6, eta = 1e-2)
    push!(optps, optp)
    push!(p0s, p0)
    push!(optvals, optval) 
    println("Run $i complete")
end

##
using DelimitedFiles
sweepcsv, sweepheader = readdlm("/Users/alasdairhastewell/Dropbox (MIT)/Research/NonlinearOscillators/HiddenDynamicsInference/examples/fhn/fhn_2D_noise10_sweep_results.csv", '\t', header = true)


sweepcsv, sweepheader = readdlm("/Users/alasdairhastewell/Dropbox (MIT)/Research/NonlinearOscillators/HiddenDynamicsInference/examples/fhn/fhn_2D_fit_sweep_results_noise10.csv", ',', header = true)
params = map(str -> parse.(Float64, split(replace(str, "[" => " ", "]" => " "), ',')), sweepcsv[:, 8])
newparams = Vector{Float64}[]
for i = axes(sweepcsv, 1)
    d1 = sweepcsv[i, 1]
    d2 = sweepcsv[i, 2]
    param = params[i]
    newparam = NaN*ones(24)
    newparam[1:2] = param[1:2]
    newparam[end - 1:end] = param[end - 1:end]
    d1offset = div((d1 + 1) * (d1 + 2), 2)
    d2offset = div((d2 + 1) * (d2 + 2), 2)
    @assert 2 + d1offset + d2offset == length(param) - 2
    newparam[3:2 + d1offset] = param[3:2 + d1offset]
    newparam[13:12 + d2offset] = param[3 + d1offset:2 + d1offset + d2offset]
    push!(newparams, newparam)
end
writedlm("fhn_2D_noise10_sweep_results.csv", [["deg1" "deg2" "lambda" "train_loss" "train_delay_loss" "test_delay_loss" "p".*string.((1:24)')]; [sweepcsv[:, 1] sweepcsv[:, 2] sweepcsv[:, 3] sweepcsv[:, 5]  sweepcsv[:, 6]  sweepcsv[:, 7] vcat(newparams'...)]])

d = 3
div((d + 1) * (d + 2), 2)