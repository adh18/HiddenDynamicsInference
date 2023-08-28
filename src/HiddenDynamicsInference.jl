module HiddenDynamicsInference
# Required packages 
# Differential equations
using OrdinaryDiffEq
export ODEProblem, solve, Tsit5
# Polynomials 
import DynamicPolynomials, StaticPolynomials
import StaticPolynomials.PolynomialSystem
# Base libraries
using LinearAlgebra, Random, Statistics
# Optimization
using Flux, Optim, ProgressMeter

# Optimization
export PolynomialHiddenDynamicsInferenceProblem, powgenerate, optimize, randinit
include("polynomials.jl") 
include("optimize.jl")

using JuMP, GLPK
using Distances, Peaks, StatsBase, Combinatorics, KernelDensity, Clustering, LsqFit

# Analysis
export find_limit_cycle, dataloss
include("dynamics_analysis.jl")

export std_scale_coefficients!, center_coefficients!, hclust
include("polynomial_analysis.jl")

export calculate_KDEthresh, compute_model_distances, hierarchical_clustering_range, coeff_var_ranking, kemeny_young, k_to_h, h_to_k
include("model_analysis.jl")

# Simple function for printing rankings
export pow2strs
function tup2str(tup, fldstr)
    @assert length(tup) == length(fldstr)
    if all(tup .== 0)
            return "1"
    end
    str = ""
    for i = 1:length(tup)
        if tup[i] == 0
        elseif tup[i] == 1
            str *= fldstr[i]
        elseif tup[i] == 2
            str *= fldstr[i] * "²"
        elseif tup[i] == 3
            str *= fldstr[i] * "³"
        elseif tup[i] == 4
            str *= fldstr[i] * "⁴"
        elseif tup[i] == 5
            str *= fldstr[i] * "⁵"
        elseif tup[i] == 6
            str *= fldstr[i] * "⁶"
        elseif tup[i] == 7
            str *= fldstr[i] * "⁷"
        elseif tup[i] == 8
            str *= fldstr[i] * "⁸"
        elseif tup[i] == 9
            str *= fldstr[i] * "⁹"
        else
            @warn "Power $(tup[i]) not implemented yet"
            str *= fldstr[i] * "^($(tup[i]))"
        end
    end
    return str
end

function pow2strs(pows, fldstr)
    strs = Vector{String}()
    for i = 1:length(pows)
        push!(strs, ("Eq.$i: " .* tup2str.(pows[i], (fldstr, )))...)
    end
    return strs
end
end #module