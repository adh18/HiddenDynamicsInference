# Helper codes for working with polynomials
"""
Ordering for polynomial terms 
"""
powsort(a::Tuple{Int, Int}, b::Tuple{Int, Int}) = sum(a) > sum(b) ? false : sum(a) < sum(b) ? true : a[1] > b[1] ? true : a[2] < b[2] ? true : false

"""
Ordering for polynomial terms 
"""
function powsort(a::Tuple{Vararg{Int}}, b::Tuple{Vararg{Int}})
    # quick return is total degree is different
    sum(a)  > sum(b) && return false
    sum(a)  < sum(b) && return true

    # Check individual powers 
    n = min(length(a), length(b))
    for i = 1:n
        a[i] > b[i] && return true
        a[i] < b[i] && return false
    end

    return false # Default the are equal 
end

"""
Calculate all power combinations upto a certain total degree
"""
function powgenerate_helper(m, N::Number)
    allinds = Tuple.(CartesianIndices(ntuple(n -> 0:N, m)))
    return sort(allinds[sum.(allinds) .<= N], lt = powsort)
end

"""
Calculate all power combinations upto certain total degrees for all equations
"""
function powgenerate(m, Ns::AbstractVector)
    length(Ns) != m && throw(DomainError(Ns, "must be an integer or vector of length $m"))
    powers = ntuple(i -> powgenerate_helper(m, Ns[i]), m)
    return powers
end
powgenerate(m, Ns::Number) = powgenerate(m, fill(Ns, m))

# Build the polynomial from a vectof power tuples
function polyeval_naive(u, p, powvec::Vector{NTuple{numvars, Int64}}) where numvars
    @assert length(p) == length(powvec)
    poly = 0
    @inbounds for i = 1:length(powvec)
        trm = p[i]
        for j = 1:numvars
            trm *= u[j] ^ powvec[i][j]
        end
        poly += trm
    end
    poly
end

"""
Build the polynomial system corresponding to the tuple of vectors of power tuples
e.g. ([(1, 0), (1, 2)], [(0, 0), (0, 1)]) => [c₁x + c₂xy², c₃ + c₄y]
"""
function buildpolysystem(powers::NTuple{numvars, S}) where {numvars, S}
    splits = [0; cumsum(length.(powers))...]
    numparams = splits[end]
    DynamicPolynomials.@polyvar x[1:numvars]
    DynamicPolynomials.@polyvar m[1:numvars]
    DynamicPolynomials.@polyvar c[1:numparams]
    polys = DynamicPolynomials.Polynomial[]
    for i = 1:numvars
        push!(polys, m[i] * polyeval_naive(x, c[splits[i] + 1:splits[i + 1]], powers[i]))
    end
    fastpolysystem = PolynomialSystem(polys, parameters = [c; m])
end

## Sensitivity functions 

# We build the forward sensitivity polynomial ourselves since we can then directly access 
# the polynomial system functions from the type
# TODO: subtype as an AbstractODEFunction ? 
# based on ODEForwardSensitivityFunction from SciMLSensitivity.jl
"""
Caches necessary matrices and polnomials for rvaluating the forward sesitivity problem
    Construction:
        fun = PolynomialForwardSensitivityFunction(polysystem::PolynomialSystem)
        fun = PolynomialForwardSensitivityFunction(powers::NTuple{N, Vector{NTuple{N, Int64}}})
    Use: 
        fun(du, u, p, t)
"""
struct PolynomialForwardSensitivityFunction{S <: PolynomialSystem, T}
    polysystem::S
    jacobian::Matrix{T}
    paramjacobian::Matrix{T}
end

# Convenience constructor from polynomial system
function PolynomialForwardSensitivityFunction(polysystem::S) where {S <: PolynomialSystem}
    numvars = StaticPolynomials.nvariables(polysystem)
    numparams = StaticPolynomials.nparameters(polysystem)
    J = Matrix{Float64}(undef, numvars, numvars)
    pJ = Matrix{Float64}(undef, numvars, numparams)
    return PolynomialForwardSensitivityFunction{S, Float64}(polysystem, J, pJ)
end

# Convenience constructor from tuple vector
PolynomialForwardSensitivityFunction(powers::NTuple{numvars, S}) where {numvars, S} = PolynomialForwardSensitivityFunction(buildpolysystem(powers))

# Evaluation of the cache in the ODE solve 
function (polyfun::PolynomialForwardSensitivityFunction)(du, u, p, t)
    # Extract the polynomial system and cached Jacobian and parameter Jacobians 
    polysystem = polyfun.polysystem
    J = polyfun.jacobian
    pJ = polyfun.paramjacobian

    # Number of variables from the system
    numvars = StaticPolynomials.nvariables(polysystem)
    numparams = StaticPolynomials.nparameters(polysystem)
    numfields = length(u)

    # Extract the base ODE variables, IC variables, sense variables
    ubase = view(u, 1:numvars)
    dubase = view(du, 1:numvars)

    uic = reshape(view(u, numvars + 1:numvars + numvars^2), numvars, numvars)
    duic = reshape(view(du, numvars + 1:numvars + numvars^2), numvars, numvars)

    usense = reshape(view(u, numvars + numvars^2 + 1:numfields), numvars, numparams)
    dusense = reshape(view(du, numvars + numvars^2 + 1:numfields), numvars, numparams)

    # Evaluate the base ODE function 
    StaticPolynomials.evaluate!(dubase, polysystem, ubase, p)

    # Evaluate the Jacobian and the parameter Jacobian
    StaticPolynomials.jacobian!(J, polysystem, ubase, p)
    StaticPolynomials.differentiate_parameters!(pJ, polysystem, ubase, p)

    # Evaluate the sensitivity ODE
    mul!(duic, J, uic)
    mul!(dusense, J, usense)
    dusense .+= pJ 

    return nothing 
end

#= Benchmarking where to put the reshape in the sensitivity function
using BenchmarkTools
M  = randn(2, 2)
U1 = zeros(2, 22)
U2 = zeros(2, 22)
v  = randn(46)
@btime mul!($U1, $M, view($v, reshape(3:46, 2, 22))) # 192.693 ns (0 allocations: 0 bytes)
@btime mul!($U2, $M, reshape(view($v, 3:46), 2, 22)) # 182.399 ns (0 allocations: 0 bytes)
=# 

"""
Contains all the information needed to solve the Hidden Dynamics Inference Problem
    Construction:
        HDIprob = PolynomialHiddenDynamicsInferenceProblem(pows::NTuple{N, Vector{NTuple{N, Int64}}}, t::Vector{T}, data::Matrix{T})
    Use: 
        optparams = optimize(HDIprob, p0; lambda = 0.0, gamma = 100_000)
"""
struct PolynomialHiddenDynamicsInferenceProblem{N, T, S, OP, SP, tV <: AbstractVector{T}, dV <: AbstractMatrix{T}}
    polysystem::S
    splits::NTuple{N, Int64}
    pows::NTuple{N, Vector{NTuple{N, Int64}}}

    # Data 
    t::tV
    data::dV

    # ODE problems
    odeprob::OP
    senprob::SP

    # Cached vectors for gradient calculations
    u0::Vector{T}
end

# Constructor
function PolynomialHiddenDynamicsInferenceProblem(pows, t, data)
    numvars = length(pows)
    splits = length.(pows)
    tspan = extrema(t)
    # Build the polynomial system
    polysystem = buildpolysystem(pows)
    # Number of parameters in the polynomial system
    numparams = StaticPolynomials.nparameters(polysystem)
    # Build the ODE problems
    u0 = zeros(numvars + numvars^2 + numparams * numvars)
    for i = 1:numvars
        u0[numvars * i + i] = 1.0
    end
    odeprob = ODEProblem((du, u, p, t) -> StaticPolynomials.evaluate!(du, polysystem, u, p), zeros(numvars), tspan)
    senfun = PolynomialForwardSensitivityFunction(polysystem)
    senprob = ODEProblem(senfun, u0, tspan)

    return PolynomialHiddenDynamicsInferenceProblem{numvars, eltype(data), typeof(polysystem), typeof(odeprob), typeof(senprob), typeof(t), typeof(data)}(polysystem, splits, pows, t, data, odeprob, senprob, u0)
end

"""
Evaluate the value and gradient of the HDIproblem for given parameters p overwriting the gradients into g
    value_gradient!(HDIprob::PolynomialHiddenDynamicsInferenceProblem, g, p, lambda = 0.0, gamma = 100_000; resetg = true, kwargs...)
    
    resetg controls if the gradients vector is reset to 0 before being accumulated
"""
function value_gradient!(HDIprob::PolynomialHiddenDynamicsInferenceProblem, g, p, lambda = 0.0, gamma = 100_000; resetg = true, verbose = false, abstol = 1e-6, reltol = 1e-6, maxiters = 15_000)
    # Extract info from struct
    polysystem = HDIprob.polysystem
    u0 = HDIprob.u0
    numvars = StaticPolynomials.nvariables(polysystem)
    numparams = StaticPolynomials.nparameters(polysystem)
    t = HDIprob.t
    data = HDIprob.data

    # reinitialize the forward ODE sensitivity problem
    u0[1:numvars] .= p[1:numvars]

    # Solve the Forward sensitivity problem
    tmpsenprob = remake(HDIprob.senprob, u0 = u0, p = view(p, numvars + 1:numparams + numvars), tspan = extrema(t))
    tmpsol = solve(tmpsenprob, Tsit5(), saveat = HDIprob.t, reltol = reltol, abstol = abstol, verbose = verbose, maxiters = maxiters)

    # Allow for safe inbounds 
    maxind = min(size(tmpsol, 2), size(data, 2))
    maxfld = min(size(tmpsol, 1), size(data, 1))
    Nfull = maxind * maxfld

    # Calculate the loss
    dataloss = 0.0
    @inbounds for i = 1:maxind
        solvec = tmpsol[i]
        for j = 1:maxfld
            dataloss += abs2(solvec[j] - data[j, i])
        end
    end
    dataloss /= Nfull
    if tmpsol.t[end] != maximum(t)
        dataloss += 1e20
    end

    # Calculate the gradient
    scale = 2 / Nfull
    if !isnothing(g)
        Ng = length(g)
        resetg && (g .= 0)
        @inbounds for i = 1:maxind
            solvec = tmpsol[i]
            for j = 1:maxfld # Loop over all the fields in the data
                Δ1 = scale * (solvec[j] - data[j, i])
                for s = 1:Ng
                    g[s] += Δ1 * solvec[numvars*s + j]
                end
            end
        end
    end

    # add the regularization to loss
    split1 = numvars
    split2 = numvars
    for i = 1:numvars
        split2 += HDIprob.splits[i]
        split1 += 1
        pnorm = norm(view(p, split1:split2))^2 - 1
        dataloss += gamma * pnorm^2 
        @inbounds for (ind, j) = enumerate(split1:split2)
            sp = sign(p[j])
            sp == 0 && (sp = 1.0)
            sqn = sqrt(1.0 + sum(HDIprob.pows[i][ind]))
            !isnothing(g) && (g[j] += 4 * gamma * pnorm * p[j] + lambda * sqn * sp)
            dataloss += lambda * abs(p[j]) * sqn
        end
        split1 = split2
    end

    return dataloss
end