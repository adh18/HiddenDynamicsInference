"""
Optimize the parameter vector p based on the given HDI problem
"""
function optimize(HDIprob, p0; lambda = 0.0, gamma = 100_000, showprogress = true, Nflux = 50_000, Noptim = 25_000, eta = 1e-2, kwargs...)
    g = similar(p0)
    p = copy(p0)
    bestp = copy(p)
    bestval = value_gradient!(HDIprob, g, p, lambda, gamma; resetg = true, kwargs...)

    # Set up the AdaBelief loop
    opt = Flux.Optimisers.AdaBelief(eta)
    st = Flux.Optimisers.setup(opt, p)
    showprogress && (pm = Progress(Nflux))
    for i = 1:Nflux
        val = value_gradient!(HDIprob, g, p, lambda)
        st, p = Flux.Optimisers.update!(st, p, g)
        if val < bestval 
            bestval = val
            bestp .= p
        end
        showprogress && next!(pm, showvalues = ((:loss, val), (:bestloss, bestval)))
    end
    showprogress && ProgressMeter.finish!(pm) # If stopped early

    # Set up the Optim Loop
    pm = Progress(Noptim)
    optimcb = showprogress ? x -> (next!(pm, showvalues = ((:iter, x.iteration), (:curr_loss, x.value))); flush(stdout); false) : nothing
    constrained_fg! = (f, g, p) -> value_gradient!(HDIprob, g, p, lambda)
    optres = Optim.optimize(Optim.only_fg!(constrained_fg!), bestp, Optim.LBFGS(m = 100),
        Optim.Options(callback = optimcb, g_tol = 1e-8, iterations = Noptim, store_trace = false, show_trace = false, show_every = 1, allow_f_increases = true)
    )    
    showprogress && ProgressMeter.finish!(pm) # If stopped early
    return optres.minimizer, optres.minimum
end

randinit(HDIprop::PolynomialHiddenDynamicsInferenceProblem) = randinit(Random.default_rng(), HDIprop)
function randinit(rng, HDIprop::PolynomialHiddenDynamicsInferenceProblem)
    np = StaticPolynomials.nparameters(HDIprop.polysystem)
    nv = StaticPolynomials.nvariables(HDIprop.polysystem)
    splits = HDIprop.splits
    p0 = vcat(ones(nv), randn(rng, np))
    p0[end - nv + 1:end] .*= 1e-5
    p0[1:size(HDIprop.data, 1)] .= HDIprop.data[:, 1]
    return p0
end