# Functions for rescaling 
function scale_coefficients!(HDIprob, p, k, scale)
    numvars = StaticPolynomials.nvariables(HDIprob.polysystem)
    1 <= k <= numvars || error("k = $k must be in the range of the number of variables 1:$(numvars)")

    splits  = cumsum([numvars; HDIprob.splits...])
    p[splits[k] + 1:splits[k+1]] .*= scale
    for i = 1:numvars
        # Rescale term coefficient 
        p[splits[i] + 1:splits[i+1]] ./= scale .^ getindex.(HDIprob.pows[i], k)
        nrm = norm(p[splits[i] + 1:splits[i+1]])
        p[splits[i] + 1:splits[i+1]] ./= nrm
        # tau 
        p[splits[end] + i] *= nrm
    end
    # IC
    p[k] *= scale
end

function shift_coefficients!(HDIprob, p, k, shift)
    numvars = StaticPolynomials.nvariables(HDIprob.polysystem)
    splits  = cumsum([numvars; HDIprob.splits...])
    shifted_optp = copy(p)

    for i = 1:numvars
        # all unique power combinations in equation i with exception of kth index
        pows_i_minus_k = [x[1:numvars .!= k] for x in HDIprob.pows[i]]
        pows_i_minus_k_unique = unique(pows_i_minus_k)
        for pow_minus_k in pows_i_minus_k_unique
            # get all powers of k in ith equation for given power combination
            # e.g. x*y*z, x^2*y*z, x^5*y*z are all powers of x for power combination y*z
            inds = findall([pow == pow_minus_k for pow in pows_i_minus_k])
            kpows = getindex.(HDIprob.pows[i][inds], k)
            inds = inds[sortperm(kpows)]
            kmaxpow = maximum(kpows)
            if kpows != 0:kmaxpow
                return false
            end

            for r = 1:kmaxpow
                # coefficient where power of k is equal to r
                c = p[splits[i] + inds[r+1]]
                # use binomial formula to update all coefficients of lower powers
                shifted_optp[splits[i] .+ inds[1:r]] .+= c * binomial.(r, 0:r-1) .* ((-shift) .^(r:-1:1))
            end
        end

        nrm = norm(shifted_optp[splits[i] + 1:splits[i+1]])
        shifted_optp[splits[i] + 1:splits[i+1]] ./= nrm
        shifted_optp[splits[end - 1] + i] *= nrm
    end

    shifted_optp[k] += shift
    p .= shifted_optp
    return true
end

function std_scale_coefficients!(HDIprob, p, k; refstd=nothing, tspan=extrema(HDIprob.t), saveat=HDIprob.t, transient=tspan[2]/4 + 3*tspan[1]/4)
    numvars = StaticPolynomials.nvariables(HDIprob.polysystem)
    newsol  = solve(HDIprob.odeprob, Tsit5(), u0 = view(p, 1:numvars), p = view(p, numvars + 1:length(p)), tspan = tspan, saveat = saveat)
    
    # If the integration failed retutrn 
    newsol.retcode != :Success && return false

    # Calculate the std of the first equation is needed
    refstd = isnothing(refstd) ? std(newsol[1, newsol.t .> transient]) : refstd

    # Rescale fields to have the same std 
    scale = refstd / std(newsol[k, newsol.t .> transient])

    # If scale is too large (std is close to 0) return 
    scale > 1e10 && return false

    scale_coefficients!(HDIprob, p, k, scale)
    return true
end

function center_coefficients!(HDIprob, p, k; tspan=extrema(HDIprob.t), saveat=HDIprob.t, transient=tspan[2]/4 + 3*tspan[1]/4)
    numvars = StaticPolynomials.nvariables(HDIprob.polysystem)
    newsol  = solve(HDIprob.odeprob, Tsit5(), u0 = view(p, 1:numvars), p = view(p, numvars + 1:length(p)), tspan = tspan, saveat = saveat)
    
    # If the integration failed retutrn 
    newsol.retcode != :Success && return false

    shift = -mean(newsol[k, newsol.t .> transient])
    shift_coefficients!(HDIprob, p, k, shift)
end
