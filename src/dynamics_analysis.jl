function dataloss(HDIprob, p, k; kwargs...)
    numflds = StaticPolynomials.nvariables(HDIprob.polysystem)
    train_sol  = solve(HDIprob.odeprob, Tsit5(), u0 = view(p, 1:numflds), p = view(p, numflds + 1:length(p)), saveat = HDIprob.t, tspan = extrema(HDIprob.t); kwargs...)
    if train_sol.retcode == ReturnCode.Success
        return sqrt(mean(abs2, train_sol[k, :] .- HDIprob.data[k, :])) / sqrt(mean(abs2, HDIprob.data[k, :] .- mean(HDIprob.data[k, :])))
    else
        return Inf
    end
end

# n is the number of delay embeddings 
# tau is the time offset in the delay embedding
# thresh is a tolerance for how close the delay time points are to |td - (t - tau)| < thresh
# default is 10% of tau
function delay_embed(t, data; tau = 1.0, n = 1, thresh = tau/10)
    numt = length(t)
    # Map the time t to the index at t - tau 
    tmap = map(curt -> searchsortedfirst(t, curt - tau), t)

    # Preallocate vectors 
    delayinds = Matrix{Int64}(undef, n + 1, numt - n)
    delayt = Vector{Float64}(undef, numt - n)
    delaydata = Matrix{Float64}(undef, n + 1, numt - n)
    tinds = Vector{Int64}(undef, n + 1)
    gooddelays = trues(numt - n)
    for i = 1:numt - n # maximum number of delays 
        cur_tind = i + n # current time ind
        delayt[i] = t[cur_tind] # current time
        delayinds[1, i] = cur_tind # current delay ind
        delaydata[1, i] = data[cur_tind] # current data at delay ind
        t1 = t[cur_tind]
        for j = 2:n + 1
            cur_tind = tmap[cur_tind] #delay time ind
            t2 = t[cur_tind] # delay time 
            gooddelays[i] &= abs(t1 - t2 - tau) < thresh # check delay is within tolerance
            !gooddelays[i] && break  
            delayinds[j , i] = cur_tind # current delay ind
            delaydata[j, i] = data[cur_tind] # current data at delay ind
            t1 = t2 
        end
    end
    # Crop to only the good inds
    delayinds = delayinds[:, gooddelays]
    delaydata = delaydata[:, gooddelays]
    delayt = delayt[gooddelays]
    return tmap, delayinds, delayt, delaydata
end

function hausdorff_distance(traj1, traj2)
    dmat = Distances.pairwise(SqEuclidean(), traj1, traj2)
    md1 = minimum(dmat, dims = 1)
    md2 = minimum(dmat, dims = 2)
    ave_err = sqrt(max(mean(md1), mean(md2)))
    max_err = sqrt(max(maximum(md1), maximum(md2)))
   
    return ave_err, max_err
end

function delay_embedding_error(t_traj, traj, t_data, data; n = 1, offset = 2, tau = mean(diff(t_data)), thresh = min(minimum(diff(t_traj)), minimum(diff(t_data))) / 2)
    delay_traj = delay_embed(t_traj, traj; tau = offset*tau, n = n, thresh = thresh)[4]
    numtraj = size(delay_traj, 2)
    
    delay_data = delay_embed(t_data, data; tau = offset*tau, n = n, thresh = thresh)[4]
    numdata = size(delay_data, 2)

    if numtraj == 0 || numdata == 0
        print("Threshold $thresh for delay embedding is too low")
        return Inf, Inf
    end
    
    center = mean(delay_data, dims=2)
    delay_traj .-= center
    delay_data .-= center

    total_norm = sqrt(sum(abs2, delay_data) / numdata)
    delay_traj ./= total_norm
    delay_data ./= total_norm

    ave_err, max_err = hausdorff_distance(delay_traj, delay_data)
    return ave_err, max_err
end

function compute_limit_cycle(traj_t, traj; tol = 1e-1, reps_checked = 2)
    # Normalize the trajectory
    traj_normalized = traj .- mean(traj, dims=2)
    traj_normalized ./= std(traj_normalized, dims=2)

    # Compute the distance between the end point and all points in phase space
    dists = sqrt.(sum(abs2, traj_normalized .- traj_normalized[:, end], dims=1)[:])
    
    # Locate the position of local minima in the distances 
    min_inds = reverse(Peaks.argminima(dists))
    # No local minima => no limit cycle retrun empty vectors
    isempty(min_inds) && return Vector{eltype(traj_t)}(), Vector{eltype(traj)}()

    # 
    for cyclestart = min_inds
        cyclelen = length(traj_t) - cyclestart + 1 # In steps

        # cannot repeat limit cycle because time series is too short
        length(traj_t) <= (reps_checked+1)*(cyclelen-1) && break 

        # Loop backwards in the solution to check persistence of limitcylce
        no_cycle_found = false
        for rep1 = 1:reps_checked + 1
            traj_rep1 = traj_normalized[:, end - rep1*(cyclelen - 1):end - (rep1 - 1)*(cyclelen - 1)]
            for rep2 = rep1 + 1:reps_checked + 1
                traj_rep2 = traj_normalized[:, end - rep2*(cyclelen - 1):end - (rep2 - 1)*(cyclelen - 1)]
                # Compare distance between succesive limit cycles
                traj_rep_dists = sqrt.(sum((traj_rep1 - traj_rep2).^2, dims=1)[:])
                # If the trajectories too far apart then no limit cycle has been found
                no_cycle_found = mean(traj_rep_dists) .>= tol
                no_cycle_found && break
            end
            no_cycle_found && break
        end
        
        # limit cycle was repeated a "reps_checked" number of times
        if !no_cycle_found
            limit_cycle = traj[:, cyclestart:end]
            limit_t = traj_t[cyclestart:end] .- traj_t[cyclestart]
            return limit_t, limit_cycle
        end
    end

    # No limit cycle found 
    return Vector{eltype(traj_t)}(), Vector{eltype(traj)}()
end

function find_limit_cycle(HDIprob, p, fullt, fulldata; dt = 1e-3, tspan = (0.0, 250.0), period_len = 38.0, kwargs...)
    numflds = StaticPolynomials.nvariables(HDIprob.polysystem)
    numvars = StaticPolynomials.nvariables(HDIprob.polysystem)
    full_sol  = solve(HDIprob.odeprob, Tsit5(), u0 = view(p, 1:numflds), p = view(p, numflds + 1:length(p)), saveat = dt, tspan = tspan; kwargs...)
    
    # Check if the solution blows up
    full_sol.retcode != ReturnCode.Success && return NaN, NaN, "failed to simulate state over large timespan"

    # Check if converged to a fixed point
    du = HDIprob.polysystem(full_sol[end], view(p, numvars + 1:length(p)))
    sqrt(sum(abs2, du)) < 1e-5 && return NaN, NaN, "converged to fixed point"

    # Check for the limit cycle
    limit_t, limit_cycle = compute_limit_cycle(full_sol.t, Matrix(full_sol))
    isempty(limit_cycle) && return NaN, NaN, "limit cycle not found"

    # Check the length of the limit cycle
    limit_cycle_dt = limit_t[end] - limit_t[1]
    (3*period_len/4 < limit_cycle_dt < 5*period_len/4) || return NaN, NaN, "limit cycle too short or too long in time"

    # Check the limit cycle magnitude
    std_ratio = sqrt(mean(abs2, limit_cycle[1, :] .- mean(HDIprob.data[1, :]))) / sqrt(mean(abs2, HDIprob.data[1, :] .- mean(HDIprob.data[1, :])))
    0.8 < std_ratio < 1.2 || return NaN, NaN, "limit cycle too small or too large in magnitude"

    # Calculate the extended limit cycle 
    limit_t2 = [limit_t; limit_t[2:end].+limit_t[end]]
    limit_cycle2 = [limit_cycle limit_cycle[:, 2:end]]

    # Compute the train and test delay loss
    # This is super slow? 
    n = 1
    offset = 3
    tau = mean(diff(HDIprob.t))
    
    train_delay_loss = delay_embedding_error(limit_t2, limit_cycle2[1, :], HDIprob.t, HDIprob.data[1, :]; n = n, offset = offset, tau = tau, thresh = tau/1000)[2]
    test_delay_loss  = delay_embedding_error(limit_t2, limit_cycle2[1, :], fullt, fulldata[1, :]; n = n, offset = offset, tau = tau, thresh = tau/1000)[2]

    return train_delay_loss, test_delay_loss, "limit cycle found"
end