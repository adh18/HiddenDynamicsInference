function calculate_KDEthresh(losses)
    U = kde_lscv(losses)
    pksind = argminima(U.density/maximum(U.density))
    pks, proms = Peaks.peakproms(pksind, U.density/maximum(U.density))
    train_loss_thresh = U.x[pks[proms .> 0.01]][1]
    return train_loss_thresh, U
end


# compute coefficient vector distances modulo sign flips and permutations
# returns Dcoeffs, the aligned distance matrix between models,
# coeffmat_symms, a list of copies of the original coefficient matrix with symmetries appied (e.g. sign flips and permutations)
# linear_transforms, a list of numvars x numvars matrices specifying the linear transformation for each symmetry
# best_symm, a matrix where the (i, j)th entry specifies which symmetry to apply to model j such that it is closest to model i
function compute_model_distances(coeffmat, pows::NTuple{numvars, Vector{NTuple{numvars, Int64}}}; dist=CosineDist(), numobs=numvars, signflip=false, permute=false) where numvars
    coeffmat_symms = [copy(coeffmat)]
    linear_transforms = [Matrix{Float64}(I, numvars, numvars)]
    if signflip
        itr = powerset(numobs+1:numvars, 1)
        for flip_inds in itr
            signs = Float64[]
            for i = 1:numvars
                append!(signs, [sum(x[flip_inds]) for x in pows[i]] .+ (i in flip_inds))
            end
            push!(coeffmat_symms, (-1) .^(signs) .* coeffmat)

            linear_transform = ones(numvars)
            linear_transform[flip_inds] .= -1.0
            linear_transform = diagm(linear_transform)
            push!(linear_transforms, linear_transform)
        end
    end
    num_symms = length(coeffmat_symms)

    if permute
        itr = permutations(numobs+1:numvars, numvars-numobs)
        for perm in itr
            if perm == numobs+1:numvars
                continue
            end
            var_perm = [1:numobs; perm]
            perm_pows, coeff_perm = coeffpermute(pows, (var_perm...,))
            if perm_pows == pows
                println(perm)
                println(perm_pows)
                for i = 1:num_symms
                    push!(coeffmat_symms, coeffmat_symms[i][coeff_perm, :])
                    push!(linear_transforms, linear_transforms[i][var_perm, :])
                end
            end
        end
    end
    num_symms = length(coeffmat_symms)

    Dcoeffs = fill(Inf, size(coeffmat, 2), size(coeffmat, 2))
    best_symm = zeros(Int64, size(Dcoeffs))
    for i = 1:num_symms
        Dcoeffs_i = pairwise(dist, coeffmat, coeffmat_symms[i])
        best_symm[Dcoeffs_i .< Dcoeffs] .= i
        Dcoeffs = min.(Dcoeffs, Dcoeffs_i)
    end
    Dcoeffs = (Dcoeffs + Dcoeffs') / 2
    Dcoeffs[diagind(Dcoeffs)] .= 0
    Dcoeffs[Dcoeffs .< 0] .= 0

    return Dcoeffs, coeffmat_symms, linear_transforms, best_symm
end

function coeffpermute(powers::NTuple{numvars, Vector{NTuple{numvars, Int64}}}, perm::NTuple{numvars, Int64}) where numvars
    splits = [0; cumsum(length.(powers))...]

    # reorder equations of hidden fields
    new_powers = Vector{NTuple{numvars, Int64}}[]
    coeff_perm = Int64[]
    for i = 1:numvars
        eqpowers = copy(powers[perm[i]])
        eqpowers_perm =  NTuple{numvars, Int64}[tple[[perm...]] for tple in eqpowers]
        push!(new_powers, sort(eqpowers_perm, lt=powsort))

        inds = sortperm(eqpowers_perm, lt = powsort)
        append!(coeff_perm, splits[perm[i]] .+ inds)
    end

    new_powers = tuple(new_powers...)
    return new_powers, coeff_perm
end

h_to_k(result, h) = all(result.heights .<= h) ? 1 : max(1, length(result.heights) - sum(result.heights .<= h)) + 1
k_to_h(result, k) = k >= length(result.heights)+1 ? -Inf : result.heights[length(result.heights) - k + 1]

# fit a sigmoidal curve to the function h(k) which is the hierarchical cluster distance vs. number of clusters
# use this fitted curve to determine the threshold range to return to the user
# most of the interesting clusters (not too sparse or too dense) should live in this range
function hierarchical_clustering_range(Dcoeffs; rng=Random.GLOBAL_RNG)
    result = hclust(Dcoeffs; linkage=:single, branchorder=:optimal)
    heights = reverse(result.heights)
    heights = heights[heights .> 0]

    xs = Float64.(collect(1:length(heights)))
    ys = log10.(heights)
    sigmoid(y, p) = (1 .+ p[3]*exp.(-p[1]*(y.-p[2]))).^(-p[4])

    p_est = zeros(4)
    tries = 0
    failed = true
    while failed && tries < 10
        failed = false
        try
            p_est = LsqFit.coef(LsqFit.curve_fit(sigmoid, ys, xs / maximum(xs), rand(rng, 4)))
        catch
            failed = true
        end
        tries += 1
    end
    if failed
        error("Hierarchical clustering height curve could not be fit with a sigmoid function.")
    end

    pred_xs = maximum(xs) * sigmoid(ys, p_est)
    a = p_est[3]
    b = p_est[2]
    c = p_est[1]
    d = p_est[4]

    #ddx = a^2*c^2*d*(d+1) * (1 .+ a*exp.(-c*(ys .- b))).^(-d-2) .* exp.(-2*c*(ys .- b))
    #ddx -= a*c^2*d * (1 .+ a*exp.(-c*(ys .- b))).^(-d-1) .* exp.(-c*(ys .- b))

    inflection_pt = log(a*d)/c + b
    elbow_pt = log(a/2*(3*d + 1 + sqrt((d+1)*(5*d+1))))/c + b

    #min_ind = findlast(ddx ./ minimum(ddx) .>= upper_bound)
    #max_ind = findlast(ddx ./ minimum(ddx) .>= lower_bound)
    min_ind = findlast(ys .>= inflection_pt)
    max_ind = findlast(ys .>= elbow_pt)
    Dmin = heights[max_ind]
    Dmax = heights[min_ind]
    kmin = h_to_k(result, Dmax)
    kmax = h_to_k(result, Dmin)

    return kmin, Dmin, kmax, Dmax, result, pred_xs
end

# align coefficients given list of coefficients with all possible symmetries applied
function align_coefficients(coeffmat_symms, best_symm; cluster=1:size(coeffmat_symms[1], 2))
    n = length(cluster)
    aligned_coeffmat = coeffmat_symms[1][:, cluster]
    cluster_best_symm = Matrix(hcat(unique(eachrow(best_symm[cluster, cluster]))...)')

    # loop over all greedy ways of symmetrizing each coefficient vector in the cluster
    alignment_ind = 1
    alignment_score = Inf
    for i = axes(cluster_best_symm, 1)
        coeffmat_symm = hcat([coeffmat_symms[cluster_best_symm[i, k]][:, cluster[k]] for k = 1:n]...)
        score = sqrt(sum(std(coeffmat_symm, dims=2).^2))
        if score < alignment_score
            aligned_coeffmat = coeffmat_symm
            alignment_score = score
            alignment_ind = i
        end
    end

    best_alignment = cluster_best_symm[alignment_ind, :]
    return aligned_coeffmat, best_alignment
end

# align coefficients searching over all sign flips and permutations
function align_coefficients(coeffmat, pows::NTuple{numvars, Vector{NTuple{numvars, Int64}}}; dist=CosineDist(), numobs=numvars, signflip=false, permute=false) where numvars
    Dcoeffs, coeffmat_symms, linear_transforms, best_symm = compute_model_distances(coeffmat, pows; dist=dist, numobs=numobs, signflip=signflip, permute=permute)
    aligned_coeffmat, best_alignment = align_coefficients(coeffmat_symms, best_symm)
    return aligned_coeffmat, best_alignment, Dcoeffs, coeffmat_symms, linear_transforms
end

function compute_coeff_var(coeffmat)
    coeff_var = zeros(size(coeffmat, 1))
    coeff_modes = zeros(size(coeffmat, 1))
    for i = axes(coeffmat, 1)
        full_xs = coeffmat[i, :]
        num_nonzero = sum(abs.(full_xs) .> 1e-10)
        xs = full_xs[abs.(full_xs) .> 1e-10]

        Qs = length(xs) < size(coeffmat, 2)/10 ? [0, 0, 0] : quantile(xs, [0.25, 0.5, 0.75])
        coeff_var[i] = abs(Qs[2]) < 1e-10 ? NaN : (1 + 1/sqrt(2*num_nonzero)) * (Qs[3] - Qs[1]) / abs(Qs[2])
        coeff_modes[i] = Qs[2]
    end

    return coeff_var, coeff_modes
end

function coeff_var_ranking(result, kmin, kmax; cluster_min_size=50, top_cluster_k=1, coeffmat_symms=nothing, best_symm=nothing, plot_freq=10)
    # Count number of times any leaf was in the top cluster at each level k
    search_ks = kmin:1:kmax
    leaf_in_top_cluster = zeros(length(result.order))
    top_cluster_inds = zeros(length(search_ks))
    for k_ind = eachindex(search_ks)
        cluster_labels = cutree(result; k=search_ks[k_ind])
        unique_cluster_labels = [x[1] for x in countmap(cluster_labels)]
        cluster_sizes = [x[2] for x in countmap(cluster_labels)]
        largest_cluster_label =  unique_cluster_labels[partialsortperm(cluster_sizes, top_cluster_k, rev=true)]
        largest_cluster_size = partialsort(cluster_sizes, top_cluster_k, rev=true)
        if largest_cluster_size < cluster_min_size
            break
        end
        #println("Index $(k_ind): Size $(largest_cluster_size)")

        sub_inds = findall(cluster_labels .== largest_cluster_label)
        leaf_in_top_cluster[sub_inds] .+= 1
        top_cluster_inds[k_ind] = largest_cluster_label
    end

    # Extract the best set of roots
    max_leaf_count = maximum(leaf_in_top_cluster)
    best_leaves = findall(leaf_in_top_cluster .== max_leaf_count)
    println("$(length(best_leaves)) / $(length(result.order))")

    # Sanity check that all the best roots are actually in the same cluster
    best_leaf_inclusion = []
    for k_ind = eachindex(search_ks)
        cluster_labels = cutree(result; k=search_ks[k_ind])
        cluster_sizes = [x[2] for x in countmap(cluster_labels)]
        largest_cluster_size = partialsort(cluster_sizes, top_cluster_k, rev=true)
        if largest_cluster_size < cluster_min_size
            break
        end

        push!(best_leaf_inclusion, (cluster_labels .== top_cluster_inds[k_ind])[best_leaves])
    end
    @assert all([length(unique(x)) for x in best_leaf_inclusion] .== 1)

    # Choose the first root as the path to follow down the tree
    best_leaf = best_leaves[1]

    # Plot the cluster at intermediate values of k along this best path
    coeff_var_sequence = []
    for k_ind = eachindex(search_ks)
        cluster_labels = cutree(result; k=search_ks[k_ind])
        best_label = cluster_labels[best_leaf]
        cluster_inds = findall(cluster_labels .== best_label)
        if length(cluster_inds) < cluster_min_size
            continue
        end

        cluster_coeffmat, _ = align_coefficients(coeffmat_symms, best_symm; cluster=cluster_inds)
        coeff_var, coeff_modes = compute_coeff_var(cluster_coeffmat)
        push!(coeff_var_sequence, coeff_var)
    end
    coeff_var_sequence_sort = hcat([sortperm(x) for x in coeff_var_sequence]...)
    coeff_var_sequence = hcat(coeff_var_sequence...)

    term_pos_counts = zeros(size(coeff_var_sequence_sort, 1), size(coeff_var_sequence_sort, 1))
    for i = axes(coeff_var_sequence_sort, 1)
        for j = axes(coeff_var_sequence_sort, 2)
            term_pos_counts[coeff_var_sequence_sort[i, j], i] += 1
        end
    end

    coeff_ranks = hcat([StatsBase.ordinalrank(coeff_var_sequence[:, i]) for i = axes(coeff_var_sequence, 2)]...)
    return coeff_ranks, term_pos_counts
end

# taken from https://vene.ro/blog/kemeny-young-optimal-rank-aggregation-in-python.html
function kemeny_young(ranks)
    n_candidates, n_voters = size(ranks)

    prgrm = Model()
    set_optimizer(prgrm, GLPK.Optimizer)
    @JuMP.variable(prgrm, 0<=X[1:n_candidates, 1:n_candidates], Int)

    # constraints for every pair
    for i = 1:n_candidates
        for j = i+1:n_candidates
            @constraint(prgrm, X[i, j] + X[j, i] == 1)
        end
    end

    # and for every cycle of length 3
    for i = 1:n_candidates
        for j = 1:n_candidates
            for k = 1:n_candidates
                if i != j && j != k && k != i
                    @constraint(prgrm, X[i, j] + X[j, k] + X[k, i] >= 1)
                end
            end
        end
    end

    # maximize C * X
    C = zeros(Int, n_candidates, n_candidates)
    for i = 1:n_candidates
        for j = i+1:n_candidates
            preference = ranks[i, :] - ranks[j, :]
            h_ij = sum(preference .< 0)  # prefers i to j
            h_ji = sum(preference .> 0)  # prefers j to i
            if h_ij > h_ji
                C[i, j] = h_ij - h_ji
            elseif h_ij < h_ji
                C[j, i] = h_ji - h_ij
            end
        end
    end
    @objective(prgrm, Min, sum(C*X))

    # optimize the objective
    optimize!(prgrm)

    opt_X = Int.(JuMP.value.(X))
    obj_val = sum(C.*opt_X)
    aggr_rank = sum(opt_X, dims=1)[:] .+ 1
    return aggr_rank, opt_X, obj_val
end