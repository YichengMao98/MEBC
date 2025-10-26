# ----------------- constants and helpers -----------------

clamp01(x) = x < 0 ? 0.0 : (x > 1 ? 1.0 : x)

baseline_alarm(psi, alpha) = (alpha <= 0) ? error("alpha>0") :
                             (psi = clamp01(psi); 1 - (1 - psi)^(1/alpha))

function compute_psi!(psi::Vector{Float64}, Istar::Vector{Int}, N::Int;
                      mechanism::Symbol=:memoryless, k_max::Union{Nothing,Int}=nothing,
                      lambda_P::Union{Nothing,Float64}=nothing,
                      lambda_E::Union{Nothing,Float64}=nothing,
                      lambda_R::Union{Nothing,Float64}=nothing)   
    τ = length(Istar)
    psi[1] = 0.0
    if mechanism === :memoryless
        @inbounds for t in 2:τ
            psi[t] = Istar[t-1] / N
        end
    elseif mechanism === :sliding
        isnothing(k_max) && error("k_max needed for :sliding")
        @inbounds for t in 2:τ
            m = min(k_max, t-1)
            psi[t] = (m == 0) ? 0.0 : (sum(@view Istar[(t-m):(t-1)]) / (m*N))
        end
    elseif mechanism === :powerlaw
        isnothing(lambda_P) && error("lambda_P needed for :powerlaw")
        @inbounds for t in 2:τ
            L = t-1
            if L == 0
                psi[t] = 0.0
            else
                num = 0.0
                for j in 1:L
                    w = j^(-lambda_P)
                    num += w * Istar[t-j]
                end
                psi[t] = num/N   
            end
        end
    elseif mechanism === :exponential
        isnothing(lambda_E) && error("lambda_E needed for :exponential")
        @inbounds for t in 2:τ
            L = t-1
            if L == 0
                psi[t] = 0.0
            else
                num = 0.0
                for j in 0:(L-1)
                    w = exp(-lambda_E * j)
                    num += w * Istar[t-1-j]
                end
                psi[t] = num/N   
            end
        end
    elseif mechanism === :reciprocal
        isnothing(lambda_R) && error("lambda_R needed for :reciprocal")
        (lambda_R <= 0) && error("lambda_R must be > 0")
        @inbounds for t in 2:τ
            L = t-1
            if L == 0
                psi[t] = 0.0
            else
                num = 0.0
                for j in 0:(L-1)
                    w = 1.0 / (1.0 + lambda_R * j)   
                    num += w * Istar[t-1-j]
                end
                psi[t] = num / N                   
            end
        end
    else
        error("unknown mechanism")
    end
    return psi
end

function loglik_aug(Istar::Vector{Int}, Rstar::Vector{Int}, N::Int, I0::Int;
                    beta::Float64, alpha::Float64, gamma::Float64,
                    psi::Vector{Float64}, return_vec::Bool=false)
    τ = length(Istar)
    (beta <= 0 || alpha <= 0 || gamma <= 0) && return return_vec ? vcat(fill(-Inf, τ)) : -Inf
    length(Rstar) == τ || error("length(Rstar)!=τ")
    S = N - I0
    Iinf = I0
    pIR = clamp01(1 - exp(-gamma))
    if return_vec
        ll_vec = Vector{Float64}(undef, τ)
        @inbounds for t in 1:τ
            (Istar[t] < 0 || Istar[t] > S) && return vcat(fill(-Inf, τ))
            pSI = clamp01(1 - exp(-beta * (1 - baseline_alarm(psi[t], alpha)) * (Iinf / N)))
            l1 = logpdf(Binomial(S, pSI), Istar[t])
            (Rstar[t] < 0 || Rstar[t] > Iinf) && return vcat(fill(-Inf, τ))
            l2 = logpdf(Binomial(Iinf, pIR), Rstar[t])
            ll_vec[t] = l1 + l2
            S  -= Istar[t]
            Iinf += Istar[t] - Rstar[t]
            (S < 0 || Iinf < 0) && return vcat(fill(-Inf, τ))
        end
        return ll_vec
    else
        ll_sum = 0.0
        @inbounds for t in 1:τ
            (Istar[t] < 0 || Istar[t] > S) && return -Inf
            pSI = clamp01(1 - exp(-beta * (1 - baseline_alarm(psi[t], alpha)) * (Iinf / N)))
            l1 = logpdf(Binomial(S, pSI), Istar[t])
            (Rstar[t] < 0 || Rstar[t] > Iinf) && return -Inf
            l2 = logpdf(Binomial(Iinf, pIR), Rstar[t])
            ll_sum += (l1 + l2)
            S  -= Istar[t]
            Iinf += Istar[t] - Rstar[t]
            (S < 0 || Iinf < 0) && return -Inf
        end
        return ll_sum
    end
end

function rstar_block_update!(R::Vector{Int}, Istar::Vector{Int}, N::Int, I0::Int;
                             beta::Float64, alpha::Float64, gamma::Float64,
                             psi::Vector{Float64},
                             nUpdates::Int=200)
    τ = length(R)
    ll_cur = loglik_aug(Istar, R, N, I0; beta=beta, alpha=alpha, gamma=gamma, psi=psi)
    @inbounds for _ in 1:nUpdates
        moveType = rand(1:3)
        Rprop = copy(R)
        g = 0.0
        if moveType == 1
            addIdx = rand(1:τ)
            Rprop[addIdx] += 1
            newPossibleSubtract = count(>(0), Rprop)
            g = -log(newPossibleSubtract) + log(τ)
        elseif moveType == 2
            possibleSubtract = findall(>(0), R)
            isempty(possibleSubtract) && continue
            subtractIdx = rand(possibleSubtract)
            addIdx = rand(1:τ)
            Rprop[subtractIdx] -= 1
            Rprop[addIdx]      += 1
            newPossibleSubtract = count(>(0), Rprop)
            g = -log(newPossibleSubtract) + log(length(possibleSubtract))
        else
            possibleSubtract = findall(>(0), R)
            isempty(possibleSubtract) && continue
            subtractIdx = rand(possibleSubtract)
            Rprop[subtractIdx] -= 1
            g = -log(τ) + log(length(possibleSubtract))
        end
        ll_prop = loglik_aug(Istar, Rprop, N, I0; beta=beta, alpha=alpha, gamma=gamma, psi=psi)
        loga = (ll_prop - ll_cur) + g
        if log(rand()) < loga
            R .= Rprop
            ll_cur = ll_prop
        end
    end
    return R, ll_cur
end



function logprior_cont(θ::Vector{Float64}, fit_mech::Symbol)
    (θ[1] <= 0 || θ[2] <= 0 || θ[3] <= 0) && return -Inf
    lp = logpdf(Uniform(0,100), θ[1]) +
         logpdf(Gamma(0.1,0.1), θ[2]) +
         logpdf(Gamma(aa, 1/bb), θ[3])
    if fit_mech === :powerlaw
        (θ[4] <= 0) && return -Inf
        lp += logpdf(Uniform(0,100), θ[4])
    elseif fit_mech === :exponential
        (θ[4] <= 0) && return -Inf
        lp += logpdf(Uniform(0,100), θ[4])
    elseif fit_mech === :reciprocal                    
        (θ[4] <= 0) && return -Inf
        lp += logpdf(Uniform(0,100), θ[4])
    end
    return lp
end




function update_param!(param_idx::Int, theta_cont::Vector{Float64}, ll::Float64,
                       lp::Float64, Istar_obs, Rstar, N, I0, psi,
                       step_size::Float64, fit_mech::Symbol)
    
    param_cur = theta_cont[param_idx]
    param_prop = rand(Normal(param_cur, step_size))
    
    theta_prop = copy(theta_cont)
    theta_prop[param_idx] = param_prop
    if param_prop <= 0
        return ll, lp, false
    end

    ll_prop = loglik_aug(Istar_obs, Rstar, N, I0;
                         beta = (param_idx == 1) ? param_prop : theta_cont[1],
                         alpha = (param_idx == 2) ? param_prop : theta_cont[2],
                         gamma = (param_idx == 3) ? param_prop : theta_cont[3],
                         psi=psi)
    
    lp_prop = logprior_cont(theta_prop, fit_mech)

    if log(rand()) < (ll_prop + lp_prop - ll - lp)
        theta_cont[param_idx] = param_prop
        return ll_prop, lp_prop, true
    end
    
    return ll, lp, false
end

mutable struct OnlineCovariance{T<:AbstractFloat}
    n::Int          # Number of samples
    mean::Vector{T} # Mean vector
    cov::Matrix{T}  # Covariance matrix (unnormalized)
    d::Int          # Dimension
end

# Constructor
function OnlineCovariance(d::Int)
    OnlineCovariance(0, zeros(d), zeros(d, d), d)
end

# Incremental update function
function update_online_cov!(oc::OnlineCovariance, new_sample::Vector{<:Number})
    oc.n += 1
    if oc.n == 1
        oc.mean = new_sample
    else
        old_mean = oc.mean
        oc.mean = old_mean + (new_sample .- old_mean) ./ oc.n
        oc.cov = oc.cov + (new_sample .- old_mean) * (new_sample .- oc.mean)'
    end
    return
end

# Helper function to extract diagonal variances from covariance matrix
function get_diagonal_variances(online_cov::OnlineCovariance)
    if online_cov.n > 1
        cov_matrix = online_cov.cov ./ (online_cov.n - 1)
        return diag(cov_matrix)
    else
        return zeros(online_cov.d)
    end
end

function mcmc_one_chain_with_Rstar!(Istar_obs::Vector{Int}, N::Int, I0::Int;
                                      fit_mech::Symbol=:memoryless,
                                      n_iter::Int=1_000_000,
                                      initθ::Vector{Float64},
                                      KMAX_UPPER::Int=30,                
                                      k_max_fixed::Union{Nothing,Int}=nothing)  
    tau = length(Istar_obs)

    d_cont = (fit_mech in [:memoryless, :sliding]) ? 3 : 4
    d_total = d_cont
    theta_cont = initθ[1:d_cont]

    gamma0 = theta_cont[3]
    pIR0 = clamp(1 - exp(-gamma0), 0, 1)
    R = zeros(Int, tau)
    S = N - I0; Iinf = I0
    for t in 1:tau
        R[t] = min(Iinf, round(Int, Iinf*pIR0))
        S -= Istar_obs[t]
        Iinf += Istar_obs[t] - R[t]
        (S < 0 || Iinf < 0) && error("Bad initial Rstar; change init.")
    end

    psi = zeros(Float64, tau)
    if fit_mech === :memoryless
        compute_psi!(psi, Istar_obs, N; mechanism=:memoryless)
    elseif fit_mech === :powerlaw
        compute_psi!(psi, Istar_obs, N; mechanism=:powerlaw, lambda_P=theta_cont[4])
    elseif fit_mech === :exponential
        compute_psi!(psi, Istar_obs, N; mechanism=:exponential, lambda_E=theta_cont[4])
    elseif fit_mech === :reciprocal
        compute_psi!(psi, Istar_obs, N; mechanism=:reciprocal, lambda_R=theta_cont[4])
    elseif fit_mech === :sliding
        isnothing(k_max_fixed) && error("For :sliding, please pass a fixed k_max via k_max_fixed=...")
        (k_max_fixed < 1 || k_max_fixed > KMAX_UPPER) && @warn "k_max_fixed outside [1,$KMAX_UPPER]; using as-is."
        compute_psi!(psi, Istar_obs, N; mechanism=:sliding, k_max=k_max_fixed)
    else
        error("Unknown fit_mech")
    end

    ll = loglik_aug(Istar_obs, R, N, I0; beta=theta_cont[1], alpha=theta_cont[2], gamma=theta_cont[3], psi=psi)
    lp = logprior_cont(theta_cont, fit_mech)

    samples = Array{Float32}(undef, n_iter, d_total)

    burnin = n_iter ÷ 2
    thin = 10
    n_saved_loglik = (n_iter - burnin) ÷ thin
    loglik_aug_vecs = Array{Float32}(undef, n_saved_loglik, tau)
    loglik_save_counter = 0

    if d_cont == 3
        # β, α, γ
        initial_stds = [0.005, 0.00005, 0.005]
    else
        # β, α, γ, λ
        initial_stds = [0.005, 0.00005, 0.005, 0.005]
    end
    Sigma_initial = Diagonal(initial_stds.^2) / 2

    online_cov = OnlineCovariance(d_cont)
    adaptation_cutoff = n_iter ÷ 2
    prop_dist_fixed = nothing
    prop_std_fixed = nothing

    accept_count = 0

    t0 = now()
    for i in 1:n_iter
        R, ll = rstar_block_update!(R, Istar_obs, N, I0;
                                    beta=theta_cont[1], alpha=theta_cont[2], gamma=theta_cont[3],
                                    psi=psi, nUpdates=200)

        in_adaptation_phase = i <= adaptation_cutoff

        theta_prop = copy(theta_cont)
        if in_adaptation_phase
            update_online_cov!(online_cov, theta_cont)
            if online_cov.n > 20000
                try
                    cov_dat = online_cov.cov ./ (online_cov.n - 1) + (1e-6) * I(online_cov.d)
                    Sigma_adaptive = (cov_dat * (2.38)^2) / online_cov.d
                    prop_dist = 0.8 * MvNormal(zeros(online_cov.d), Sigma_adaptive) +
                                0.2 * MvNormal(zeros(online_cov.d), Sigma_initial)
                catch
                    prop_dist = MvNormal(zeros(online_cov.d), Sigma_initial)
                end
            else
                prop_dist = MvNormal(zeros(online_cov.d), Sigma_initial)
            end
            if i == adaptation_cutoff && online_cov.n > 20000
                prop_dist_fixed = prop_dist
                try
                    cov_dat = online_cov.cov ./ (online_cov.n - 1) + (1e-6) * I(online_cov.d)
                    Sigma_adaptive = (cov_dat * (2.38)^2) / online_cov.d
                    actual_sigma = 0.8 * Sigma_adaptive + 0.2 * Sigma_initial
                    prop_std_fixed = sqrt.(diag(actual_sigma))
                catch
                    prop_std_fixed = sqrt.(diag(Sigma_initial))
                end
            end
        else
            prop_dist = prop_dist_fixed !== nothing ? prop_dist_fixed : MvNormal(zeros(online_cov.d), Sigma_initial)
        end

        theta_prop += rand(prop_dist)

        if all(theta_prop .> 0)
            lp_prop = logprior_cont(theta_prop, fit_mech)
            if lp_prop !== -Inf
                old_psi = copy(psi)
                if d_cont == 4
                    if fit_mech === :powerlaw
                        compute_psi!(psi, Istar_obs, N; mechanism=:powerlaw,   lambda_P=theta_prop[4])
                    elseif fit_mech === :exponential
                        compute_psi!(psi, Istar_obs, N; mechanism=:exponential,lambda_E=theta_prop[4])
                    elseif fit_mech === :reciprocal
                        compute_psi!(psi, Istar_obs, N; mechanism=:reciprocal, lambda_R=theta_prop[4])
                    end
                end

                ll_prop = loglik_aug(Istar_obs, R, N, I0;
                                     beta=theta_prop[1], alpha=theta_prop[2], gamma=theta_prop[3], psi=psi)

                if log(rand()) < (ll_prop + lp_prop - ll - lp)
                    theta_cont .= theta_prop
                    ll = ll_prop
                    lp = lp_prop
                    accept_count += 1
                else
                    if d_cont == 4
                        psi .= old_psi
                    end
                end
            end
        end

        samples[i, :] = Float32.(theta_cont)

        if i > burnin && (i - burnin) % thin == 0
            loglik_save_counter += 1
            loglik_aug_vecs[loglik_save_counter, :] =
                Float32.(loglik_aug(Istar_obs, R, N, I0;
                                    beta=theta_cont[1], alpha=theta_cont[2], gamma=theta_cont[3],
                                    psi=psi, return_vec=true))
        end

        if i % 1000 == 0
            el = Dates.value(now() - t0) / 1000
            accept_rate = i > 0 ? accept_count / i : 0.0
            median_theta = online_cov.n > 0 ? online_cov.mean : theta_cont

            if in_adaptation_phase
                if online_cov.n > 20000
                    try
                        cov_dat = online_cov.cov ./ (online_cov.n - 1) + (1e-6) * I(online_cov.d)
                        Sigma_adaptive = (cov_dat * (2.38)^2) / online_cov.d
                        actual_sigma = 0.8 * Sigma_adaptive + 0.2 * Sigma_initial
                        prop_std = sqrt.(diag(actual_sigma))
                    catch
                        prop_std = sqrt.(diag(Sigma_initial))
                    end
                else
                    prop_std = sqrt.(diag(Sigma_initial))
                end
            else
                prop_std = prop_std_fixed !== nothing ? prop_std_fixed : sqrt.(diag(Sigma_initial))
            end

            phase_info = in_adaptation_phase ? " [ADAPT]" : " [FIXED]"

            if d_cont == 4
                @info(@sprintf("[%s] iter %d/%d elapsed=%.1fs, rate=%.3f, medians=[%.3f, %.5f, %.3f, %.3f], std=[%.4f, %.6f, %.4f, %.4f]%s",
                                 String(fit_mech), i, n_iter, el, accept_rate,
                                 median_theta[1], median_theta[2], median_theta[3], median_theta[4],
                                 prop_std[1],    prop_std[2],    prop_std[3],    prop_std[4], phase_info))
            else
                @info(@sprintf("[%s] iter %d/%d elapsed=%.1fs, rate=%.3f, medians=[%.3f, %.5f, %.3f], std=[%.4f, %.6f, %.4f]%s",
                                 String(fit_mech), i, n_iter, el, accept_rate,
                                 median_theta[1], median_theta[2], median_theta[3],
                                 prop_std[1],    prop_std[2],    prop_std[3], phase_info))
            end
        end
    end

    return samples, loglik_aug_vecs
end


function initθ_for_chain(tag::Symbol)
    init_beta  = rand(Uniform(0.6, 1))
    init_alpha = rand(Uniform(0.004, 0.005))
    init_gamma = rand(Gamma(aa, 1/bb))

    if tag === :memoryless
        return [init_beta, init_alpha, init_gamma]
    elseif tag === :sliding
        return [init_beta, init_alpha, init_gamma]
    elseif tag === :powerlaw
        init_lambdaP = rand(Uniform(0.6, 0.7))
        return [init_beta, init_alpha, init_gamma, init_lambdaP]
    elseif tag === :exponential
        init_lambdaE = rand(Uniform(0.2, 0.3))
        return [init_beta, init_alpha, init_gamma, init_lambdaE]
    elseif tag === :reciprocal                       
        init_lambdaR = rand(Uniform(0.4, 0.6))
        return [init_beta, init_alpha, init_gamma, init_lambdaR]
    else
        error("Invalid model tag.")
    end
end

# --- Gelman-Rubin R-hat implementation ---
function rhat_gelman_rubin(chains_2d::AbstractMatrix)
    m, n = size(chains_2d)

    chain_means = vec(mean(chains_2d, dims=2))
    chain_vars  = vec(var(chains_2d, dims=2, corrected=true))

    W = mean(chain_vars)

    if !isfinite(W) || W <= 0
        return 1.0
    end

    B = n * var(chain_means, corrected=true)

    var_hat = ((n - 1) / n) * W + (1.0 / n) * B

    var_hat = max(var_hat, 1e-16)
    Rhat = sqrt(var_hat / W)

    return max(Rhat, 1.0)
end

function compute_waic(all_loglik_aug_vecs)
    combined_logliks = vcat([ll[BURN_IN+1:end, :] for ll in all_loglik_aug_vecs]...)

    if any(isinf, combined_logliks)
        @warn "Encountered infinite log-likelihoods. WAIC will be -Inf."
        return -Inf
    end

    lppd = sum(log.(mean(exp.(combined_logliks), dims=1)))
    pwaic = sum(var(combined_logliks, dims=1))

    waic = -2 * lppd + 2 * pwaic
    return waic
end

function summarize_posterior(samples, header_cont)
    samples_burned = samples[BURN_IN+1:end, :]
    summary_dict = Dict{String, Any}()

    for (i, p_name) in enumerate(header_cont)
        param_samples = samples_burned[:, i]
        if p_name == "k_max"
            mode_val = mode(round.(Int, param_samples))
            mode_freq = count(x -> x == mode_val, round.(Int, param_samples)) / length(param_samples)
            summary_dict[p_name] = Dict("mode" => mode_val, "frequency" => mode_freq)
        else
            median_val = median(param_samples)
            ci_95 = quantile(param_samples, [0.025, 0.975])
            summary_dict[p_name] = Dict("median" => median_val, "95%CI_low" => ci_95[1], "95%CI_high" => ci_95[2])
        end
    end
    return summary_dict
end

function get_hdpi(x, alpha=0.95)
    sorted_x = sort(x)
    n = length(sorted_x)
    n_hdi = floor(Int, alpha * n)
    if n_hdi == 0
        return [sorted_x[1], sorted_x[end]]
    end

    range_hdi = sorted_x[n_hdi+1:end] - sorted_x[1:n-n_hdi]
    idx = argmin(range_hdi)
    return [sorted_x[idx], sorted_x[idx+n_hdi]]
end

function write_summary_csv(path::String, summary_data)
    open(path, "w") do io
        println(io, "Dataset,Parameter,Median_or_Mode,CI_Low_or_Frequency,CI_High")
        for (dataset_idx, summary_dict) in enumerate(summary_data)
            for (param, values) in summary_dict
                if haskey(values, "mode")
                    val1 = values["mode"]
                    val2 = values["frequency"]
                    val3 = ""
                else
                    val1 = values["median"]
                    val2 = values["95%CI_low"]
                    val3 = values["95%CI_high"]
                end
                println(io, "$dataset_idx,$param,$val1,$val2,$val3")
            end
        end
    end
end

function write_waic_csv(path::String, waic_data)
    open(path, "w") do io
        println(io, "Dataset,WAIC")
        for (idx, waic_val) in enumerate(waic_data)
            println(io, "$idx,$waic_val")
        end
    end
end

function write_gelman_rubin_csv(path::String, gr_data)
    open(path, "w") do io
        println(io, "Dataset,Parameter,GR_Value")
        for (dataset_idx, gr_dict) in enumerate(gr_data)
            for (param, gr_val) in gr_dict
                println(io, "$dataset_idx,$param,$gr_val")
            end
        end
    end
end

function write_simulation_stats(path::String, data::Vector{Matrix{Float64}})
    header = ["Day", "median", "mean", "hdi_low", "hdi_high"]

    all_datasets_matrix = hcat(data...)

    avg_matrix = mean(all_datasets_matrix, dims=2)

    output_matrix = hcat(1:size(data[1], 1), avg_matrix)
    write_csv(path, header, output_matrix)
end

