using Convex, SCS, BenchmarkTools, Suppressor

function projOntoProb(v,b)
    mu = sort(v, rev=true)
    s = [(1/i) * (sum(mu[1:i]) - b) for i in 1:length(v)]
    rho = argmax( [i * (mu[i] > s[i]) for i in 1:length(v)] )
    w = [max(v[i] - s[rho], 0) for i in 1:length(v)]
    return w
end

function newProj(y,s)
    x = projOntoProb(y,s)
    k = 0
    while sum(x .> 1) > 0 && k < 100
        x[x .> 1] .= 1
        x = projOntoProb(x,s)
        k = k + 1
    end
    return x
end
    
function newFastProj(y,s)
    D = length(y)
    y_sorted = sort(y)
    T = [0.0]
    for i = 2:D+1
        push!(T,T[i-1] + y_sorted[i-1])
    end
    pushfirst!(y_sorted,-Inf)
    push!(y_sorted,Inf)
    
    a = 0
    b = 0
    terminate = false
    for ai = 0:D
        a = ai
        if (s == D - a) && (y_sorted[a + 1 + 1] - y_sorted[a + 1] >= 1)
            b = a
            break
        end
        for bi = a+1:D
            b = bi
            gamma = (s + b - D + T[a + 1] - T[b + 1]) / (b - a)
            if (y_sorted[a+1] + gamma <= 0) && (y_sorted[a + 1 + 1] + gamma > 0) && (y_sorted[b + 1] + gamma < 1) && (y_sorted[b + 1 + 1] + gamma >= 1)
                terminate = true
                break
            end
        end
        if terminate
            break
        end
    end
    gamma = (s + b - D + T[a + 1] - T[b + 1]) / (b - a)
    if a == D+1
        x = [0 for i=1:D]
    end
    if b == D+1
        x = [1 for i=1:D]
    end
    if b > a
        x = [min(max(y[i]+gamma,0),1) for i = 1:D]
    else
        ys = sort(y)
        ai = 1
        while ai <= D-1
            if ys[ai+1] - ys[ai] >= 1
                break
            end
            ai += 1
        end
        thresh = ys[ai]
        x = (y .> thresh) .* 1
    end
    return x
end

function ogProj(y,s)
    w_p = Variable(length(y))
    problem = minimize(norm(w_p - y), [w_p >= 0, w_p <= 1, sum(w_p) == s]) # problem definition (Convex.jl), total power constraint)
    solve!(problem, SCS.Optimizer; silent=true)
    w_p = evaluate(w_p)
    return w_p
end

function isApproxEq(x,y,eps)
    res = true
    if length(x) != length(y)
        println("Mismatched dims")
        return false
    end
    for i = 1:length(x)
        if abs(x[i] - y[i]) > eps
            res = false
            break
        end
    end
    return res
end
    

function compareFuncs(iters, input_size)
    equal_rate = 0
    avg_time_fast_proj = 0
    avg_time_og_proj = 0
    for _ in 1:iters
        s = 5
        y = rand(-.5:.0001:1.5, input_size)

        # Time each function
        fast_proj_stats = @suppress @timed newProj(y,s)
        og_proj_stats = @suppress @timed ogProj(y,s)

        fast_proj_time = fast_proj_stats.time
        og_proj_time = og_proj_stats.time
        fast_proj_res = fast_proj_stats.value
        og_proj_res = og_proj_stats.value

        equal_outputs = isApproxEq(fast_proj_res, og_proj_res, 1e-2)

        equal_rate += equal_outputs
        avg_time_fast_proj += fast_proj_time
        avg_time_og_proj += og_proj_time
    end

    return equal_rate/iters, avg_time_fast_proj/iters, avg_time_og_proj/iters 
end