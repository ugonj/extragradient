# # Extragradient Method with feasible inexact projection
#
# ## Introduction
#
# We address the variational inequality problem in finite-dimensional Euclidean space.
# This problem is formally stated as follows: let ``F::ℝ^n→ℝ`` be an operator and ``𝒞 ⊂ ℝ^n`` be a nonsmpty and closed convex set. The Variational inequality problem ``\mathrm{VIP}(F,𝒞)`` associated with ``F`` and ``𝒞`` consists in finding ``x^*∈𝒞`` such that
# ```math
# ⟨F(x^*), x − x^*⟩ ≥ 0 ∀x∈𝒞
# ```

# ## Extragradient inexact method with constant step size
"""
The extragradient method is a method for solving Variational Inequality Problems. The aim is to find a point ``x^*`` such that
```math
⟨F(x^*), x − x^*⟩ ≥ 0 ∀x∈𝒞.
```

This structure contains parameters for the VIP itself (`F` and `C`) as well as the parameters for the extragradient method: positive numbers `α` and `γ`.

If `F` is a Lipschitz pseudomonotone operator with Lipschitz constant ``L`` and 
```math
α < \\frac{\\sqrt{1-2γ}}{L}
```
then the method converges.
"""
struct Extragradient
  F
  C
  α
  γ
  it :: Iterationcounter
end

function Base.iterate(m::Extragradient)
  reset(m.it)
  x = an_element(m.C)
  return x,(x,1)
end

function Base.iterate(m::Extragradient,(x,k))
  F = m.F(x)
  αk = 1/(k+1)^2.1 
  γ = min(αk/(F⋅F),m.γ)
  y = P(m.C,x,x-m.α*F,γ,m.it)
  if y ≈ x return nothing end
  x = P(m.C,x,y-m.α*F,γ,m.it)
  return x,(x,k+1)
end

Base.IteratorSize(m::Extragradient) = Base.SizeUnknown()
Base.IteratorSize(m::Base.Iterators.Enumerate{Extragradient}) = Base.SizeUnknown()
Base.IteratorSize(m::Base.Iterators.Take{Extragradient}) = Base.SizeUnknown()

struct InexactExtragradient
  F
  C
  β
  σ
  ρ
  α
  it :: Iterationcounter
end

# Step 1: Let ``x¹∈𝒮``
function Base.iterate(m::InexactExtragradient)
  x = an_element(m.C)
  return x,(x,1)
end

function Base.iterate(m::InexactExtragradient,(x,k))
  F = m.F
  C = m.C
  β = m.β
  σ = m.σ
  ρ = m.ρ
  α = m.α
  # Choose an error tolerance
  γ = 0.9*min(1-ρ,2-√3)
  # Compute the inexact projection:
  y = P(C,x,x-β*F(x),γ,m.it)
  # Step 3:
  if y==x return nothing end
  i = 0
  while ((F(x+σ*α^i*(y-x))⋅(y - x)) > ρ*(F(x)⋅(y-x))) i+=1 end
  z = x + σ*α^i*(y-x)
  λ = -1/(F(z)⋅F(z))*(F(z)⋅(z-x))
  x = P(C,x,x-λ*F(z),γ, m.it)
  return x,(x,k+1)
end

Base.IteratorSize(m::InexactExtragradient) = Base.SizeUnknown()
Base.IteratorSize(m::Base.Iterators.Enumerate{InexactExtragradient}) = Base.SizeUnknown()
Base.IteratorSize(m::Base.Iterators.Take{InexactExtragradient}) = Base.SizeUnknown()

## VIP

"""
A Variational inequality problem.

The aim is to find a point ``x^*`` such that
```math
⟨F(x^*), x − x^*⟩ ≥ 0, ∀x∈𝒞.
```
"""
struct VIP
  C
  F
  L::Float64
end

"""
An Experiment
"""
struct VIPExperiment
  Pb
  m
end

VIP(C,F) = VIP(C,F,Inf64)

Extragradient(pb::VIP,α,γ) = Extragradient(pb.F,pb.C,α,γ, Iterationcounter())
InexactExtragradient(pb::VIP, β, σ, ρ, α) = InexactExtragradient(pb.F, pb.C, β, σ, ρ, α, Iterationcounter())
Extragradient(pb,γ) = Extragradient(pb,0.5*(sqrt(1-g)/pb.L),γ)
