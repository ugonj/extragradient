using LazySets
using LinearAlgebra

import LazySets: an_element, HalfSpace,σ, LazySet, HPolyhedron, project

mutable struct Iterationcounter
  value::Integer
end

Iterationcounter() = Iterationcounter(0)
function increment(it::Iterationcounter) it.value = it.value+1 end 
function reset(it::Iterationcounter) it.value = 0 end
value(it::Iterationcounter) = it.value

# Base.show(it::Iterationcounter) = Base.show(it.value)
# Base.show(io, it::Iterationcounter) = Base.show(io, it.value)
# Base.show(io, mime, it::Iterationcounter) = Base.show(io, mime, it.value)



## PART 1: Conditional Gradient.

# The approximate alternating projection methods is based on calling
# the conditional gradient method (Frank-Wolfe method) at each step
# to find an approximate projection to a point onto a set, so that 
# the approximate projection is *inside* the set.

"""
The Conditional Gradient method (also called Frank Wolfe method) is an iterative method for solving problems of the form:

minimize ``f(x)`` subject to ``x∈ X``

where the function ``f`` is differentiable, and the set ``X`` is convex and compact.

By default the method uses a finite difference approximation of the gradient, and the default step size at iteration ``i`` is ``\frac{1}{i}``.

To use this method, initialise it as follows:
```
julia > m = ConditionalGradient(f,X,x₀)

julia > m = ConditionalGradient(f,X,x₀)
```

This will create an iterable julia object. You can run through a sequence of iterations of the method.
"""
struct ConditionalGradient{T,F1,F2,F3,F4}
  f  :: F1          # The objective function.
  x₀ :: Vector{T}   # The initial solution.
  LO :: F2          # The Linear Optimisation Oracle used to solve the linear subproblem.
  ∇f :: F3          # The gradient of the function $f$.
  lineSearch :: F4  # Line Search.
  it :: Iterationcounter # Number of line searches
end

"""
Compute an approximation of the derivative via the finite difference method.
"""
finiteDifference(f;ε=1e-10) = x -> (f(x+ε)-f(x-ε))/2ε

# A few constructors for the conditionalGradient algorithm.
ConditionalGradient(f,x,l,df) = ConditionalGradient(f,x,l,df, Iterationcounter())
ConditionalGradient(f,x₀,LO::Function) = ConditionalGradient(f,x₀,LO,finiteDifference(f), (x,s,i) -> 2/i)
ConditionalGradient(f,x₀,X) = ConditionalGradient(f,x₀,linearOptimiser(X),finiteDifference(f), (x,s,i) -> 2/i)

function Base.iterate(m::ConditionalGradient)
  g = -m.∇f(m.x₀)             # First step: get the gradient of $f$.
  s = m.LO(g)              # Second step: solve a LO using the LO oracle.
  return (m.x₀,(m.x₀,g,s,1))
end

function Base.iterate(m::ConditionalGradient,(x,g,s,i)) 
  increment(m.it)
  λ = m.lineSearch(x,s,i)        # Third step: perform a line search.
  x1 = x + λ*(s-x)               # Compute the next iterate.
  g1 = -m.∇f(x1)                 # First step: get the gradient of $f$.
  s1 = m.LO(g1)                  # Second step: solve a LO using the LO oracle.
  return (x1,(x1,g1,s1,i+1))     # Return the next iterate.
end

## Part 2: Approximate Projection onto an ellipsoid.

"""
Generates a linear optimiser oracle for a convex set ``X``. The oracle solves the problem

minimise ``⟨s,x⟩``, subject to ``x∈X``

This oracle will be used as part of the Conditional Gradient for minimising a convex function (such as the distance) in an ellipsoid.
"""
linearOptimiser(C) = x -> σ(x,C)

"""
The Conditional Gradient algorithm for approximately projecting a point onto a set.
"""
struct ApproximateCG{P <: Function, T <: Real, S <: LazySet}
  y :: Vector{T} 
  s :: S
  x :: Vector{T}
  pred :: P
  it :: Iterationcounter
end

function Base.iterate(m::ApproximateCG) 
  LO = linearOptimiser(m.s)
  ls(x) = 0.5*(x-m.y)⋅(x-m.y) # Least Squares Function
  ∇ls(x) = (x-m.y)          # Gradient of the least squares function
  search(x,s,i) = min(1.0,((m.y-x)⋅(s-x)/((s-x)⋅(s-x))))
  cg = ConditionalGradient(ls,m.x,LO,∇ls,search,m.it)
  (x1,s1) = Base.iterate(cg)
  return (x1,(s1,cg))
end

function Base.iterate(m::ApproximateCG,((x,g,s,i),cg))
  #sl = g⋅(s-x)
  #if sl < φ(m.γ,m.θ,m.λ,m.x,m.y,x) return nothing end
  !m.pred(x,g,s,i)  || return nothing
  (x1,s1) = Base.iterate(cg,(x,g,s,i))
  return (x1,(s1,cg))
end

function stopping(ε,x,g,s,i)
  sl = g⋅(s-x)
  return sl < ε
end

"""
Approximate projection using the Conditional Gradient method.
"""
function aproject(s,y;x = an_element(s), pred = (x,g,s,i) -> stopping(1e-6,x,g,s,i), it = Iterationcounter())
  if(y in s) return y end
  acg = ApproximateCG(y,s,x,pred,it)
  yp = Float64[]
  for outer yp in acg end;
  return yp
end


"""
The feasible inexact projection mapping relative to ``u∈C`` with error tolerance `γ` 
is a set defined as follows:

``` math
P_C^γ(u,v) = {x∈ C: ⟨v − w, s − x⟩ ≤ γ∥x − u∥², ∀y∈C}.
```

"""
P(C,u,v,γ,it) = aproject(C,v,; pred = (x,g,s,i) -> stopping(γ*dot(u-x,u-x),x,g,s,i), it=it)

# For projecting onto a hyperplane, we just compute the exact projection, since it is straightforward.
aproject(s::HalfSpace,y;x,pred) = project(s,y)
aproject(s::Hyperplane,y;x,pred) = project(s,y)

project(p::HalfSpace,x) = x∈p ? x : x + (p.b - p.a⋅x) * p.a/norm(p.a)
project(p::Hyperplane,x) = x + (p.b - p.a⋅x) * p.a/norm(p.a)
