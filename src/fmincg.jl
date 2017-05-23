function fmincg(f, X, options)
# Minimize a continuous differentialble multivariate function. Starting point
# is given by "X" (D by 1), and the function named in the string "f", must
# return a function value and a vector of partial derivatives. The Polack-
# Ribiere flavour of conjugate gradients is used to compute search directions,
# and a line search using quadratic and cubic polynomial approximations and the
# Wolfe-Powell stopping criteria is used together with the slope ratio method
# for guessing initial step sizes. Additionally a bunch of checks are made to
# make sure that exploration is taking place and that extrapolation will not
# be unboundedly large. The "length" gives the length of the run: if it is
# positive, it gives the maximum number of line searches, if negative its
# absolute gives the maximum allowed number of function evaluations. You can
# (optionally) give "length" a second component, which will indicate the
# reduction in function value to be expected in the first line-search (defaults
# to 1.0). The function returns when either its length is up, or if no further
# progress can be made (ie, we are at a minimum, or so close that due to
# numerical problems, we cannot get any closer). If the function terminates
# within a few iterations, it could be an indication that the function value
# and derivatives are not consistent (ie, there may be a bug in the
# implementation of your "f" function). The function returns the found
# solution "X", a vector of function values "fX" indicating the progress made
# and "i" the number of iterations (line searches or function evaluations,
# depending on the sign of "length") used.
#
# Usage: [X, fX, i] = fmincg(f, X, options)
#
# See also: checkgrad
#
# Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
#
#
# (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
#
# Permission is granted for anyone to copy, use, or modify these
# programs and accompanying documents for purposes of research or
# education, provided this copyright notice is retained, and note is
# made of any changes that have been made.
#
# These programs and documents are distributed without any warranty,
# express or implied.  As the programs were written for research
# purposes only, they have not been tested to the degree that would be
# advisable in any important application.  All use of these programs is
# entirely at the user's own risk.
#
# [ml-class] Changes Made:
# 1) Function name and argument specifications
# 2) Output display
#
# [hiroyuki] Changes Made:
# 1) Change syntax according to Julia
# 2) Fix function arguments comments

    # Read options
    if (haskey(options, "MaxIter"))
        length = get(options, "MaxIter", 100)
    else
        length = 100
    end

    RHO = 0.01                             # a bunch of constants for line searches
    SIG = 0.5        # RHO and SIG are the constants in the Wolfe-Powell conditions
    INT = 0.1     # don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0                     # extrapolate maximum 3 times the current bracket
    MAX = 20                          # max 20 function evaluations per line search
    RATIO = 100                                       # maximum allowed slope ratio

    # octave could reduce code with using "max(Tuple...)"
    if (max( size(length, 1), size(length, 2) ) == 2)
        red    = length(2)
        length = length(1)
    else
        red    = 1
    end
    S="Iteration "                           # only to display progress

    i = 0                                             # zero the run length counter
    ls_failed = 0                              # no previous line search has failed
    fX = []
    f1,df1 = eval(f)(X)                           # get function value and gradient
    i = i + (length<0)                                             # count epochs?!
    s = -df1                                         # search direction is steepest

    d1 = (-s'*s)                                                # this is the slope

    #@printf("d1[1] %s = %f \n", typeof(d1[1]), d1[1])
    #@printf("d1[2] %s = %d \n", typeof(d1[2]), d1[2])
    #@printf("red = %d, d1 = %f\n", red, d1[1])

    z1 = red / (1.0 - d1[1])                          # initial step is red/(|s|+1)
    #@printf("z1 = %f \n", z1)

    while i < abs(length)
        i = i + (length > 0)                                   # count iterations?!

        X0, f0, df0 = X, f1, df1                    # make a copy of current values
        X = X + z1*s                                            # begin line search
        f2, df2 = eval(f)(X)
        df2 = df2[1]
        i = i + (length<0)                                         # count epochs?!
        d2 = df2' * s
        f3, d3, z3 = f1, d1, -z1              # initialize point 3 equal to point 1
        if length > 0
            M = MAX
        else
            M = min(MAX, -length-i)
        end
        success, limit = 0, -1                              # initialize quanteties
    end

    return X, fX, i
end
