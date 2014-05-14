library(multicore)

load("psave.Rdata")
load("fmx_v10.Rdata")

fmx <- as.matrix(cbind(p1, p2, p3, p4))
truth <- as.logical(VP1$return)

bootstrap.err.rate <- function(w, fm, y, N=1000) {
    # convert scale-free parameters to mixing proportions
    eta <- c(exp(w) / (1 + sum(exp(w))), 1 / (1 + sum(exp(w))))

    # bootstrap error rate
    err.rate <- rep(NA, N)
    for (i in 1:N) {
        idx <- sample(1:length(y), length(y), replace=T)
        yi <- y[idx]
        fmi <- fm[idx, ]

        err.rate[i] <- sum(yi != (fmi %*% eta >= 0.5)) / length(y)
    }

    return (mean(err.rate))
}

set.seed(2014)

N.init <- 64

# w.init <- c(2, 1.5, 1)
# Nelder-Mead seems to be searching around this "prior" value, 
# SANN definitely searches for a much larger space. 

# So to use NM, we rely on a good prior to start with?
# And we will be using parcheck.R to validate its relative stability in 
# hypercube?
init.w <- cbind(rnorm(N.init, 2, 0.75), 
                rnorm(N.init, 1.5, 0.75), 
                rnorm(N.init, 1, 0.75)) + 0.25

bser.wrapper <- function(i, W) {
    iw <- W[i, ] # initial weights

    w.opt <- optim(iw, bootstrap.err.rate, method="Nelder-Mead",
                   fm=fmx, y=truth, 
                   control=list(trace=1))

    #w.opt <- optim(iw, bootstrap.err.rate, method="SANN",
    #               fm=fmx, y=truth, 
    #               control=list(trace=2, maxit=100, REPORT=10))
    return (w.opt)
}

all.optim <- mclapply(1:N.init, bser.wrapper, W=init.w, mc.cores=16)
save(all.optim, file="NM_optim2.Rdata")
