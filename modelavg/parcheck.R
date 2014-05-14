load("NM_optim2.Rdata")
load("psave.Rdata")
load("fmx_v10.Rdata")

w2eta <- function(w) {
    w <- c(exp(w), 1)
    return (w / sum(w)) 
}

w <- (all.optim[[19]]$par)
eta <- w2eta(w)

X <- cbind(p1, p2, p3, p4)

noise <- replicate(3, rnorm(1000, 0, 0.15))
w.cube <- w + t(noise)

err.rates <- rep(NA, 1000)
for (i in 1:1000) {
    reta <- X %*% w2eta(w.cube[, i])
    tbl <- table(VP1$return, reta >= 0.5)
    err.rates[i] <- 1 - sum(diag(tbl)) / length(VP1$return) 
}

mean(err.rates)
sd(err.rates)

erod <- order(err.rates, decreasing=T)
err.rates[erod][1:50]

eta
t(apply(w.cube[, erod[1:50]], 2, w2eta))

#apply(eta.cube, 1, mean)
#apply(eta.cube, 1, sd)
