## Feature matrix generation in R with dplyr
## 

library(dplyr)

load("~/Dropbox/DMC/Cory/dataclean/data_v1.Rdata")
zoe.size.type <- read.csv("~/projects/dmc14/data/tr_sizetype.csv", header=T)
names(zoe.size.type) <- c('zsize', 'ztype')
vs <- read.csv("~/Dropbox/DMC/Cory/validset/batchValidSet_v2.csv",
               header=T)

raw.tr <- cbind(tr, zoe.size.type, vs)
# remove all rows with deldate == NA
raw.tr <- raw.tr[!is.na(raw.tr$deldate), ]
N <- nrow(raw.tr)

# batch id
cid.changes <- c(1, raw.tr$cid[2:N] - raw.tr$cid[1:(N-1)])
cid.changes[cid.changes != 0] <- 1
raw.tr$batch <- cumsum(cid.changes)

# avoid leakage
validset <- as.logical(raw.tr$valid)
raw.tr$return[validset] <- NA

# split into training and validation
trt <- raw.tr[validset, ]
trv <- raw.tr[!validset, ]

batches <- group_by(raw.tr, 'batch')

## To compute counts and LLRs for given "feats", the combation of features.
counts.and.llrs <- function(df, feats, c1=1.0, c2=1.0) {
    # use do.call to expand combination of features into arguments
    grp <- do.call(group_by, c(list(df), list(feats)))

    # overall counts and returns, sans validation set
    N <- grp %.% mutate(counts=sum(!is.na(return))) 
    R <- grp %.% mutate(returns=sum(return, na.rm=T))

    # per batch counts (which will be used to compute correction factor)
    bat.grp <- do.call(group_by, c(list(df), list(c('batch', feats))))
    k <- bat.grp %.% mutate(counts=sum(!is.na(return)))

    llr <- log((R$returns + c1) / (N$counts - R$returns + c2))
    adj.llr <- (counts - k) / counts * llr

    return (cbind(N$counts, adj.llr))
}


## batch features
bfeats <- batches %.% mutate(bat.n=length(oid), 
                   bat.uniq.iid=length(unique(iid)),
                   bat.uniq.mid=length(unique(mid)),
                   bat.uniq.size=length(unique(size)),
                   bat.uniq.color=length(unique(color)),
                   bat.uniq.ztype=length(unique(ztype)),
                   bat.uniq.zsize=length(unique(zsize)),
                   bat.prank=rank(price))

## within batch features

## customer per batch features
cbatches <- group_by(raw.tr, cid, batch)
cb.ret.rates <- cbatches %.% mutate(rrate=sum(return)/length(return), 
                                    krate=1-sum(return)/length(return)) %.% 
                select(cid, batch, rrate, krate)

# only set the first order of each batch to be the true rate, others set to be
# NA
cb.ret.srates <- cbatches %.% 
                 mutate(srrate=c(sum(return)/length(return), 
                                 rep(NA, length(return)-1)),
                        skrate=c(1-sum(return)/length(return), 
                                 rep(NA, length(return)-1))) %.%
                 select(cid, batch, srrate, skrate)

cb.ret.rates$srrate <- cb.ret.srates$srrate
cb.ret.rates$skrate <- cb.ret.srates$skrate

# average return/keep rate, weighted and unweighted,
cb.avg.feats <- group_by(cb.ret.rates, cid) %.%
                murate(cid.wavg.rrate=mean(rrate, na.rm=T),
                       cid.wavg.krate=mean(krate, na.rm=T),
                       # simple averages
                       cid.avg.rrate=mean(srrate, na.rm=T),
                       cid.avg.krate=mean(skrate, na.rm=T),
                       cid.sum.rrate=sum(srrate, na.rm=T),
                       cid.sum.krate=sum(skrate, na.rm=T)) %.%
                select(cid, starts_with('cid.'))

# log-likelihood ratio of return over kept
cb.avg.feats$cid.llr.rk <- log((cb.avg.feats$cid.avg.rrate+1) /
                               (cb.avg.feats$cid.avg.krate+1))


