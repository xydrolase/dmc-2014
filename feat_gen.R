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
    N <- (grp %.% mutate(counts=sum(!is.na(return))))$counts
    R <- (grp %.% mutate(returns=sum(return, na.rm=T)))$returns

    # per batch counts (which will be used to compute correction factor)
    bat.grp <- do.call(group_by, c(list(df), list(c('batch', feats))))
    k <- (bat.grp %.% mutate(counts=sum(!is.na(return))))$counts

    llr <- log((R + c1) / (N - R + c2))
    adj.llr <- (N - k) / N * llr

    return (as.data.frame(cbind(N, adj.llr)))
}

## historical features
all.cols <- c("cid", "iid", "mid", "ztype", "zsize", "size", "color", 
               "state", "month", "season", "dow")
feats.2way <- combn(all.cols, 2)

all.feats <- NULL

for (cidx in all.cols) {
    fnames <- paste(c("all.cnt.", "all.llr."), cidx, sep="") 
    if (is.null(all.feats)) {
        all.feats <- counts.and.llrs(raw.tr, cidx)
        names(all.feats) <- fnames
    } else {
        .feats = counts.and.llrs(raw.tr, cidx)
        names(.feats) <- fnames
        all.feats <- cbind(all.feats, .feats)
    }
}

for (i in 1:ncol(feats.2way)) {
    cols <- feats.2way[, i]
    cat(" :: ", cols, fill=T)

    fnames <- paste(c("all.cnt.", "all.llr."), 
                    paste(cols, collapse="_"), sep="") 

    .feats = counts.and.llrs(raw.tr, cols)
    names(.feats) <- fnames
    all.feats <- cbind(all.feats, .feats)
}

names(all.feats)

## batch features
bfeats <- batches %.% mutate(bat.n=length(oid), 
                   bat.uniq.iid=length(unique(iid)),
                   bat.uniq.mid=length(unique(mid)),
                   bat.uniq.size=length(unique(size)),
                   bat.uniq.color=length(unique(color)),
                   bat.uniq.ztype=length(unique(ztype)),
                   bat.uniq.zsize=length(unique(zsize)),
                   bat.prank=rank(price))

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
                mutate(cid.wavg.rrate=mean(rrate, na.rm=T),
                       cid.wavg.krate=mean(krate, na.rm=T),
                       # simple averages
                       cid.avg.rrate=mean(srrate, na.rm=T),
                       cid.avg.krate=mean(skrate, na.rm=T),
                       cid.sum.rrate=sum(srrate, na.rm=T),
                       cid.sum.krate=sum(skrate, na.rm=T)) %.%
                select(cid, batch, starts_with('cid.'))

# log-likelihood ratio of return over kept
cb.avg.feats$cid.llr.rk <- log((cb.avg.feats$cid.avg.rrate+1) /
                               (cb.avg.feats$cid.avg.krate+1))

feat.mat <- cbind(raw.tr, all.feats, bfeats[, -1], cb.avg.feats[, -2])

# output
write.csv(feat.mat, file="featmatrix_v4_part1.csv", row.names=F)
