## Data Mining Cup 2014
## Feature matrix generation (v4) in R with dplyr

## Authors:
##    Xin Yin <xinyin at iastate dot edu>
##    Fan Cao <fancao at iastate dot edu>

## NOTE:
## This script has some paths to the datafiles hardcoded,
## make sure all the required files are in place before running
## the script.

## USAGE:
## R CMD BATCH --no-restore --no-save "--args 1" feat_gen.R

rm(list = ls())
library(reshape2)
library(dplyr)

args <- commandArgs(trailingOnly = T)
if (length(args) == 0) {
    stop("You need to specify the set index.")
}

sidx <- as.numeric(args[1])
set.name <- paste("ftr_v4.1_set", sidx, ".Rdata", sep="")

if (!(sidx %in% c(1, 2, 3, 4, 5, 6))) {
    stop("Invalid set index.")
}

## Set indices
if (sidx == 4) {
    load("data/set4_v4.Rdata")
}
else {
    load("data/sets_v4.Rdata")
}

# Import within batch features and batch interval features
# generated previously with feature_extract.py (pandas)

#fv32 <- read.csv("data/intvl_v4.csv", header=T)
#bint.idx <- c(grep("bint.", names(fv32)))
#bint.feats <- fv32[, bint.idx]
#rm(fv32)

####### Import Common data

load("data/data_v2.Rdata")
#tr <- read.csv("data/dataclean_v2_train.csv", header=T)

# validation indicators
vs <- NULL

if (sidx <= 4) {
    vs <- data.frame(valid=part_va[, paste("S", sidx, sep="")])
}

if (sidx >= 5) {
    vs <- data.frame(valid=part_va$S1)
}

raw.tr <- cbind(tr, vs)

# If to generate subset...
if (sidx >= 5) {
    # combine training and validation
    sset.idx <- as.logical(part_red == sidx) | vs$valid
    raw.tr <- raw.tr[sset.idx, ]

    # subset interval features as well
    #bint.idx <- as.logical(part0_red == sidx) | part0_va$S1
    #bint.feats <- bint.feats[bint.idx, ]
}

# remove all rows with deldate == NA
raw.tr <- raw.tr[!is.na(raw.tr$deldate), ]
N <- nrow(raw.tr)

## group orders into batches and compute batch id
cid.changes <- c(1, raw.tr$cid[2:N] - raw.tr$cid[1:(N-1)])
cid.changes[cid.changes != 0] <- 1
raw.tr$batch <- cumsum(cid.changes)

# avoid leakage
validset <- as.logical(raw.tr$valid)
raw.tr$return[validset] <- NA

#### End of data preprocessing


######################################
####       Fan's features         ####
######################################

# alias
train <- raw.tr

#disc 
#percentage the price compare to the full price of an item
disc = (train %.% group_by(iid) %.% mutate(disc = price/max(price))) $ disc
disc[is.na(disc)] = 0

#discr 
#the rank of disc in each batch
discr = (cbind(train,disc) %.% tbl_df() %.% group_by(batch) 
         %.% mutate(discr = rank(disc))) $ discr

train = cbind(train , disc , discr) %.% as.data.frame() %.% tbl_df()
#ggplot(filter(train , iid == 1) , aes(x = date , y = disc)) + geom_point(aes(color = season))

#pricer
#the rank of price in each batch
train = train %.% group_by(batch) %.% mutate(pricer = rank(price)) %.%
arrange(oid) %.% as.data.frame() %.% tbl_df()

#the max,min,mean price/disc by each item 
#mindisc.by.iid
#meandisc.by.iid
#maxprice.by.iid
#minprice.by.iid
#meanprice.by.iid
train = train %.% group_by(iid) %.% 
mutate(mindisc.by.iid = min(disc , na.rm = T) ,  
       meandisc.by.iid = mean(disc  , na.rm = T) , minprice.by.iid = min(price) , 
       maxprice.by.iid = max(price) , meanprice.by.iid = mean(price)) %.%
arrange(oid) %.% as.data.frame() %.% tbl_df()

#considering the local(the nearest 15 order) disc/price trend for items 
#smooth with neighbor to be 15
library(FNN)
keverage = function(x , y) {
    if(length(x) < 16) {
        return(1)
    } else {
        res = knn.reg(train = x , y = y , k = 15)$pred
        return(res)
    }
}

#localdisc
#localprice
#pricediff
#discdiff
#outday.by.iid
#deal
train = train %.% group_by(iid) %.%
mutate(localdisc = keverage(date , disc) , 
       localprice = keverage(date , price)) %.%
mutate(pricediff = price - localprice , 
       discdiff = disc - localdisc , 
       outday.by.iid = min(date)) %.%
mutate(deal = pricediff < 0) %.%
arrange(oid) %.% as.data.frame() %.% tbl_df()

#nlowprice.by.cid
#nlowdisc.by.cid
#ndeal.by.cid
#norder.by.cid
#nreturn.by.cid
#totalspend.by.cid
#meanspend.by.cid
#nonreturnspend.by.cid
#returnspend.by.cid

train = train %.% group_by(cid) %.%
mutate(nlowprice.by.cid = sum(price < 100) , nlowdisc.by.cid = sum(disc < 0.8) , 
       ndeal.by.cid = sum(deal) , norder.by.cid = n() ,
       nreturn.by.cid = sum(return , na.rm = T) , 
       totalspend.by.cid = sum(price) , meanspend.by.cid = mean(price) ,
       nonreturnspend.by.cid = sum(price * (return==0) , na.rm = T) , 
       returnspend.by.cid = sum(price * (return == 1) , na.rm = T)) %.% 
arrange(oid) %.% as.data.frame() %.% tbl_df()


#some by.batch.cid feature(xin have done this)
#train = train %.% group_by(date , cid) %.% mutate(mbspend = mean(price) , bsize = n()) %.%
#as.data.frame() %.% tbl_df()

#train = train %.% group_by(cid) %.% mutate(nbc = length(unique(date)) , noc = n() , bspendc = sum(price)) 

#outseason.by.iid
outseason.by.iid = rep(5 , nrow(train))
outseason.by.iid[train$outday < 61] = 1
outseason.by.iid[train$outday > 60 & train$outday < 153] = 2
outseason.by.iid[train$outday > 152 & train$outdat < 245] = 3
outseason.by.iid[train$outday > 244 & train$outday < 237] = 4
train = as.data.frame(cbind(train , outseason.by.iid)) %.% tbl_df()

##############################################################

check_return_before = function(dord , ret) {
    n = length(dord)
    if (n == 1) return(0)
    else {
        res = rep(NA , n)
        for (i in 1:n) {
            pre = ret[dord < (dord[i])]
            if (length(pre)==0) res[i] = 0
            else if (sum(is.na(pre)) > 0) res[i] = 0
            else { 
                preret = sum(pre==1 , na.rm = T)
                res[i] = preret
            }
        }
    }
    return(res)
}


check_keep_before = function(dord , ret) {
    n = length(dord)
    if (n == 1) return(0)
    else {
        res = rep(NA , n)
        for (i in 1:n) {
            pre = ret[dord < (dord[i])]
            if (length(pre)==0) res[i] = 0
            else if (sum(is.na(pre)) > 0) res[i] = 0
            else {
                prekeep = sum(pre==0 , na.rm = T)
                res[i] = prekeep
            }   
        }
    }
    return(res)
}

check_order_before = function(dord , ret) {
  n = length(dord)
  if (n == 1) return(0)
  else {
    res = rep(NA , n)
    for (i in 1:n) { 
      pre = ret[dord < (dord[i])]
      res[i] = sum(is.na(pre)) + sum(!is.na(pre))
    }
  }
  return(res)
}


check_keep_future = function(dord , ret) {
    n = length(dord)
    if (n == 1) return(0)
    else {
        res = rep(NA , n)
        for (i in 1:n) {
            fu = ret[dord > (dord[i])]
            if (length(fu)==0) res[i] = 0
            else if (sum(is.na(fu)) > 0) res[i] = 0
            else {
                fukeep = sum(fu == 0 , na.rm = T)
                res[i] = fukeep
            }   
        }
    }
    return(res)
}

check_return_future = function(dord , ret) {
    n = length(dord)
    if (n == 1) return(0)
    else {
        res = rep(NA , n)
        for (i in 1:n) {
            fu = ret[dord > (dord[i])]
            if (length(fu)==0) res[i] = 0
            else if (sum(is.na(fu)) > 0) res[i] = 0
            else { 
                furet = sum(fu == 1 , rm.na = T)
                res[i] = furet
            }
        }
    }
    return(res)
}

check_order_future = function(dord , ret) {
  n = length(dord)
  if (n == 1) return(0)
  else {
    res = rep(NA , n)
    for (i in 1:n) {
      fu = ret[dord > (dord[i])]
      res[i] = length(fu)
    }
  }
  return(res)
}

#rb.by.cid.iid.price
#kb.by.cid.iid.price
#ob.by.cid.iid.price
#rf.by.cid.iid.price
#kf.by.cid.iid.price
#of.by.cid.iid.price
#If a cid returned/kept/ordered an item of the same price before/in the future
train = train %.% group_by(cid , iid , price) %.%
mutate(rb.by.cid.iid.price = check_return_before(date,return),
       kb.by.cid.iid.price = check_keep_before(date,return),
       rf.by.cid.iid.price = check_return_future(date,return),
       kf.by.cid.iid.price = check_keep_future(date,return)) %.%
arrange(oid) %.% as.data.frame() %.% tbl_df()

#rb.by.cid.iid.color.size
#kb.by.cid.iid.color.size
#ob.by.cid.iid.color.size
#rf.by.cid.iid.color.size
#kf.by.cid.iid.color.size
#of.by.cid.iid.color.size
#If a cid returned/kept/ordered an exactly same item before/in the future
train = train %.% group_by(cid , iid , color , size) %.% 
mutate(rb.by.cid.iid.color.size = check_return_before(date,return),
       kb.by.cid.iid.color.size = check_keep_before(date,return),
       rf.by.cid.iid.color.size = check_return_future(date,return),
       kf.by.cid.iid.color.size = check_keep_future(date,return)) %.%
arrange(oid) %.% as.data.frame() %.% tbl_df()

#rb.by.cid.iid
#ob.by.cid.iid
#kb.by.cid.iid
#rf.by.cid.iid
#of.by.cid.iid
#kf.by.cid.iid
#If a cid returned/kept/ordered a same iid before/in the future
train = train %.% group_by(cid , iid) %.%
mutate(rb.by.cid.iid = check_return_before(date,return),
       ob.by.cid.iid = check_order_before(date,return),
       kb.by.cid.iid = check_keep_before(date,return),
       rf.by.cid.iid = check_return_future(date,return),
       of.by.cid.iid = check_order_future(date,return),
       kf.by.cid.iid = check_keep_future(date,return)) %.%
arrange(oid) %.% as.data.frame() %.% tbl_df()

#rb.by.cid.price
#ob.by.cid.price
#kb.by.cid.price
#rf.by.cid.price
#of.by.cid.price
#kf.by.cid.price
#If a cid returned/kept/ordered a same item with same price before/in the future
train = train %.% group_by(cid , price) %.%
mutate(rb.by.cid.price = check_return_before(date,return),
       kb.by.cid.price = check_keep_before(date,return),
       rf.by.cid.price = check_return_future(date,return),
       kf.by.cid.price = check_keep_future(date,return)) %.%
arrange(oid) %.% as.data.frame() %.% tbl_df()

#the price rank of price of the iid ordered by a cid
#rankprice.by.cid.iid 
train = train %.% group_by(cid , iid) %.%
  mutate(rankprice.by.cid.iid = rank(price)) %.%
  arrange(oid) %.% as.data.frame() %.% tbl_df()


ntotal = nrow(train)
#llr.by.price
##return+1/#keep+1
train = train %.% group_by(price) %.% 
mutate(llr.by.price = log((sum(return , na.rm = T)+1) / (1+length(return)-sum(return,na.rm=T)))) %.%
arrange(oid) %.% as.data.frame() %.% tbl_df()
dummy = train
k = (dummy %.% group_by(batch , price) %.% mutate(k = sum(return , rm.na = T)))$k
n = (dummy %.% group_by(price) %.% mutate(n = n()))$n
train$llr.by.price = train$llr.by.price * (n - k)/n
rm(dummy , k , n)

#llr.by.cid.price
#return rate by each price each cid
train = train %.% group_by(cid , price) %.% 
mutate(llr.by.cid.price = log((sum(return , na.rm = T)+1) / (1+length(return)-sum(return,na.rm=T)))) %.%
arrange(oid) %.% as.data.frame() %.% tbl_df()
dummy = train
k = (dummy %.% group_by(batch , cid , price) %.% mutate(k = sum(return , rm.na = T)))$k
n = (dummy %.% group_by(cid , price) %.% mutate(n = n()))$n
train$llr.by.cid.price = train$llr.by.cid.price * (n - k)/n
rm(dummy , k , n)
#remove old features
fan.feats = train[, 31:72]

rm(train)

######################################
###    End of Fan's features      ####
######################################

# split into training and validation
#trt <- raw.tr[validset, ]
#trv <- raw.tr[!validset, ]

batches <- group_by(raw.tr, 'batch')

## item freshness 
raw.tr$f1w <- as.factor(fan.feats$outday.by.iid <= 7)
raw.tr$f2w <- as.factor(fan.feats$outday.by.iid <= 14)
raw.tr$f1m <- as.factor(fan.feats$outday.by.iid <= 30)
raw.tr$f3m <- as.factor(fan.feats$outday.by.iid <= 90)
raw.tr$f6m <- as.factor(fan.feats$outday.by.iid <= 180)
raw.tr$oseas <- fan.feats$outseason.by.iid

raw.tr$isdisc <- as.factor(fan.feats$disc < 1)
raw.tr$deal <- as.factor(fan.feats$deal)
raw.tr$lowdisc <- as.factor(fan.feats$disc <= 0.8)

## price ranges
raw.tr$pb25 <- as.factor(raw.tr$price < 25)
raw.tr$pb50 <- as.factor(raw.tr$price < 50)
raw.tr$pb100 <- as.factor(raw.tr$price < 100)
raw.tr$pb200 <- as.factor(raw.tr$price < 200)

pr <- raw.tr$price
pr[pr == 0] <- 1 
raw.tr$pct.logpr <- as.factor(round((log(pr) - log(min(pr))) / 
                   (log(max(pr)) - log(min(pr))) * 20))

## To compute counts and LLRs for given "feats", the combation of features.
counts.and.llrs <- function(df, feats, c1=1.0, c2=1.0) {
    # use do.call to expand combination of features into arguments
    grp <- do.call(group_by, c(list(df), as.list(feats)))

    # overall counts and returns, sans validation set
    N <- (grp %.% mutate(counts=sum(!is.na(return))))$counts
    R <- (grp %.% mutate(returns=sum(return, na.rm=T)))$returns

    # per batch counts (which will be used to compute correction factor)
    bat.grp <- do.call(group_by, c(list(df), as.list(c('batch', feats))))
    k <- (bat.grp %.% mutate(counts=sum(!is.na(return))))$counts

    llr <- log((R + c1) / (N - R + c2))
    #adj.llr <- (N - k) / N * llr

    return (as.data.frame(cbind(N, llr)))
    #return (as.data.frame(cbind(N, adj.llr)))
}

## historical features
# all.cols <- c("cid", "iid", "mid", "ztype", "zsize", "size", "color", 
#                "state", "month", "season", "dow", "f1w", "f2w", "f1m",
#                "f3m", "f6m", "oseas", "isdisc", "deal", "lowdisc",
#                "pb25", "pb50", "pb100", "pb200", "pct.logpr")

all.cols <- c("cid", "iid", "mid", "ztype", "zsize", "size", "color", 
               "state", "month", "season", "dow", "prend") 

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
    c1 <- cols[1]
    c2 <- cols[2]

    # skip useless 2-way interactions
    if (all(which(all.cols == c1) > 11) && 
        all(which(all.cols == c2) > 11)) next

    cat(" :: ", cols, fill=T)

    fnames <- paste(c("all.cnt.", "all.llr."), 
                    paste(cols, collapse="_"), sep="") 

    .feats = counts.and.llrs(raw.tr, cols)
    names(.feats) <- fnames
    all.feats <- cbind(all.feats, .feats)
}

# ratio of low price / low discount
fan.feats$rlowprice.by.cid <- fan.feats$nlowprice.by.cid / 
        all.feats$all.cnt.cid
fan.feats$rlowprice.by.cid[all.feats$all.cnt.cid == 0] <- 0
fan.feats$rlowdisc.by.cid <- fan.feats$nlowdisc.by.cid /
        all.feats$all.cnt.cid
fan.feats$rlowdisc.by.cid[all.feats$all.cnt.cid == 0] <- 0

## batch features
bfeats <- batches %.% mutate(bat.n=length(oid), 
                   bat.uniq.iid=length(unique(iid)),
                   bat.uniq.mid=length(unique(mid)),
                   bat.uniq.size=length(unique(size)),
                   bat.uniq.color=length(unique(color)),
                   bat.uniq.ztype=length(unique(ztype)),
                   bat.uniq.zsize=length(unique(zsize)),
                   bat.prank=rank(price)) %.%
            select(batch, starts_with('bat.'))

## within batch features
within.features <- list(size=c("iid", "mid", "color", "price"),
                        color=c("iid", "mid", "size", "price"),
                        iid=c("size", "color", "zsize", "price"),
                        mid=c("size", "color", "zsize", "price"))

bwi.feats <- NULL

i <- 1
for (fs in within.features) {
    wi.f <- names(within.features)[i]
    for (f in fs) {
        cat(" :: ", wi.f, f, fill=T)
        fname <- paste('bwi_', wi.f, '.uniq.', f, sep="")

        grp <- do.call(group_by, c(list(raw.tr), 
                                  as.list(c('batch', wi.f))))
        ucnts <- eval(substitute(
            (grp %.% mutate(counts=length(unique(f))))$count,
            list(f=as.name(f))))

        if (is.null(bwi.feats)) {
            bwi.feats <- data.frame(ucnts)
            names(bwi.feats) <- fname
        }
        else {
            .feats <- data.frame(ucnts)
            names(.feats) <- fname
            bwi.feats <- cbind(bwi.feats, .feats)
        }
    }
    i <- i + 1
}

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
                mutate(cbat.wavg.rrate=mean(rrate, na.rm=T),
                       cbat.wavg.krate=mean(krate, na.rm=T),
                       # simple averages
                       cbat.avg.rrate=mean(srrate, na.rm=T),
                       cbat.avg.krate=mean(skrate, na.rm=T),
                       cbat.sum.rrate=sum(srrate, na.rm=T),
                       cbat.sum.krate=sum(skrate, na.rm=T)) %.%
                select(cid, batch, starts_with('cbat.'))

# log-likelihood ratio of return over kept
cb.avg.feats$cbat.llr.rk <- log((cb.avg.feats$cbat.avg.rrate+1) /
                               (cb.avg.feats$cbat.avg.krate+1))

names(cb.avg.feats)

# remove color and oseas
raw.tr <- raw.tr[, -which(names(raw.tr) %in% c('oseas', 'deal'))]


ftr <- cbind(raw.tr, all.feats, bfeats[, -1], cb.avg.feats[, -c(1, 2)],
                  fan.feats, bwi.feats)
#ftr <- cbind(raw.tr, all.feats, bfeats[, -1], cb.avg.feats[, -2],
#                  fan.feats, bwi.feats, bint.feats)


# output
save(ftr, file=paste("data/", set.name, sep=""))
#write.csv(feat.mat, file="featmatrix_v5_part1.csv", row.names=F)
