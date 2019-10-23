#Stat summary function
library(PerformanceAnalytics)
summ<- function (y){
  summt <-rbind(apply(y,2,function(x) nrow(y)-1),
                
                apply(y,2,function(x) prod(1+x)-1),
                
                apply(y,2,max),apply(y,2,min),
                
                apply(y,2,function(x) prod(1+x)^(255/nrow(y))-1),
                
                apply(y,2,function(x) sd(x)*sqrt(255)),
                
                apply(y,2,function(x) SemiDeviation(x)*sqrt(255)),
                
                apply(y,2,maxDrawdown),
                
                apply(y,2,function(x) prod(1+x)^(255/nrow(y))-1)/apply(y,2,function(x) sd(x)*sqrt(255)),
                
                apply(y,2,function(x) prod(1+x)^(255/nrow(y))-1)/apply(y,2,function(x) SemiDeviation(x)*sqrt(255)),
                
                apply(y,2,function(x) prod(1+x)^(255/nrow(y))-1)/apply(y,2,maxDrawdown),
                
                apply(y,2,function(x) skewness(x)))
  
  rownames(summt)=c("Periods", "Return", "Daily_Max","Daily_Min","Annual_Return","Volatility","SemiDeviation","MaxDrawdown","Sharpe","Sortino","Calmar","Skewness")
  
  return(summt)
}

pnls = read.csv('/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/data/interim/daily_pnls.csv')
# pnls$X <- as.Date(pnls$X)

for (col in c(2,3,4,5)) {
  # a <- cbind(pnls[,col])
  # rownames(a) <- as.POSIXct(pnls[,1], format ='%Y-%m-%d')
  curr_stats <- summ(cbind(pnls[,col]))
  if (col == 2) {
    all_stats <- curr_stats
  } else{
    all_stats <- cbind(all_stats, curr_stats)
  }
}
all_stats <- round(all_stats, 3)
colnames(all_stats) <- c('cmc_ret', 'cmo_ret', 'omo_ret', 'CSI300')
write.csv(all_stats, '/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/R/stats.csv')



