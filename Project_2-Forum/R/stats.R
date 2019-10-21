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

pnls = read.csv('/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/data/interim/total_pnls.csv')
# pnls$X <- as.Date(pnls$X)

for (col in range(c(3, 4))) {
  a <- cbind.data.frame(pnls[,col])
  rownames(a) <- as.POSIXct(pnls[,1], format ='%Y-%m-%d')
  curr_stats <- summ(a)
  print(curr_stats)
}



