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

pnls = read.csv('/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/R/excess_ret/excess_daily_pnls.csv')
reads = c('/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/R/excess_ret/excess_daily_pnls.csv',
          '/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/R/excess_ret/2015.csv',
          '/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/R/excess_ret/2016.csv',
          '/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/R/excess_ret/2017.csv',
          '/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/R/excess_ret/2018.csv',
          '/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/R/excess_ret/2019.csv')
writes = c('/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/R/stats.csv',
           '/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/R/2015stats.csv',
           '/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/R/2016stats.csv',
           '/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/R/2017stats.csv',
           '/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/R/2018stats.csv',
           '/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/R/2019stats.csv')

for (path_idx in 1:6){
  pnls = read.csv(reads[path_idx])
  all_stats = c()
  for (col in c(2,3,4,5,6,7)) {
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
  colnames(all_stats) <- c('cmc3_ret', 'cmc5_ret', 'cmc10_ret', 'cmc15_ret', 'cmc20_ret', 'cmc30_ret')
  write.csv(all_stats, writes[path_idx])
}


