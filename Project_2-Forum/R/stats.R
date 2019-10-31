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
year = c('all', '2015', '2016', '2017', '2018', '2019')
writes = '/Users/andrew/Desktop/HKUST/Projects/Firm/LIM/Project_2-Forum/R/stats.csv'
all_stats = c()
for (path_idx in 1:6){
    pnls = read.csv(reads[path_idx])
    number_col = ncol(pnls) - 1
    
    this_stats = c()
    for (curr_col in 2:ncol(pnls)) {
      curr_stats <- summ(cbind(pnls[,curr_col]))
      if (curr_col == 2){this_stats = curr_stats}
      else{this_stats = cbind(this_stats, curr_stats)}
      
    }
    this_stats <- cbind(this_stats, rep(c(year[path_idx]), 12))
    if (path_idx == 1) {
      all_stats <- this_stats
    } else{
      all_stats <- rbind(all_stats, rep(c(' '), number_col), this_stats)
    }
    
  # colnames(all_stats) <- c('cmc3_ret', 'cmc5_ret', 'cmc10_ret', 'cmc15_ret', 'cmc20_ret', 'cmc30_ret')
}
# all_stats <- round(all_stats, 3)
colnames(all_stats) <- c('cmc15_ret', 'year')
write.csv(all_stats, writes)



