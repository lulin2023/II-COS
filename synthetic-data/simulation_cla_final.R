
#import packages-----

library(MASS)
library(ks)
library(kernlab)
library(randomForest)
library(foreach)
library(reshape2)
library(ggplot2)
library(glmnet)
library(caret)
library(nnet)
library(ggpubr)
library(tidyverse)
library(kedd)
library(pbmcapply)
library(onlineFDR)
library(parallel)
library(magrittr)
library(latex2exp)
library(ggsci)
library(lsa)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

source("functions_OnSel.R")
source("algoclass_OnSel.R")  ### implementation for different algorithms,such as SVM, RF

execution_time <- system.time({
N <- 5000 # number of total time points

# simulation setting-----
alpha <- 0.1 # significance level
pi <- 0.2 # Bernoulli(pi)
n <- 5000 # number of historical data
m <- 100 # number of selections when stop
n_train<- round(n/2) # number of data used for training model
#n_cal<- n-n_train #number of data used for estimating locfdr
n_cal <- 4000

Diversity_constant<-1 #determine the diversity threshold
Diversity_initial_num<-1 #when rejection number exceeds this, we consider computing diversity.
algo<- new("RFc") #algorithm used for classification or regression
lambda<- 500 #specific parameter for the algorithm

# confirm H0, 6 choices among classification and regression settings-----

### H0:Y=0  H1:Y=1 randomforest classifier or other algorithms except for SVM
Value=list(type="==A,R",v=0)
### H0:Y=1  H1:Y=-1 SVM classifier
#Value=list(type="==A,S",v=1)
### H0??Y>=A&Y<=B H1??Y<=A|Y>=B
#Value=list(type=">=A&<=B",v=c(quantile(data$y,0.1),quantile(data$y,0.9)))
### H0??Y<=A|Y>=B H1??Y>=A&Y<=B
#Value=list(type="<=A|>=B",v=c(quantile(data$y,0.4),quantile(data$y,0.7)))
### H0??Y<=A H1??Y>A
#Value=list(type="<=A",v=quantile(data$y,0.8))

# generate data
data <- data_generation_classication1(N=n)

# generate history data and estimate K (diversity threshold)---
his_data <- data_generation_classication1(N=n)

p <- ncol(his_data)-1 # dimension of covariates

### some data notations, and index for null data-----
datawork=DataSplit(his_data,n,0,n_cal)
data_train=datawork$data_train

data_cal=datawork$data_cal
data_rest=datawork$data_rest

Null_cal=NullIndex(data_cal$y,Value)
Null_rest=NullIndex(data_rest$y,Value)

X_train=as.matrix(data_train[colnames(data_train)[-p-1]])
Y_train=as.matrix(data_train$y)
X_cal=as.matrix(data_cal[colnames(data_cal)[-p-1]])
Y_cal=as.matrix(data_cal$y)

X_rest=as.matrix(data_rest[colnames(data_rest)[-p-1]])
Y_rest=as.matrix(data_rest$y)

data_test=data
Null_test=NullIndex(data_test$y,Value)
Alter_test=setdiff(1:length(data_test$y),Null_test)
X_test=as.matrix(data_test[colnames(data_test)[-p-1]])
Y_test=as.matrix(data_test$y)

X_test_scale=scale(X_test,center = TRUE,scale = TRUE)

# model and estimating locfdr -----

model=fitting(algo,X_train,Y_train,lambda) #estimate model by training data
W_cal=Pred(algo,model,X_cal) #predict classfication score of calibration data
W_test=Pred(algo,model,X_test) #predict classfication score of test data
#estimate local fdr for online(test) data. h1 and h2 are bandwidth for f0 and f. If they
#are 0, bandwidth are automatically selected by density function. If IsSame=TRUE, then h1=h2.
#If IsCali=TRUE, some calibration technique is used for isotonic local fdr curve(mainly for classification)
TN=LocalFDRcompute(W_cal,W_test,Null_cal,algo,h1=0,h2=0,IsSame = TRUE,IsCali = FALSE)
#plot(W_test,TN)#observe the minimum value of localfdr, it should be lower than alpha

### confirm the diversity threshold

#when computing diversity, we should scale each dimension of X
X_cal_scale=scale(X_cal,center = TRUE,scale = TRUE)
X_test_scale=scale(X_test,center = TRUE,scale = TRUE)

X_cal_alter=X_cal_scale[-Null_cal,]
Diversity_Base<-diversity_true_correct_rej(X_cal_alter)
Diversity_threshold<-Diversity_Base*0.4
Diversity_threshold <- 0.045

# a worker function that runs Proposed, SAST, ST, LOND, LORD++, SAFFRON and ADDIS

workerFunc <- function(iter){
  #Generate data
  data <- data_generation_classication1(N=N)
  his_data <- data_generation_classication1(N=n)
  
  ### some data notations, and index for null data-----
  datawork=DataSplit(his_data,n,0,n_cal)
  data_train=datawork$data_train
  
  data_cal=datawork$data_cal
  data_rest=datawork$data_rest
  
  Null_cal=NullIndex(data_cal$y,Value)
  Null_rest=NullIndex(data_rest$y,Value)
  
  X_train=as.matrix(data_train[colnames(data_train)[-p-1]])
  Y_train=as.matrix(data_train$y)
  X_cal=as.matrix(data_cal[colnames(data_cal)[-p-1]])
  Y_cal=as.matrix(data_cal$y)
  
  X_rest=as.matrix(data_rest[colnames(data_rest)[-p-1]])
  Y_rest=as.matrix(data_rest$y)
  
  data_test=data
  Null_test=NullIndex(data_test$y,Value)
  Alter_test=setdiff(1:length(data_test$y),Null_test)
  X_test=as.matrix(data_test[colnames(data_test)[-p-1]])
  Y_test=as.matrix(data_test$y)
  
  
  # model and estimating locfdr -----
  
  model=fitting(algo,X_train,Y_train,lambda) #estimate model by training data
  W_cal=Pred(algo,model,X_cal) #predict classfication score of calibration data
  W_test=Pred(algo,model,X_test) #predict classfication score of test data
  #estimate local fdr for online(test) data. h1 and h2 are bandwidth for f0 and f. If they
  #are 0, bandwidth are automatically selected by density function. If IsSame=TRUE, then h1=h2.
  #If IsCali=TRUE, some calibration technique is used for isotonic local fdr curve(mainly for classification)
  TN=LocalFDRcompute(W_cal,W_test,Null_cal,algo,h1=0,h2=0,IsSame = TRUE,IsCali = FALSE)
  #calculate p-values
  
  pval <- confomalPvalue(W_cal,W_test,Null_cal,Value)
  
  # raw result of Proposed
  res_Proposed.raw <- Online_selection_correct_rej(X_test_scale,Alter_test,TN,alpha,Diversity_threshold,Diversity_initial_num,m,N,IsDiversity=TRUE)
  
  # raw result of SAST
  res_SAST.raw <- Online_selection_correct_rej(X_test_scale,Alter_test,TN,alpha,Diversity_threshold,Diversity_initial_num,m,N,IsDiversity=FALSE)
  
  # raw result of LOND, LORD++, ADDIS
  res_lond.raw <- LOND(pval,alpha)
  
  res_addis.raw <- ADDIS(pval,alpha)
  #sum(res_addis.raw$R)
  res_SAFFRON.raw <- SAFFRON(pval,alpha)
  #sum(res_SAFFRON.raw$R)
  res_naive.raw <- Online_selection_naive(pval,X_test_scale,Alter_test,alpha,m,N)
  
  dec_lond.raw <- res_lond.raw$R
  dec_addis.raw <- res_addis.raw$R
  dec_SAFFRON.raw <- res_SAFFRON.raw$R
  
  # calculate the decisions and the stop time for LOND, LORD++, and ADDIS
  dec_lond <- Decision_compute(dec_lond.raw,m)
  #dec_lord_plus <- Decision_compute(dec_lord_plus.raw,m)
  dec_addis <- Decision_compute(dec_addis.raw,m)
  dec_SAFFRON <- Decision_compute(dec_SAFFRON.raw,m)
  
  # record the stop time for selecting m samples
  t_Proposed <- res_Proposed.raw$stoptime
  t_SAST <- res_SAST.raw$stoptime
  t_naive <- res_naive.raw$stoptime
  
  if(!dec_lond$isdeath){
    t_lond <- dec_lond$stoptime
  }else{t_lond <- dec_lond$deathtime}
  
  
  
  if(!dec_addis$isdeath){
    t_addis <- dec_addis$stoptime
  }else{
    t_addis <- dec_addis$deathtime
  }
  
  if(!dec_SAFFRON$isdeath){
    t_SAFFRON <- dec_SAFFRON$stoptime
  }else{
    t_SAFFRON <- dec_SAFFRON$deathtime
  }
  
  # calculate FSP and DIV use the decision and the truth for data until time t
  x.Real <- data$y
  
  t_Proposed.new <- c(1:t_Proposed)
  t_SAST.new <- c(1:t_SAST)
  t_addis.new<- c(1:t_addis)
  t_naive.new <- c(1:t_naive)
  t_SAFFRON.new <- c(1:t_SAFFRON)
  t_lond.new <- c(1:t_lond)
  
  
  res_Proposed<-  list(FDP=TimeCovert(res_Proposed.raw$FDP,res_Proposed.raw$decisions[1:res_Proposed.raw$stoptime]),
                   DIV=TimeCovert(res_Proposed.raw$DIV,res_Proposed.raw$decisions[1:res_Proposed.raw$stoptime]),
                   Power=TimeCovert(res_Proposed.raw$Power,res_Proposed.raw$decisions[1:res_Proposed.raw$stoptime]))
  res_SAST <- list(FDP=TimeCovert(res_SAST.raw$FDP,res_SAST.raw$decisions[1:res_SAST.raw$stoptime]),
                   DIV=TimeCovert(res_SAST.raw$DIV,res_SAST.raw$decisions[1:res_SAST.raw$stoptime]),
                   Power=TimeCovert(res_SAST.raw$Power,res_SAST.raw$decisions[1:res_SAST.raw$stoptime]))
  res_naive <- list(FDP=TimeCovert(res_naive.raw$FDP,res_naive.raw$decisions[1:res_naive.raw$stoptime]),
                    DIV=TimeCovert(res_naive.raw$DIV,res_naive.raw$decisions[1:res_naive.raw$stoptime]),
                    Power=TimeCovert(res_naive.raw$Power,res_naive.raw$decisions[1:res_naive.raw$stoptime]))
  
  
  res_lond <- t_lond.new %>% map(~CiterionCompute_each(X_test_scale, Alter_test, dec_lond$decisions, .x, x.Real))%>% unlist  %>% split(.,names(.))
  #res_lord_plus <- t_lord_plus.new %>% map(~CiterionCompute_each(X_test_scale, Alter_test, dec_lord_plus$decisions, .x, x.Real)) %>% unlist  %>% split(.,names(.))
  res_addis <- t_addis.new %>% map(~CiterionCompute_each(X_test_scale, Alter_test, dec_addis$decisions, .x, x.Real)) %>% unlist %>% unlist  %>% split(.,names(.))
  
  
  res_SAFFRON <- t_SAFFRON.new %>% map(~CiterionCompute_each(X_test_scale, Alter_test, dec_SAFFRON$decisions, .x, x.Real)) %>% unlist %>% unlist  %>% split(.,names(.))
  
  
  #res_Proposed.stop <- CiterionCompute(X_test_scale,Alter_test,decisions=res_Proposed.raw$decisions[1:t_Proposed],select.num = m)
  #res_SAST.stop <- CiterionCompute(X_test_scale,Alter_test,decisions=res_SAST.raw$decisions[1:t_SAST],select.num = m)
  
  #res_naive.stop <- CiterionCompute(X_test_scale,Alter_test,decisions=res_naive.raw$decisions[1:t_naive],select.num = m)
  
  res_Proposed.stop <- data.frame(FDP=res_Proposed$FDP[t_Proposed],DIV=res_Proposed$DIV[t_Proposed],Power=res_Proposed$Power[t_Proposed])
  res_SAST.stop <- data.frame(FDP=res_SAST$FDP[t_SAST],DIV=res_SAST$DIV[t_SAST],Power=res_SAST$Power[t_SAST])
  res_naive.stop <- data.frame(FDP=res_naive$FDP[t_naive],DIV=res_naive$DIV[t_naive],Power=res_naive$Power[t_naive])
  
  
  res_SAFFRON.stop <- data.frame(FDP=res_SAFFRON$FDP[t_SAFFRON],DIV=res_SAFFRON$DIV[t_SAFFRON],Power=res_SAFFRON$Power[t_SAFFRON])
  
  res_addis.stop <- data.frame(FDP=res_addis$FDP[t_addis],DIV=res_addis$DIV[t_addis],Power=res_addis$Power[t_addis])
  res_lond.stop <- data.frame(FDP=res_lond$FDP[t_lond],DIV=res_lond$DIV[t_lond],Power=res_lond$Power[t_lond])
  
  
  return(list(iter=iter,
              t_Proposed=t_Proposed,t_SAST=t_SAST,t_addis=t_addis,t_naive=t_naive,t_SAFFRON=t_SAFFRON,
              t_lond=t_lond,
              
              
              Proposed_FDP=res_Proposed$FDP,
              Proposed_DIV=res_Proposed$DIV,
              SAST_FDP=res_SAST$FDP,
              SAST_DIV=res_SAST$DIV,
              lond_FDP=res_lond$FDP,
              lond_DIV=res_lond$DIV,
              
              addis_FDP=res_addis$FDP,
              addis_DIV=res_addis$DIV,
              naive_FDP=res_naive$FDP,
              naive_DIV=res_naive$DIV,
              
              SAFFRON_FDP=res_SAFFRON$FDP,
              SAFFRON_DIV=res_SAFFRON$DIV,
              
              lond_FDP.stop=res_lond.stop$FDP,
              lond_DIV.stop=res_lond.stop$DIV,
              lond_Power.stop=res_lond.stop$Power,
              
              
              Proposed_FDP.stop=res_Proposed.stop$FDP,
              Proposed_DIV.stop=res_Proposed.stop$DIV,
              Proposed_Power.stop=res_Proposed.stop$Power,
              
              SAST_FDP.stop=res_SAST.stop$FDP,
              SAST_DIV.stop=res_SAST.stop$DIV,
              SAST_Power.stop=res_SAST.stop$Power,
              
              SAFFRON_FDP.stop=res_SAFFRON.stop$FDP,
              SAFFRON_DIV.stop=res_SAFFRON.stop$DIV,
              SAFFRON_Power.stop=res_SAFFRON.stop$Power,
              
              addis_FDP.stop=res_addis.stop$FDP,
              addis_DIV.stop=res_addis.stop$DIV,
              addis_Power.stop=res_addis.stop$Power,
              
              naive_FDP.stop=res_naive.stop$FDP,
              naive_DIV.stop=res_naive.stop$DIV,
              naive_Power.stop=res_naive.stop$Power
  ))
}

#res <- workerFunc(iter=1)
#attributes(res)

#trails <- seq(1:nrep)
#results <- lapply(trails, workerFunc)
#results1 <- results %>% unlist %>% split(.,names(.))

nrep <- 500
time_Proposed <- time_SAST  <- time_addis <- time_lond <- time_SAFFRON <- time_naive <- rep(NA,nrep)
Proposed_FDP <- SAST_FDP <- addis_FDP <- naive_FDP <- lond_FDP  <- SAFFRON_FDP <- matrix(NA,N,nrep)
Proposed_DIV <- SAST_DIV <- addis_DIV <- naive_DIV <- lond_DIV  <- SAFFRON_DIV <- matrix(NA,N,nrep)

Proposed_FDP.stop <- SAST_FDP.stop <- addis_FDP.stop <- naive_FDP.stop <- lond_FDP.stop  <- SAFFRON_FDP.stop <- rep(NA,nrep)
Proposed_DIV.stop <- SAST_DIV.stop <- addis_DIV.stop <- naive_DIV.stop <- lond_DIV.stop  <- SAFFRON_DIV.stop <- rep(NA,nrep)
Proposed_Power.stop <- SAST_Power.stop <- addis_Power.stop <- naive_Power.stop <- lond_Power.stop  <- SAFFRON_Power.stop <- rep(NA,nrep)


# repeat implement nrep times-----

for(i in 1:nrep){
  res <- workerFunc(iter=1)
  time_Proposed[i] <- res$t_Proposed
  time_SAST[i] <- res$t_SAST
  time_naive[i] <- res$t_naive
  
  time_lond[i] <- res$t_lond
  time_SAFFRON[i] <- res$t_SAFFRON
  time_addis[i] <- res$t_addis
  
  Proposed_FDP[1:res$t_Proposed,i] <- res$Proposed_FDP
  SAST_FDP[1:res$t_SAST,i] <- res$SAST_FDP
  naive_FDP[1:res$t_naive,i] <- res$naive_FDP
  
  lond_FDP[1:res$t_lond,i] <- res$lond_FDP
  SAFFRON_FDP[1:res$t_SAFFRON,i] <- res$SAFFRON_FDP
  addis_FDP[1:res$t_addis,i] <- res$addis_FDP
  
  Proposed_DIV[1:res$t_Proposed,i] <- res$Proposed_DIV
  SAST_DIV[1:res$t_SAST,i] <- res$SAST_DIV
  naive_DIV[1:res$t_naive,i] <- res$naive_DIV
  
  lond_DIV[1:res$t_lond,i] <- res$lond_DIV
  SAFFRON_DIV[1:res$t_SAFFRON,i] <- res$SAFFRON_DIV
  addis_DIV[1:res$t_addis,i] <- res$addis_DIV
  
  Proposed_FDP.stop[i] <- res$Proposed_FDP.stop
  SAST_FDP.stop[i] <- res$SAST_FDP.stop
  naive_FDP.stop[i] <- res$naive_FDP.stop
  
  lond_FDP.stop[i] <- res$lond_FDP.stop
  SAFFRON_FDP.stop[i] <- res$SAFFRON_FDP.stop
  addis_FDP.stop[i] <- res$addis_FDP.stop
  
  
  Proposed_DIV.stop[i] <- res$Proposed_DIV.stop
  SAST_DIV.stop[i] <- res$SAST_DIV.stop
  naive_DIV.stop[i] <- res$naive_DIV.stop
  
  lond_DIV.stop[i] <- res$lond_DIV.stop
  SAFFRON_DIV.stop[i] <- res$SAFFRON_DIV.stop
  addis_DIV.stop[i] <- res$addis_DIV.stop
  
  Proposed_Power.stop[i] <- res$Proposed_Power.stop
  SAST_Power.stop[i] <- res$SAST_Power.stop
  naive_Power.stop[i] <- res$naive_Power.stop
  lond_Power.stop[i] <- res$lond_Power.stop
  SAFFRON_Power.stop[i] <- res$SAFFRON_Power.stop
  addis_Power.stop[i] <- res$addis_Power.stop
}
})

print(execution_time)

# tidy the results when stop------
FDP.stop <- as.data.frame(cbind(Proposed_FDP.stop,SAST_FDP.stop,lond_FDP.stop,SAFFRON_FDP.stop,addis_FDP.stop))
DIV.stop <- as.data.frame(cbind(Proposed_DIV.stop,SAST_DIV.stop,lond_DIV.stop,SAFFRON_DIV.stop,addis_DIV.stop))
Power.stop <- as.data.frame(cbind(Proposed_Power.stop,SAST_Power.stop,lond_Power.stop,SAFFRON_Power.stop,addis_Power.stop))

summary(DIV.stop)
summary(FDP.stop)
result_stop <- as.data.frame(cbind(FDP.stop,DIV.stop,Power.stop))
summary(result_stop)
result_stop.ave <- colMeans(result_stop,na.rm = TRUE)

stop_time <- as.data.frame(cbind(time_Proposed,time_SAST,time_lond,time_SAFFRON,time_addis))
stop_time.ave <- colMeans(stop_time,na.rm = TRUE)

summary(stop_time)

names(FDP.stop) <- c('II-COS','SAST','LOND','SAFFRON','ADDIS')
names(DIV.stop) <- c('II-COS','SAST','LOND','SAFFRON','ADDIS')
head(FDP.stop)
head(DIV.stop)



Diversity_threshold <- 0.045
alpha <- 0.1

level <- as.data.frame(cbind(Diversity_threshold,alpha))
level


#  plots the results when stop and save the results--------

FDP_stop_value <- melt(FDP.stop)
names(FDP_stop_value) <- c('Method','FSR')

p_FSR <- ggplot(data=FDP_stop_value,aes(x=Method,y=FSR,color=Method))
p_FSR <- p_FSR+geom_boxplot() +
  stat_summary(mapping=aes(group=Method,fill=Method),fun="mean",
               geom="point",shape=23,size=2.5,
               position=position_dodge(0.8))+
  theme_bw() +
  labs(y = TeX("$C_1(FSR)$"))+
  xlab(NULL)+
  scale_fill_nejm(palette = c("default"))+
  scale_colour_nejm(palette = c("default"))+guides(color=FALSE,fill=FALSE)+
  theme(axis.text = element_text(size = 16), axis.title = element_text(size = 20),
        axis.text.x = element_text(angle=45, hjust=.5, vjust=.5),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank(),
        legend.position="bottom")
p_FSR <- p_FSR + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")+theme(text=element_text(size=16,  family="serif"))
p_FSR



ES_stop_value <- melt(DIV.stop)
names(ES_stop_value) <- c('Method','ES')
p_ES <- ggplot(data=ES_stop_value,aes(x=Method,y=ES,color=Method))
p_ES <- p_ES+geom_boxplot()+
  stat_summary(mapping=aes(group=Method,fill=Method),fun="mean",
               geom="point",shape=23,size=2.5,
               position=position_dodge(0.8))+
  theme_bw() +
  labs(y = TeX("$\\tilde{C}_2(ES)$"))+
  xlab(NULL)+
  scale_fill_nejm(palette = c("default"))+
  scale_colour_nejm(palette = c("default"))+guides(color=FALSE,fill=FALSE)+
  theme(axis.text = element_text(size = 16), axis.title = element_text(size = 20),
        axis.text.x = element_text(angle=45, hjust=.5, vjust=.5),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())
p_ES <- p_ES + geom_hline(aes(yintercept=Diversity_threshold), colour="black", linetype="dashed")+theme(text=element_text(size=16,  family="serif"))
p_ES


names(stop_time) <- c('II-COS','SAST','LOND','SAFFRON','ADDIS')


stop_time_value <- melt(stop_time)
names(stop_time_value) <- c('Method','Stoptime')


p_Time <- ggplot(data=stop_time_value,aes(x=Method,y=Stoptime,color=Method))
p_Time <- p_Time+geom_boxplot() +
  stat_summary(mapping=aes(group=Method,fill=Method),fun="mean",
               geom="point",shape=23,size=2.5,
               position=position_dodge(0.8))+
  theme_bw() +
  labs(y = TeX("$T_{m}$"))+
  xlab(NULL)+
  scale_fill_nejm(palette = c("default"))+
  scale_colour_nejm(palette = c("default"))+guides(color=FALSE,fill=FALSE)

p_Time <- p_Time+theme(axis.text = element_text(size = 16),
                       axis.title = element_text(size = 18),
                       axis.text.x = element_text(angle=45, hjust=.5, vjust=.5),
                       panel.grid.major=element_line(colour=NA),
                       panel.background = element_rect(fill = "transparent",colour = NA),
                       plot.background = element_rect(fill = "transparent",colour = NA),
                       panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif"))
p_Time

plotcla_stop <- ggarrange(p_FSR, p_ES, p_Time,ncol=3, nrow=1, common.legend = FALSE, legend = 'none',
                          font.label = list(size = 20, face = "bold"))
pdf(file = "plotcla_stop.pdf",width = 12,height = 5) 
plotcla_stop
dev.off()



# tidy the results at every time t--------

t <- seq(30,300,10)



tidy_res.each <- function(t, Method_FDP, Method_DIV){
  na_cols_FDP <- which(colSums(is.na(Method_FDP[t,])) > 0)
  na_cols_DIV <- which(colSums(is.na(Method_DIV[t,])) > 0)
  FDP_Method.ave <- rowMeans(Method_FDP[t,],na.rm = TRUE)
  DIV_Method.ave <- rowMeans(Method_DIV[t,],na.rm = TRUE)
  if(sum(na_cols_FDP!=0)){
    FDP_Method.se <- apply(Method_FDP[t,-na_cols_FDP], 1, function(x) sd(x) / sqrt(length(x)))
  }else{
    FDP_Method.se <- apply(Method_FDP[t,], 1, function(x) sd(x) / sqrt(length(x)))
  }
  if(sum(na_cols_DIV!=0)){
    DIV_Method.se <- apply(Method_DIV[t,-na_cols_DIV], 1, function(x) sd(x) / sqrt(length(x)))
  }else{
    DIV_Method.se <- apply(Method_DIV[t,], 1, function(x) sd(x) / sqrt(length(x)))
  }
  return(data.frame(FDP.ave=FDP_Method.ave,DIV.ave=DIV_Method.ave,
                    FDP.se=FDP_Method.se,DIV.se=DIV_Method.se)
  )
}


tidy_res.each(t,lond_FDP,lond_DIV)

result_each <- as.data.frame(cbind(tidy_res.each(t,Proposed_FDP,Proposed_DIV),tidy_res.each(t,SAST_FDP,SAST_DIV),
                                   tidy_res.each(t,lond_FDP,lond_DIV),tidy_res.each(t,SAFFRON_FDP,SAFFRON_DIV),
                                   tidy_res.each(t,addis_FDP,addis_DIV)))
dim(result_each)
head(result_each)




result_FSR.ave <- result_each[,c(1,5,9,13,17)]
result_ES.ave <- result_each[,c(2,6,10,14,18)]
result_FSR.ave$time <- t
result_ES.ave$time <- t
names(result_FSR.ave) <- c('II-COS','SAST','LOND','SAFFRON','ADDIS','time')
names(result_ES.ave) <- c('II-COS','SAST','LOND','SAFFRON','ADDIS','time')
head(result_FSR.ave)
head(result_ES.ave)

result_FSR.se <- result_each[,c(3,7,11,15,19)]
result_ES.se <- result_each[,c(4,8,12,16,20)]
result_FSR.se$time <- t
result_ES.se$time <- t
names(result_FSR.se) <- c('II-COS','SAST','LOND','SAFFRON','ADDIS','time')
names(result_ES.se) <- c('II-COS','SAST','LOND','SAFFRON','ADDIS','time')
head(result_FSR.se)
head(result_ES.se)


# plots the results at every time t--------


data_FSR.ave <- melt(result_FSR.ave,id="time")
data_FSR.se <- melt(result_FSR.se,id="time")

colnames(data_FSR.ave) <- c("Time","Method","Value.mean")
dim(data_FSR.ave)
colnames(data_FSR.se) <- c("Time","Method","Value.se")
dim(data_FSR.se)
data_FSR <- as.data.frame(cbind(data_FSR.ave,data_FSR.se[,3]))
colnames(data_FSR) <- c("Time","Method","Value.mean","Value.se")
dim(data_FSR)
head(data_FSR)



# plot the line charts and save the results

alpha=0.1
Diversity_threshold=0.045
p1 <- ggplot(data = data_FSR,aes(x=Time,y=Value.mean,group =Method,color=Method,shape=Method,fill=Method))+
  geom_point(size=2.0)+geom_ribbon(aes(ymin = Value.mean-Value.se,ymax = Value.mean+Value.se),
                                   alpha = 0.1,
                                   linetype = 1,
                                   color=NA)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab("Time")+
  ylab(TeX("$C_1(FSR)$"))+
  ylim(0,0.2)+
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 0.8)+
  scale_fill_manual(values=c("#BC3C29FF","#0072B5FF","#E18727FF","#20854EFF","#7876B1FF"))+
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif"))

p1 <- p1 + geom_hline(aes(yintercept=alpha), colour="black", linetype="dashed")

p1




data_ES.ave <- melt(result_ES.ave,id="time")
data_ES.se <- melt(result_ES.se,id="time")

colnames(data_ES.ave) <- c("Time","Method","Value.mean")
dim(data_ES.ave)
colnames(data_ES.se) <- c("Time","Method","Value.se")
dim(data_ES.se)
data_ES <- as.data.frame(cbind(data_ES.ave,data_ES.se[,3]))
colnames(data_ES) <- c("Time","Method","Value.mean","Value.se")
dim(data_ES)
head(data_ES)


p2 <- ggplot(data = data_ES,aes(x=Time,y=Value.mean,group=Method,shape=Method,color=Method,fill=Method))+
  geom_point(size=2.5)+geom_ribbon(aes(ymin = Value.mean-Value.se,ymax = Value.mean+Value.se,color=Method),
                                   alpha = 0.1,
                                   linetype = 1,
                                   color = NA)+
  geom_line(aes(linetype=Method,color=Method),linewidth=0.8)+
  xlab("Time")+
  ylab(TeX("$\\tilde{C}_2(ES)$"))+
  theme_bw() +scale_color_nejm(palette = c("default"), alpha = 1)+
  scale_fill_manual(values=c("#BC3C29FF","#0072B5FF","#E18727FF","#20854EFF","#7876B1FF"))+
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 16),
        legend.title = element_text(size = 16),
        panel.grid.major=element_line(colour=NA),
        panel.background = element_rect(fill = "transparent",colour = NA),
        plot.background = element_rect(fill = "transparent",colour = NA),
        panel.grid.minor = element_blank())+theme(text=element_text(size=16,  family="serif"))

p2 <- p2 + geom_hline(aes(yintercept=Diversity_threshold), colour="black", linetype="dashed")

p2


plot_cla <- ggarrange(p1, p2, ncol=2, nrow=1, common.legend = TRUE, legend="bottom", 
                      font.label = list(size = 20, face = "bold"))
pdf(file = "plot_cla.pdf",width = 10,height = 4) 
plot_cla
dev.off()

