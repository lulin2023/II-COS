
# Candidate Selection Analysis

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
library(grid)
library(gridExtra)
library(magrittr)
library(latex2exp)
library(ggsci)


setwd(dirname(rstudioapi::getSourceEditorContext()$path))

source("functions_OnSel.R")  
source("algoclass_OnSel.R")  ### implementation for different algorithms,such as SVM, RF


# import the Candidate Selection data----

data <- read.csv("train.csv")


summary(data)


# 数据预处???----
sum(is.na(data))

data1 <- na.omit(data)[,-c(1,2,5,8)]
table(data$is_pass)





varcontinue <- c("program_duration","age","total_programs_enrolled","trainee_engagement_rating")  
#连续变量转化为数值型并与因子型变量合???
colname=colnames(data1)


data2 <- cbind(lapply(data1[,varcontinue],function(x) as.numeric(as.character(x))),as.data.frame(lapply(data1[,setdiff(colname,c(varcontinue,"is_pass"))],function(x) factor(x))))
summary(data2)




dummy <- dummyVars(" ~ .", data=data2[,-length(colname)])

#perform one-hot encoding on data frame
data3 <- data.frame(predict(dummy, newdata=data2))

data3$y=data1$is_pass



# simulation setting-----
alpha <- 0.2 # significance level

pi <- 0.2 # Bernoulli(pi)
n <- 2000 # number of historical data
N <- 8000 # number of total time points
m <- 100 # number of selections when stop
n_train<- round(n/2) # number of data used for training model
n_cal<- n-n_train #number of data used for estimating locfdr
Diversity_constant<-1 #determine the diversity threshold
Diversity_initial_num<-1 #when rejection number exceeds this, we consider computing diversity.
algo<- new("RFc") #algorithm used for classification or regression
lambda<- 500 #specific parameter for the algorithm


colnames(data3)
# draw n history sample 
id <- c(1:nrow(data3))
his_id <- sample(id,n)
his_data <- data3[his_id,]
id1 <- id[-his_id]

id2 <- sample(id1,N)
data_test <- data3[id2,]
p <- ncol(data3)-1 # dimension of covariates
# Random forest--------

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

Null_test=NullIndex(data_test$y,Value)
Alter_test=setdiff(1:length(data_test$y),Null_test)
X_test=as.matrix(data_test[colnames(data_test)[-p-1]])
Y_test=as.matrix(data_test$y)


# model and estimating locfdr -----
colnames(X_train)
model=fitting(algo,X_train,Y_train,lambda) #estimate model by training data
W_cal=Pred(algo,model,X_cal) #predict classfication score of calibration data
W_test=Pred(algo,model,X_test) #predict classfication score of test data
#estimate local fdr for online(test) data. h1 and h2 are bandwidth for f0 and f. If they 
#are 0, bandwidth are automatically selected by density function. If IsSame=TRUE, then h1=h2.
#If IsCali=TRUE, some calibration technique is used for isotonic local fdr curve(mainly for classification)
TN=LocalFDRcompute(W_cal,W_test,Null_cal,algo,h1=0,h2=0,IsSame = TRUE,IsCali = TRUE)
#plot(W_test,TN)#observe the minimum value of localfdr, it should be lower than alpha

### confirm the diversity threshold

#when computing diversity, we should scale each dimension of X


colnames(data)

X_cal_scale=scale(X_cal,center = TRUE,scale = TRUE)
X_cal_scale[which(is.na(X_cal_scale))]=0
X_cal_scale[,5:11]=X_cal_scale[,5:11]/7
X_cal_scale[,12:13]=X_cal_scale[,12:13]/2
X_cal_scale[,14:17]=X_cal_scale[,14:17]/4
X_cal_scale[,18:19]=X_cal_scale[,18:19]/2
X_cal_scale[,20:24]=X_cal_scale[,20:24]/5
X_cal_scale[,25:28]=X_cal_scale[,25:28]/4

X_test_scale=scale(X_test,center = TRUE,scale = TRUE)
X_test_scale[which(is.na(X_test_scale))]=0
X_test_scale[,5:11]=X_test_scale[,5:11]/7
X_test_scale[,12:13]=X_test_scale[,12:13]/2
X_test_scale[,14:17]=X_test_scale[,14:17]/4
X_test_scale[,18:19]=X_test_scale[,18:19]/2
X_test_scale[,20:24]=X_test_scale[,20:24]/5
X_test_scale[,25:28]=X_test_scale[,25:28]/4
X_test_scale[,]

X_cal_alter=X_cal_scale[-Null_cal,]
Diversity_Base<-diversity_true_correct_rej(X_cal_alter)
Diversity_threshold<-0.001#Diversity_Base*0.3


workerFunc <- function(iter){
  
  # draw n history sample as his_data and draw N sample as data
  id <- c(1:nrow(data3))
  his_id <- sample(id,n)
  his_data <- data3[his_id,]
  id1 <- id[-his_id]
  
  id2 <- sample(id1,N)
  data_test <- data3[id2,]
  p <- ncol(data3)-1 
  
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
  
  Null_test=NullIndex(data_test$y,Value)
  Alter_test=setdiff(1:length(data_test$y),Null_test)
  X_test=as.matrix(data_test[colnames(data_test)[-p-1]])
  Y_test=as.matrix(data_test$y)
  
  X_test_scale=scale(X_test,center = TRUE,scale = TRUE)
  X_test_scale[which(is.na(X_test_scale))]=0
  X_test_scale[,5:11]=X_test_scale[,5:11]/7
  X_test_scale[,12:13]=X_test_scale[,12:13]/2
  X_test_scale[,14:17]=X_test_scale[,14:17]/4
  X_test_scale[,18:19]=X_test_scale[,18:19]/2
  X_test_scale[,20:24]=X_test_scale[,20:24]/5
  X_test_scale[,25:28]=X_test_scale[,25:28]/4
  X_test_scale[,]
  # model and estimating locfdr -----
  
  model=fitting(algo,X_train,Y_train,lambda) #estimate model by training data
  W_cal=Pred(algo,model,X_cal) #predict classfication score of calibration data
  W_test=Pred(algo,model,X_test) #predict classfication score of test data
  #estimate local fdr for online(test) data. h1 and h2 are bandwidth for f0 and f. If they 
  #are 0, bandwidth are automatically selected by density function. If IsSame=TRUE, then h1=h2.
  #If IsCali=TRUE, some calibration technique is used for isotonic local fdr curve(mainly for classification)
  TN=LocalFDRcompute(W_cal,W_test,Null_cal,algo,h1=0,h2=0,IsSame = TRUE,IsCali = TRUE)
  plot(W_test,TN)
  #calculate p-values
  
  pval <- confomalPvalue(W_cal,W_test,Null_cal,Value)
  
  # raw result of Proposed
  res_Proposed.raw <- Online_selection_correct_rej(X_test_scale,Alter_test,TN,alpha,Diversity_threshold,Diversity_initial_num,m,N,IsDiversity=TRUE)
  
  # raw result of SAST
  res_SAST.raw <- Online_selection_correct_rej(X_test_scale,Alter_test,TN,alpha,Diversity_threshold,Diversity_initial_num,m,N,IsDiversity=FALSE)
  
  # raw result of LOND, LORD++, ADDIS 
  #sum(res_lond.raw$R)
  #res_lord_plus.raw <- LORD(pval,alpha)
  
  #sum(res_SAFFRON.raw$R)
  res_naive.raw <- Online_selection_naive(pval,X_test_scale,Alter_test,alpha,m,N)
  
  
  
  
  # record the stop time for selecting m samples
  t_Proposed <- res_Proposed.raw$stoptime
  t_SAST <- res_SAST.raw$stoptime
  t_naive <- res_naive.raw$stoptime
  
  
  
  # calculate FSP and DIV use the decision and the truth for data until time t
  x.Real <- data$y
  
  t_Proposed.new <- c(1:t_Proposed)
  t_SAST.new <- c(1:t_SAST)
  t_naive.new <- c(1:t_naive)
  
  #t_lord_plus.new <- c(1:t_lord_plus)
  
  res_Proposed<-  list(FDP=TimeCovert(res_Proposed.raw$FDP,res_Proposed.raw$decisions[1:res_Proposed.raw$stoptime]),
                   DIV=TimeCovert(res_Proposed.raw$DIV,res_Proposed.raw$decisions[1:res_Proposed.raw$stoptime]),
                   Power=TimeCovert(res_Proposed.raw$Power,res_Proposed.raw$decisions[1:res_Proposed.raw$stoptime]))
  res_SAST <- list(FDP=TimeCovert(res_SAST.raw$FDP,res_SAST.raw$decisions[1:res_SAST.raw$stoptime]),
                   DIV=TimeCovert(res_SAST.raw$DIV,res_SAST.raw$decisions[1:res_SAST.raw$stoptime]),
                   Power=TimeCovert(res_SAST.raw$Power,res_SAST.raw$decisions[1:res_SAST.raw$stoptime]))
  res_naive <- list(FDP=TimeCovert(res_naive.raw$FDP,res_naive.raw$decisions[1:res_naive.raw$stoptime]),
                    DIV=TimeCovert(res_naive.raw$DIV,res_naive.raw$decisions[1:res_naive.raw$stoptime]),
                    Power=TimeCovert(res_naive.raw$Power,res_naive.raw$decisions[1:res_naive.raw$stoptime]))
  
  
  
  res_Proposed.stop <- data.frame(FDP=res_Proposed$FDP[t_Proposed],DIV=res_Proposed$DIV[t_Proposed],Power=res_Proposed$Power[t_Proposed])
  res_SAST.stop <- data.frame(FDP=res_SAST$FDP[t_SAST],DIV=res_SAST$DIV[t_SAST],Power=res_SAST$Power[t_SAST])
  res_naive.stop <- data.frame(FDP=res_naive$FDP[t_naive],DIV=res_naive$DIV[t_naive],Power=res_naive$Power[t_naive])
  
  #res_lord_plus.stop <- data.frame(FDP=res_lord_plus$FDP[t_lord_plus],DIV=res_lord_plus$DIV[t_lord_plus],Power=res_lord_plus$Power[t_lord_plus])
  
  #tidy the final results
  res_Proposed_final <- list(method='Proposed',t_stop=t_Proposed,FDP=res_Proposed$FDP,DIV=res_Proposed$DIV,FDP.stop=res_Proposed.stop$FDP,DIV.stop=res_Proposed.stop$DIV)
  res_SAST_final <- list(method='SAST',t_stop=t_SAST,FDP=res_SAST$FDP,DIV=res_SAST$DIV,FDP.stop=res_SAST.stop$FDP,DIV.stop=res_SAST.stop$DIV)
  res_ST_final <- list(method='ST',t_stop=t_naive,FDP=res_naive$FDP,DIV=res_naive$DIV,FDP.stop=res_naive.stop$FDP,DIV.stop=res_naive.stop$DIV)
  
  
  ClassData=data1[id2,]
  Proposed_X=ClassData[intersect(which(res_Proposed.raw$decisions==1),Alter_test),]
  SAST_X=ClassData[intersect(which(res_SAST.raw$decisions==1),Alter_test),]
  naive_X=ClassData[intersect(which(res_naive.raw$decisions==1),Alter_test),]
  #  return(list(iter=iter,
  #              result_Proposed=res_Proposed_final,
  #              result_SAST=res_SAST_final,
  #              result_ST=res_ST_final,
  #              result_LOND=res_LOND_final,
  #              result_LORD_plus=res_LORD_plus_final,
  #             result_SAFFRON=res_SAFFRON_final,
  #              result_ADDIS=res_ADDIS_final
  #  ))
  return(list(iter=iter,
              t_Proposed=t_Proposed,t_SAST=t_SAST,t_naive=t_naive,
              Proposed_FDP=res_Proposed$FDP,
              Proposed_DIV=res_Proposed$DIV,
              SAST_FDP=res_SAST$FDP,
              SAST_DIV=res_SAST$DIV,
              
              naive_FDP=res_naive$FDP,
              naive_DIV=res_naive$DIV,
              
              
              Proposed_FDP.stop=res_Proposed.stop$FDP,
              Proposed_DIV.stop=res_Proposed.stop$DIV,
              Proposed_Power.stop=res_Proposed.stop$Power,
              
              SAST_FDP.stop=res_SAST.stop$FDP,
              SAST_DIV.stop=res_SAST.stop$DIV,
              SAST_Power.stop=res_SAST.stop$Power,
              
              
              naive_FDP.stop=res_naive.stop$FDP,
              naive_DIV.stop=res_naive.stop$DIV,
              naive_Power.stop=res_naive.stop$Power,
              
              Proposed_X=Proposed_X,
              SAST_X=SAST_X,
              naive_X=naive_X
  ))
}

#res <- workerFunc(iter=1) 
#attributes(res)

#trails <- seq(1:nrep)
#results <- lapply(trails, workerFunc)
#results1 <- results %>% unlist %>% split(.,names(.))

nrep <- 500
time_Proposed <- time_SAST  <- time_addis <- time_lond <- time_lord_plus <- time_SAFFRON <- time_naive <- rep(NA,nrep)
Proposed_FDP <- SAST_FDP <- addis_FDP <- naive_FDP <- lond_FDP <- lord_plus_FDP <- SAFFRON_FDP <- matrix(NA,N,nrep)
Proposed_DIV <- SAST_DIV <- addis_DIV <- naive_DIV <- lond_DIV <- lord_plus_DIV <- SAFFRON_DIV <- matrix(NA,N,nrep)

Proposed_FDP.stop <- SAST_FDP.stop <- addis_FDP.stop <- naive_FDP.stop <- lond_FDP.stop <- lord_plus_FDP.stop <- SAFFRON_FDP.stop <- rep(NA,nrep)
Proposed_DIV.stop <- SAST_DIV.stop <- addis_DIV.stop <- naive_DIV.stop <- lond_DIV.stop <- lord_plus_DIV.stop <- SAFFRON_DIV.stop <- rep(NA,nrep)
Proposed_Power.stop <- SAST_Power.stop <- addis_Power.stop <- naive_Power.stop <- lond_Power.stop <- lord_plus_Power.stop <- SAFFRON_Power.stop <- rep(NA,nrep)

reslist<-list()
# repeat implement nrep times-----

for(i in 1:nrep){
  res <- workerFunc(iter=1)
  time_Proposed[i] <- res$t_Proposed
  time_SAST[i] <- res$t_SAST
  time_naive[i] <- res$t_naive
  
  
  reslist=append(reslist,list(res=res))
  Proposed_FDP[1:res$t_Proposed,i] <- res$Proposed_FDP
  SAST_FDP[1:res$t_SAST,i] <- res$SAST_FDP
  naive_FDP[1:res$t_naive,i] <- res$naive_FDP
  
  
  
  Proposed_DIV[1:res$t_Proposed,i] <- res$Proposed_DIV
  SAST_DIV[1:res$t_SAST,i] <- res$SAST_DIV
  naive_DIV[1:res$t_naive,i] <- res$naive_DIV
  
  
  
  Proposed_FDP.stop[i] <- res$Proposed_FDP.stop
  SAST_FDP.stop[i] <- res$SAST_FDP.stop
  naive_FDP.stop[i] <- res$naive_FDP.stop
  
  
  
  Proposed_DIV.stop[i] <- res$Proposed_DIV.stop
  SAST_DIV.stop[i] <- res$SAST_DIV.stop
  naive_DIV.stop[i] <- res$naive_DIV.stop
  
  
  
  Proposed_Power.stop[i] <- res$Proposed_Power.stop
  SAST_Power.stop[i] <- res$SAST_Power.stop
  naive_Power.stop[i] <- res$naive_Power.stop
  
}

# tidy the results when stop------
FDP.stop <- as.data.frame(cbind(Proposed_FDP.stop,SAST_FDP.stop,naive_FDP.stop))
DIV.stop <- as.data.frame(cbind(Proposed_DIV.stop,SAST_DIV.stop,naive_DIV.stop))
Power.stop <- as.data.frame(cbind(Proposed_Power.stop,SAST_Power.stop,naive_Power.stop))
result_stop <- as.data.frame(cbind(FDP.stop,DIV.stop,Power.stop))
head(result_stop)
result_stop.ave <- colMeans(result_stop,na.rm = TRUE)

stop_time <- as.data.frame(cbind(time_Proposed,time_SAST,time_naive))
stop_time.ave <- colMeans(stop_time,na.rm = TRUE)

names(FDP.stop) <- c('Proposed','SAST','ST')
names(DIV.stop) <- c('Proposed','SAST','ST')
names(Power.stop) <- c('Proposed','SAST','ST')



### output table results and figure

#table results
apply(stop_time, 2, mean)
apply(FDP.stop, 2, mean)
apply(DIV.stop, 2, mean)*1000

apply(stop_time, 2, sd)/sqrt(nrep)
apply(FDP.stop, 2, sd)/sqrt(nrep)
apply(DIV.stop, 2, sd)*1000/sqrt(nrep)

## female proportion in table
genderdata=data.frame()
for (res in reslist) {
  aa=count(res$Proposed_X,gender)
  aa["prop"]=aa$n/sum(aa$n)
  genderdata=rbind(genderdata,list(
    class="F",n=aa$n[1],prop=aa$prop[1],method="Proposed"
  ))
  genderdata=rbind(genderdata,list(
    class="N",n=aa$n[2],prop=aa$prop[2],method="Proposed"
  ))
  aa=count(res$SAST_X,gender)
  aa["prop"]=aa$n/sum(aa$n)
  genderdata=rbind(genderdata,list(
    class="F",n=aa$n[1],prop=aa$prop[1],method="SAST"
  ))
  genderdata=rbind(genderdata,list(
    class="N",n=aa$n[2],prop=aa$prop[2],method="SAST"
  ))
  aa=count(res$naive_X,gender)
  aa["prop"]=aa$n/sum(aa$n)
  genderdata=rbind(genderdata,list(
    class="F",n=aa$n[1],prop=aa$prop[1],method="naive"
  ))
  genderdata=rbind(genderdata,list(
    class="N",n=aa$n[2],prop=aa$prop[2],method="naive"
  ))
}

genderdata[is.na(genderdata)]=0

pp2=genderdata%>%
  group_by(class,method)%>%
  dplyr::summarize(n=mean(n),prop_mean=mean(prop),prop_sd=sd(prop)/sqrt(nrep))
pp2[1:3,c(1,2,4,5)]


## handicapated porpotion in table
handdata=data.frame()
for (res in reslist) {
  aa=count(res$Proposed_X,is_handicapped)
  aa["prop"]=aa$n/sum(aa$n)
  handdata=rbind(handdata,list(
    class="N",n=aa$n[1],prop=aa$prop[1],method="Proposed"
  ))
  handdata=rbind(handdata,list(
    class="Y",n=aa$n[2],prop=aa$prop[2],method="Proposed"
  ))
  aa=count(res$SAST_X,is_handicapped)
  aa["prop"]=aa$n/sum(aa$n)
  handdata=rbind(handdata,list(
    class="N",n=aa$n[1],prop=aa$prop[1],method="SAST"
  ))
  handdata=rbind(handdata,list(
    class="Y",n=aa$n[2],prop=aa$prop[2],method="SAST"
  ))
  aa=count(res$naive_X,is_handicapped)
  aa["prop"]=aa$n/sum(aa$n)
  handdata=rbind(handdata,list(
    class="N",n=aa$n[1],prop=aa$prop[1],method="ST"
  ))
  handdata=rbind(handdata,list(
    class="Y",n=aa$n[2],prop=aa$prop[2],method="ST"
  ))
}

handdata[is.na(handdata)]=0
pp3=handdata%>%
  group_by(class,method)%>%
  dplyr::summarize(n=mean(n),prop_mean=mean(prop),prop_sd=sd(prop)/sqrt(nrep))
pp3[4:6,c(1,2,4,5)]
