
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

load("Census-results.RData")

##education

edudata=data.frame()
for (res in reslist) {
  if(dim(res$Proposed_X)[1]!=0){
    aa=count(res$Proposed_X,education.num)
  aa["prop"]=aa$n/sum(aa$n)
  aa["method"]="II-COS"
  aa=melt(aa,measure.vars="education.num",value.name = "education")
  edudata=rbind(edudata,aa)
  }
  
  
  if(dim(res$SAST_X[1]!=0)){
   aa=count(res$SAST_X,education.num)
  aa["prop"]=aa$n/sum(aa$n)
  aa["method"]="SAST"
  aa=melt(aa,measure.vars="education.num",value.name = "education")
  edudata=rbind(edudata,aa) 
  }
  
  
  #aa=count(res$naive_X,education.num)
  #aa["prop"]=aa$n/sum(aa$n)
  #aa["method"]="ST"
  #aa=melt(aa,measure.vars="education.num",value.name = "education")
  #edudata=rbind(edudata,aa)
  
}

pp5=edudata%>%
  group_by(education,method)%>%
  dplyr::summarize(n=mean(n),prop_mean=mean(prop),prop_sd=sd(prop,na.rm = TRUE)/sqrt(nrep))
pp5

#pp5$education=factor(pp5$education,levels = c("No Qualification", "High School Diploma", "Matriculation" ,"Bachelors", "Masters"))

pp5["label"]=round(pp5$prop_mean,2)
#pp5$label[pp5$method=="SAST"]=NA
pp5$label=NA
#pp5$label[pp5$method=="II-COS"][c(6,9,11,12)]=c(0.09,0.35, 0.17, 0.11)
#pp5$label[pp5$method=="SAST"][c(6,9,11,12)]=c(0.07,0.45,0.13,0.07)
pp5$label[pp5$method=="II-COS"][9]=c(0.35)
pp5$label[pp5$method=="SAST"][9]=c(0.45)
P3<-ggplot(data = pp5, aes(x = education, y = prop_mean,fill=method)) +
  geom_bar(stat = "identity",position = position_dodge(0.9),alpha=1)+
  geom_errorbar(aes(ymin=prop_mean-prop_sd,ymax=prop_mean+prop_sd),width=0.1,position = position_dodge(0.9))+
  scale_fill_nejm(palette = c("default"),name="Method")+
  scale_y_continuous(name = "Propotion")+
  scale_x_continuous(name = "Education Length") +
  geom_text(mapping = aes(label = pp5$label,y=prop_mean+0.04),size = 3.7, colour = 'black', vjust = 1, hjust = .5, position = position_dodge(0.9))+
  theme_classic()+
  theme(legend.position="top",legend.text=element_text(size=17),legend.title=element_text(size=17),axis.text.y=element_text(size=13),
        axis.text.x = element_text(size=13,angle = 0,hjust=0.5),axis.title.x = element_text(size=15),axis.title.y = element_text(size=15))
P3

ggsave('income.pdf',width=7,height=4)
pdf('income.pdf',width=7,height=4)
P3
dev.off()




pp5$n[pp5$method=="SAST"]=-pp5$n[pp5$method=="SAST"]
pp5$age=as.numeric(pp5$education)
pp5$method=factor(pp5$method,levels=c("Proposed","SAST"))

P1=pp5%>%ggplot(aes(x = education, y = n, fill = method)) +
  geom_col()+scale_y_continuous(name = "Number")+ theme_bw()+
  scale_x_continuous(name = "Education Length") 
P1


###sex
sexdata=data.frame()
for (res in reslist) {
  if(dim(res$Proposed_X)[1]!=0){
    aa=count(res$Proposed_X,sex)
    aa["prop"]=aa$n/sum(aa$n)
    aa["method"]="Proposed"
    aa=melt(aa,measure.vars="sex",value.name = "sex")
    sexdata=rbind(sexdata,aa)
  }
  
  
  if(dim(res$SAST_X[1]!=0)){
    aa=count(res$SAST_X,sex)
    aa["prop"]=aa$n/sum(aa$n)
    aa["method"]="SAST"
    aa=melt(aa,measure.vars="sex",value.name = "sex")
    sexdata=rbind(sexdata,aa) 
  }
  
  
  aa=count(res$naive_X,sex)
  aa["prop"]=aa$n/sum(aa$n)
  aa["method"]="ST"
  aa=melt(aa,measure.vars="sex",value.name = "sex")
  sexdata=rbind(sexdata,aa)
  
}


pp4=sexdata%>%
  group_by(sex,method)%>%
  dplyr::summarize(n=mean(n),prop=mean(prop))
pp4

P4<-ggplot(data = pp4, aes(x = sex, y = prop,fill=method)) +
  geom_bar(stat = "identity",position = position_dodge(0.9),alpha=0.8)+
  scale_fill_manual(values=c("#E31A1C","#1F78B4","#B2DF8A"))+
  scale_y_continuous(name = "Porpotion")+
  scale_x_discrete(name = "Gender") +theme_bw()+
  theme(legend.position="top",
        axis.text.x = element_text(size=11,angle = 0,hjust=0.5))
P4

PP=ggarrange(P4,P3,common.legend = TRUE)
ggsave('income.pdf',width=7,height=4)
pdf('income.pdf',width=7,height=4)
PP
dev.off()
### race

racedata=data.frame()
for (res in reslist) {
  if(dim(res$Proposed_X)[1]!=0){
    aa=count(res$Proposed_X,race)
    aa["prop"]=aa$n/sum(aa$n)
    aa["method"]="Proposed"
    aa=melt(aa,measure.vars="race",value.name = "race")
    racedata=rbind(racedata,aa)
  }
  
  
  if(dim(res$SAST_X[1]!=0)){
    aa=count(res$SAST_X,race)
    aa["prop"]=aa$n/sum(aa$n)
    aa["method"]="SAST"
    aa=melt(aa,measure.vars="race",value.name = "race")
    racedata=rbind(racedata,aa) 
  }
  
  
  aa=count(res$naive_X,race)
  aa["prop"]=aa$n/sum(aa$n)
  aa["method"]="ST"
  aa=melt(aa,measure.vars="race",value.name = "race")
  racedata=rbind(racedata,aa)
  
}


pp2=racedata%>%
  group_by(race,method)%>%
  dplyr::summarize(n=mean(n),prop=1-mean(prop))
pp2

P2<-ggplot(data = pp2, aes(x = race, y = prop,fill=method)) +
  geom_bar(stat = "identity",position = position_dodge(0.9),alpha=0.8)+
  scale_fill_manual(values=c("#E31A1C","#1F78B4","#B2DF8A"))+
  scale_y_continuous(name = "Porpotion")+
  scale_x_discrete(name = "race") +theme_bw()+
  theme(legend.position="top",
        axis.text.x = element_text(size=11,angle = 0,hjust=0.5))
P2


### age

agedata=data.frame()
for (res in reslist) {
  if(dim(res$Proposed_X)[1]!=0){
    aa=count(res$Proposed_X,age)
    aa["prop"]=aa$n/sum(aa$n)
    aa["method"]="Proposed"
    aa=melt(aa,measure.vars="age",value.name = "age")
    agedata=rbind(agedata,aa)
  }
  
  
  if(dim(res$SAST_X[1]!=0)){
    aa=count(res$SAST_X,age)
    aa["prop"]=aa$n/sum(aa$n)
    aa["method"]="SAST"
    aa=melt(aa,measure.vars="age",value.name = "age")
    agedata=rbind(agedata,aa) 
  }
  
  
  #aa=count(res$naive_X,sex)
  #aa["prop"]=aa$n/sum(aa$n)
  #aa["method"]="ST"
  #aa=melt(aa,measure.vars="sex",value.name = "sex")
  #sexdata=rbind(sexdata,aa)
  
}


pp1=agedata%>%
  group_by(age,method)%>%
  dplyr::summarize(n=mean(n),prop=mean(prop))
pp1


pp1$n[pp1$method=="SAST"]=-pp1$n[pp1$method=="SAST"]
pp1$age=as.numeric(pp1$age)
pp1$method=factor(pp1$method,levels=c("Proposed","SAST"))

P1=pp1%>%ggplot(aes(x = age, y = n, fill = method)) +
  geom_col()+scale_y_continuous(name = "Number")+ theme_bw()+
  scale_x_continuous(name = "Age") 
P1


P1<-ggplot(data = pp1, aes(x = age, y = prop,fill=method)) +
  geom_bar(stat = "identity",position = position_dodge(0.9),alpha=0.8)+
  scale_fill_manual(values=c("#E31A1C","#1F78B4","#B2DF8A"))+
  scale_y_continuous(name = "Porpotion")+
  scale_x_continuous(name = "age") +theme_bw()+
  theme(legend.position="top",
        axis.text.x = element_text(size=11,angle = 0,hjust=0.5))
P1