
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

load("Candidate results.RData")





##education

edudata=data.frame()
for (res in reslist) {
  aa=count(res$Proposed_X,education)
  aa["prop"]=aa$n/sum(aa$n)
  aa["method"]="Proposed"
  aa=melt(aa,measure.vars="education",value.name = "education")
  edudata=rbind(edudata,aa)
  
  
  aa=count(res$SAST_X,education)
  aa["prop"]=aa$n/sum(aa$n)
  aa["method"]="SAST"
  aa=melt(aa,measure.vars="education",value.name = "education")
  edudata=rbind(edudata,aa)
  
  aa=count(res$naive_X,education)
  aa["prop"]=aa$n/sum(aa$n)
  aa["method"]="CP"
  aa=melt(aa,measure.vars="education",value.name = "education")
  edudata=rbind(edudata,aa)
  
}

pp5=edudata%>%
  group_by(education,method)%>%
  dplyr::summarize(n=mean(n),prop_mean=mean(prop),prop_sd=sd(prop)/sqrt(nrep))
pp5

pp5$method[pp5$method=="Proposed"]="II-COS"

pp5$method=factor(pp5$method,levels=c("II-COS","SAST","CP"))


pp5$education=factor(pp5$education,levels = c("No Qualification", "High School Diploma", "Matriculation" ,"Bachelors", "Masters"))

P3<-ggplot(data = pp5, aes(x = education, y = prop_mean,fill=method)) +
  geom_bar(stat = "identity",position = position_dodge(0.9),alpha=1)+
  geom_errorbar(aes(ymin=prop_mean-prop_sd,ymax=prop_mean+prop_sd),width=0.1,position = position_dodge(0.9))+
  #scale_fill_manual(values=c("#E31A1C","#1F78B4","#B2DF8A"))+
  scale_fill_nejm(palette = c("default"),name="Method")+
  scale_y_continuous(name = "Propotion")+
  scale_x_discrete(name = "Education Status",labels=c("No Qual","High School","Matriculation","Bachelors","Master")) +#labels=c("NQ","HS","Mc","Ba","Ma")) +
  geom_text(mapping = aes(label = round(prop_mean,2),y=prop_mean+0.03),size = 3.7, colour = 'black', vjust = 1, hjust = .5, position = position_dodge(0.9))+
  theme_classic()+
  theme(legend.position="top",legend.text=element_text(size=17),legend.title=element_text(size=17),axis.text.y=element_text(size=15),
        axis.text.x = element_text(size=15,angle = 0,hjust=0.5),axis.title.x = element_text(size=15),axis.title.y = element_text(size=15))
P3


ggsave('candidate.pdf',width=7,height=4)
pdf('candidate.pdf',width=7,height=4)
P3
dev.off()
