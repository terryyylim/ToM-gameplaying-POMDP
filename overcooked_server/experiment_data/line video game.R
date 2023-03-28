###
#exp 1 pilot: yes, maybe, no cases; 1, some, 8 cut
####

setwd("~/Dropbox/Research Projects/Postdoc/Lines/lines video game/pilot1_data")
subj9<-read.csv("subj9.csv")
subj11<-read.csv("subj11.csv")
subj28<-read.csv("subj28.csv")
subj37<-read.csv("subj37.csv")
subj57<-read.csv("subj57.csv")
subj70<-read.csv("subj70.csv")
subj76<-read.csv("subj76.csv")

data<-rbind(subj9,subj11,subj28,subj37,subj57,subj70,subj76)

#by trial_category
data.sum <- data %>% 
  group_by(trial_category) %>%
  summarize(
    n = length(response),
    mean = mean(response, na.rm = T),
    se = sqrt(mean*(1-mean)/n),
    error = qnorm(0.975)*se, 
    CI.left = mean-error,
    CI.right = mean+error
  )

data.sum %>%
  ggplot(aes(y=mean, x=trial_category)) + 
  geom_bar(position="dodge", stat="identity")+
  geom_errorbar(aes(ymin=CI.left, ymax=CI.right), width=.2,position=position_dodge(.9))+
  coord_cartesian(ylim = c(0,1))
  #labs(title="Novel Rule Study \nIs it OK or not OK to cannonball into the pool?", 
       #x="Context", y = "Mean Judgment")


#by individual video
data.sum2 <- data %>% 
  group_by(stimulus_name) %>%
  summarize(
    n = length(response),
    mean = mean(response, na.rm = T),
    se = sqrt(mean*(1-mean)/n),
    error = qnorm(0.975)*se, 
    CI.left = mean-error,
    CI.right = mean+error
  )

data.sum2 %>%
  ggplot(aes(y=mean, x=stimulus_name)) + 
  geom_bar(position="dodge", stat="identity")+
  geom_errorbar(aes(ymin=CI.left, ymax=CI.right), width=.2,position=position_dodge(.9))+
  coord_cartesian(ylim = c(0,1))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

###
#universalization pilot
####

setwd("~/Dropbox/Research Projects/Postdoc/Lines/lines video game")
univ<-read.csv("univ-pilot.csv")
univ$trial_category<-factor(univ$trial_category)
univ$stimulus_name<-factor(univ$stimulus_name)

#by trial_category
univ.sum <- univ %>% 
  group_by(trial_category) %>%
  summarize(
    n = length(response),
    mean = mean(response, na.rm = T),
    sd = sd(response, na.rm=T),
    se = sd/sqrt(n)
  )

univ.sum %>%
  ggplot(aes(y=mean, x=trial_category)) + 
  geom_bar(position="dodge", stat="identity")+
  geom_errorbar(aes(ymin=mean-se, ymax=mean+se), width=.2,position=position_dodge(.9))+
  coord_cartesian(ylim = c(-50,50))
#labs(title="Novel Rule Study \nIs it OK or not OK to cannonball into the pool?", 
#x="Context", y = "Mean Judgment")


#by individual video
univ.sum2 <- univ %>% 
  group_by(stimulus_name,trial_category) %>%
  summarize(
    n = length(response),
    mean = mean(response, na.rm = T),
    sd = sd(response, na.rm=T),
    se = sd/sqrt(n)
  )

#graph shows that we get the full range of graded judgments
univ.sum2 %>%
  ggplot(aes(y=mean, x=reorder(stimulus_name,mean))) + 
  geom_bar(position="dodge", stat="identity")+
  geom_errorbar(aes(ymin=mean-se, ymax=mean+se), width=.2,position=position_dodge(.9))+
  coord_cartesian(ylim = c(-50,50))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

#without maybes
univ.sum2 %>%
  filter(trial_category!="maybe_1")%>%
  ggplot(aes(y=mean, x=reorder(stimulus_name,mean))) + 
  geom_bar(position="dodge", stat="identity")+
  geom_errorbar(aes(ymin=mean-se, ymax=mean+se), width=.2,position=position_dodge(.9))+
  coord_cartesian(ylim = c(-50,50))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

#histograms show that distributions are what we would expect
hist(univ$response[univ$trial_category=="maybe_1"], xlim=c(-50,50), main="maybe line")
hist(univ$response[univ$trial_category=="yes_1"], xlim=c(-50,50), main = "yes line")
hist(univ$response[univ$trial_category=="no_1"], xlim=c(-50,50), main = "no line")

hist(univ$response[univ$stimulus_name=="video/maybe_line_new_1_1cut.mp4"], xlim=c(-50,50), breaks=10, main = "maybe new-1")
hist(univ$response[univ$stimulus_name=="video/maybe_line_new_3_1cut.mp4"], xlim=c(-50,50), breaks=10, main = "maybe new-1")

####
#exp 1 "final"
#just yes and no cases (maybes removed)
#1, some, 8 people leaving line
####

exp1<-read.csv("exp1.csv")
exp1<-exp1[c(1,15,16,17)] #also available: trail number, rt, demographic data
exp1<-exp1[exp1$response==1 | exp1$response==0,]
exp1$run_id<-as.integer(exp1$run_id)
exp1$response<-as.integer(exp1$response)
exp1$trial_category<-factor(exp1$trial_category)
exp1$stimulus_name<-factor(exp1$stimulus_name)
exp1$trial_category2[exp1$trial_category=="no_1"]<-"no"
exp1$trial_category2[exp1$trial_category=="no_34"]<-"no"
exp1$trial_category2[exp1$trial_category=="no_8"]<-"no"
exp1$trial_category2[exp1$trial_category=="yes_1"]<-"yes"
exp1$trial_category2[exp1$trial_category=="yes_34"]<-"yes"
exp1$trial_category2[exp1$trial_category=="yes_8"]<-"yes"
exp1$trial_category2<-factor(exp1$trial_category2)
                          

#by trial_category
exp1.sum <- exp1 %>% 
  group_by(trial_category) %>%
  summarize(
    n = length(response),
    mean = mean(response, na.rm = T),
    se = sqrt(mean*(1-mean)/n),
    error = qnorm(0.975)*se, 
    CI.left = mean-error,
    CI.right = mean+error
  )

exp1.sum %>%
  ggplot(aes(y=mean, x=trial_category)) + 
  geom_bar(position="dodge", stat="identity")+
  geom_errorbar(aes(ymin=CI.left, ymax=CI.right), width=.2,position=position_dodge(.9))+
  coord_cartesian(ylim = c(0,1))
#labs(title="Novel Rule Study \nIs it OK or not OK to cannonball into the pool?", 
#x="Context", y = "Mean Judgment")


#by individual video
exp1.sum2 <- exp1 %>% 
  group_by(stimulus_name) %>%
  summarize(
    n = length(response),
    mean = mean(response, na.rm = T),
    se = sqrt(mean*(1-mean)/n),
    error = qnorm(0.975)*se, 
    CI.left = mean-error,
    CI.right = mean+error
  )

exp1.sum2 %>%
  ggplot(aes(y=mean, x=stimulus_name)) + 
  geom_bar(position="dodge", stat="identity")+
  geom_errorbar(aes(ymin=CI.left, ymax=CI.right), width=.2,position=position_dodge(.9))+
  coord_cartesian(ylim = c(0,1))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

#by trial_category2 (just yes/no)
exp1.sum3 <- exp1 %>% 
  group_by(trial_category2) %>%
  summarize(
    n = length(response),
    mean = mean(response, na.rm = T),
    se = sqrt(mean*(1-mean)/n),
    error = qnorm(0.975)*se, 
    CI.left = mean-error,
    CI.right = mean+error
  )

exp1.sum3 %>%
  ggplot(aes(y=mean, x=trial_category2)) + 
  geom_bar(position="dodge", stat="identity")+
  geom_errorbar(aes(ymin=CI.left, ymax=CI.right), width=.2,position=position_dodge(.9))+
  coord_cartesian(ylim = c(0,1))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


#####
#exp 2 final
#lock-in mechanism
#just 4 cases: yes line_0 cut, yes line_1 cut, no line_0, no line_1
#compare to same cases with no lock-in
####

exp2<-read.csv("exp2.csv")
exp2<-exp2[c(1,15,16,17)] #also available: trail number, rt, demographic data
exp2<-exp2[exp2$response==1 | exp2$response==0,]
exp2$run_id<-as.integer(exp2$run_id)
exp2$response<-as.integer(exp2$response)
exp2$trial_category<-factor(exp2$trial_category)
exp2$stimulus_name<-factor(exp2$stimulus_name)


#by trial_category
exp2.sum <- exp2 %>% 
  group_by(trial_category) %>%
  summarize(
    n = length(response),
    mean = mean(response, na.rm = T),
    se = sqrt(mean*(1-mean)/n),
    error = qnorm(0.975)*se, 
    CI.left = mean-error,
    CI.right = mean+error
  )

exp2.sum %>%
  ggplot(aes(y=mean, x=trial_category)) + 
  geom_bar(position="dodge", stat="identity")+
  geom_errorbar(aes(ymin=CI.left, ymax=CI.right), width=.2,position=position_dodge(.9))+
  coord_cartesian(ylim = c(0,1))+
  labs(title="Lock-in Version")

#get no-lock-in data 
#n=52 (missing some cases)

nolock<-read.csv("exp2-part2.csv")
nolock<-nolock[c(1,15,16,17)] #also available: trail number, rt, demographic data
nolock<-nolock[nolock$response==1 | nolock$response==0,]
nolock$run_id<-as.integer(nolock$run_id)
nolock$response<-as.integer(nolock$response)
nolock$trial_category<-factor(nolock$trial_category)
nolock$stimulus_name<-factor(nolock$stimulus_name)

#by trial_category
nolock.sum <- nolock %>% 
  group_by(trial_category) %>%
  summarize(
    n = length(response),
    mean = mean(response, na.rm = T),
    se = sqrt(mean*(1-mean)/n),
    error = qnorm(0.975)*se, 
    CI.left = mean-error,
    CI.right = mean+error
  )

nolock.sum %>%
  ggplot(aes(y=mean, x=trial_category)) + 
  geom_bar(position="dodge", stat="identity")+
  geom_errorbar(aes(ymin=CI.left, ymax=CI.right), width=.2,position=position_dodge(.9))+
  coord_cartesian(ylim = c(0,1))+
  labs(title="No Lock-in Version")


######
#exp 3: yes, maybe, no cases; 1 cut only
#looking for graded permissibility judgments
#####

exp3<-read.csv("exp3.csv")
exp3<-exp3[c(1,15,16,17)] #also available: trail number, rt, demographic data
exp3<-exp3[exp3$response==1 | exp3$response==0,]
exp3$run_id<-as.integer(exp3$run_id)
exp3$response<-as.integer(exp3$response)
exp3$trial_category<-factor(exp3$trial_category)
exp3$stimulus_name<-factor(exp3$stimulus_name)


#by trial_category
exp3.sum <- exp3 %>% 
  group_by(trial_category) %>%
  summarize(
    n = length(response),
    mean = mean(response, na.rm = T),
    se = sqrt(mean*(1-mean)/n),
    error = qnorm(0.975)*se, 
    CI.left = mean-error,
    CI.right = mean+error
  )

exp3.sum %>%
  ggplot(aes(y=mean, x=trial_category)) + 
  geom_bar(position="dodge", stat="identity")+
  geom_errorbar(aes(ymin=CI.left, ymax=CI.right), width=.2,position=position_dodge(.9))+
  coord_cartesian(ylim = c(0,1))+
  labs(title="Exp 3")

#by individual video
exp3.sum2 <- exp3 %>% 
  group_by(stimulus_name) %>%
  summarize(
    n = length(response),
    mean = mean(response, na.rm = T),
    se = sqrt(mean*(1-mean)/n),
    error = qnorm(0.975)*se, 
    CI.left = mean-error,
    CI.right = mean+error
  )

exp3.sum2 %>%
  ggplot(aes(y=mean, x=reorder(stimulus_name,mean))) + 
  geom_bar(position="dodge", stat="identity")+
  geom_errorbar(aes(ymin=CI.left, ymax=CI.right), width=.2,position=position_dodge(.9))+
  coord_cartesian(ylim = c(0,1))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

#universalization data
univ<-read.csv("exp3-part2.csv")
univ<-univ[c(1,15,16,19)] #also available: trail number, rt, demographic data
univ$response<-as.numeric(univ$response)
univ<-univ[complete.cases(univ),]
univ$run_id<-as.integer(univ$run_id)
univ$response<-as.integer(univ$response)
univ$trial_category<-factor(univ$trial_category)
univ$stimulus_name<-factor(univ$stimulus_name)

#by trial_category
univ.sum <- univ %>% 
  group_by(trial_category) %>%
  summarize(
    n = length(response),
    mean = mean(response, na.rm = T),
    sd = sd(response, na.rm=T),
    se = sd/sqrt(n)
  )

univ.sum %>% 
  ggplot(aes(y=mean, x=trial_category)) + 
  geom_bar(position="dodge", stat="identity")+
  geom_errorbar(aes(ymin=mean-se, ymax=mean+se), width=.2,position=position_dodge(.9))+
  coord_cartesian(ylim = c(-50,50))
#labs(title="Novel Rule Study \nIs it OK or not OK to cannonball into the pool?", 
#x="Context", y = "Mean Judgment")


#by individual video
univ.sum2 <- univ %>% 
  group_by(stimulus_name,trial_category) %>%
  summarize(
    n = length(response),
    mean = mean(response, na.rm = T),
    sd = sd(response, na.rm=T),
    se = sd/sqrt(n)
  )

#graph shows that we get the full range of graded judgments
univ.sum2 %>%
  ggplot(aes(y=mean, x=reorder(stimulus_name,mean))) + 
  geom_bar(position="dodge", stat="identity")+
  geom_errorbar(aes(ymin=mean-se, ymax=mean+se), width=.2,position=position_dodge(.9))+
  coord_cartesian(ylim = c(-50,50))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

#histograms show that distributions are what we would expect
hist(univ$response[univ$trial_category=="maybe_1"], xlim=c(-50,50), main="maybe line")
hist(univ$response[univ$trial_category=="yes_1"], xlim=c(-50,50), main = "yes line")
hist(univ$response[univ$trial_category=="no_1"], xlim=c(-50,50), main = "no line")

plot(exp3.sum2$mean,univ.sum2$mean)
cor(exp3.sum2$mean,univ.sum2$mean) #.84

means<-data.frame(exp3.sum2$mean)
means<-cbind(means,univ.sum2$mean)
means<-cbind(means,exp3.sum2$stimulus_name)
names(means)[1]<-"judgment"
names(means)[2]<-"univ"
names(means)[3]<-"stimulus_name"

library(ggrepel)

means %>% 
  ggplot(aes(y=univ, x=judgment)) + 
  geom_point()+
  geom_smooth(method = lm, level=.95)+
  coord_cartesian(ylim = c(-50,50))+
  theme(legend.position="right")+
  xlab("Judgment")+
  ylab("Universalization")+
  geom_text_repel(aes(label = stimulus_name), size = 3)


  
  
#geom_jitter(width=.2, height=.1)
#scale_color_manual(name = "Condition", 
# labels = c("High Interest","Low Interest"),
#values=c("blue","red"))+
#ggtitle("4-7 Threshold \nSubject Moral Acceptability Data and Models")+
